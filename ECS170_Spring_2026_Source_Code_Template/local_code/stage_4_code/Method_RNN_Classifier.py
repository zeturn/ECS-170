from local_code.base_class.method import method
import os
import time
import copy
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt


class RecurrentSentimentNet(nn.Module):
    def __init__(self, vocab_size, cell_type='lstm', embed_dim=128, hidden_dim=128,
                 num_layers=1, dropout=0.35, bidirectional=True):
        super().__init__()
        self.cell_type = cell_type.lower()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        rnn_dropout = dropout if num_layers > 1 else 0.0
        rnn_cls = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[self.cell_type]
        self.rnn = rnn_cls(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
            bidirectional=bidirectional,
        )
        direction_factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * direction_factor, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, lengths=None):
        embedded = self.dropout(self.embedding(x))
        if lengths is not None:
            packed = pack_padded_sequence(
                embedded,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            output, hidden = self.rnn(packed)
        else:
            output, hidden = self.rnn(embedded)
        if self.cell_type == 'lstm':
            hidden = hidden[0]
        if self.rnn.bidirectional:
            features = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            features = hidden[-1]
        return self.classifier(features)


class Method_RNN_Classifier(method):
    data = None
    history_destination_folder_path = '../../result/stage_4_result/classification/'

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        self.cell_type = 'lstm'
        self.vocab_size = None
        self.max_epoch = 8
        self.learning_rate = 0.001
        self.batch_size = 256
        self.embed_dim = 128
        self.hidden_dim = 256
        self.num_layers = 1
        self.dropout = 0.35
        self.bidirectional = True
        self.weight_decay = 1e-5
        self.progress_every = 25
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.history = []
        self.model = None

    def _make_loaders(self):
        X_train = torch.LongTensor(np.asarray(self.data['train']['X'], dtype=np.int64))
        y_train = torch.LongTensor(np.asarray(self.data['train']['y'], dtype=np.int64))
        X_test = torch.LongTensor(np.asarray(self.data['test']['X'], dtype=np.int64))
        y_test = torch.LongTensor(np.asarray(self.data['test']['y'], dtype=np.int64))
        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            TensorDataset(X_test, y_test),
            batch_size=self.batch_size,
            shuffle=False,
        )
        return train_loader, test_loader

    def _evaluate(self, loader, loss_function):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                lengths = (X != 0).sum(dim=1).clamp(min=1)
                logits = self.model(X, lengths)
                loss = loss_function(logits, y)
                total_loss += loss.item() * X.shape[0]
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += X.shape[0]
        return total_loss / total, correct / total

    def train_model(self):
        if self.vocab_size is None:
            self.vocab_size = int(self.data.get('vocab_size', 0))
        if not self.vocab_size:
            self.vocab_size = int(np.max(self.data['train']['X'])) + 1

        print('device:', self.device)
        if self.device.type == 'cuda':
            print('gpu:', torch.cuda.get_device_name(0))
        print('model:', self.cell_type.upper())
        print('vocab size:', self.vocab_size)
        print('sequence length:', self.data['train']['X'].shape[1])

        train_loader, test_loader = self._make_loaders()
        self.model = RecurrentSentimentNet(
            vocab_size=self.vocab_size,
            cell_type=self.cell_type,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_function = nn.CrossEntropyLoss()
        self.history = []

        best_acc = 0.0
        best_state = None
        best_epoch = 0
        for epoch in range(1, self.max_epoch + 1):
            start_time = time.time()
            self.model.train()
            total_loss, correct, total = 0.0, 0, 0
            num_batches = len(train_loader)
            print(f'[{self.cell_type.upper()}] Epoch {epoch}/{self.max_epoch} started ({num_batches} batches)')
            for batch_idx, (X, y) in enumerate(train_loader, start=1):
                X, y = X.to(self.device), y.to(self.device)
                lengths = (X != 0).sum(dim=1).clamp(min=1)
                optimizer.zero_grad()
                logits = self.model(X, lengths)
                loss = loss_function(logits, y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()

                total_loss += loss.item() * X.shape[0]
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += X.shape[0]

                if batch_idx == 1 or batch_idx % self.progress_every == 0 or batch_idx == num_batches:
                    running_loss = total_loss / total
                    running_acc = correct / total
                    pct = 100.0 * batch_idx / num_batches
                    print(
                        f'[{self.cell_type.upper()}] Epoch {epoch}/{self.max_epoch} '
                        f'batch {batch_idx:03d}/{num_batches} ({pct:5.1f}%) '
                        f'loss={running_loss:.4f} acc={running_acc:.4f}',
                        flush=True,
                    )

            train_loss = total_loss / total
            train_acc = correct / total
            test_loss, test_acc = self._evaluate(test_loader, loss_function)
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
            row = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'test_loss': test_loss,
                'test_accuracy': test_acc,
            }
            self.history.append(row)
            elapsed = time.time() - start_time
            print(
                f'Epoch {epoch:02d}: '
                f'train_loss={train_loss:.4f} train_acc={train_acc:.4f} '
                f'test_loss={test_loss:.4f} test_acc={test_acc:.4f} '
                f'best_test_acc={best_acc:.4f} best_epoch={best_epoch} time={elapsed:.1f}s'
            )

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f'restored best {self.cell_type.upper()} checkpoint from epoch {best_epoch} '
                  f'with test_acc={best_acc:.4f}')
        self._save_history()
        return best_acc

    def test(self, X):
        X = torch.LongTensor(np.asarray(X, dtype=np.int64))
        loader = DataLoader(TensorDataset(X), batch_size=self.batch_size, shuffle=False)
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for (batch_X,) in loader:
                batch_X = batch_X.to(self.device)
                lengths = (batch_X != 0).sum(dim=1).clamp(min=1)
                logits = self.model(batch_X, lengths)
                predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        return np.array(predictions)

    def run(self):
        print('--start training...')
        self.train_model()
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        true_y = np.asarray(self.data['test']['y'], dtype=np.int64)
        return {'pred_y': pred_y, 'true_y': true_y, 'history': self.history}

    def _save_history(self):
        os.makedirs(self.history_destination_folder_path, exist_ok=True)
        prefix = self.cell_type.lower()
        csv_path = os.path.join(self.history_destination_folder_path, f'{prefix}_learning_curve.csv')
        with open(csv_path, 'w') as f:
            f.write('epoch,train_loss,train_accuracy,test_loss,test_accuracy\n')
            for row in self.history:
                f.write(
                    f"{row['epoch']},{row['train_loss']},{row['train_accuracy']},"
                    f"{row['test_loss']},{row['test_accuracy']}\n"
                )
        print('saved learning curve data to:', csv_path)

        epochs = [row['epoch'] for row in self.history]
        plt.figure()
        plt.plot(epochs, [row['train_loss'] for row in self.history], label='Train Loss')
        plt.plot(epochs, [row['test_loss'] for row in self.history], label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{self.cell_type.upper()} Classification Loss')
        plt.legend()
        plt.savefig(os.path.join(self.history_destination_folder_path, f'{prefix}_loss.png'), bbox_inches='tight')
        plt.close()

        plt.figure()
        plt.plot(epochs, [row['train_accuracy'] for row in self.history], label='Train Accuracy')
        plt.plot(epochs, [row['test_accuracy'] for row in self.history], label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'{self.cell_type.upper()} Classification Accuracy')
        plt.legend()
        plt.savefig(os.path.join(self.history_destination_folder_path, f'{prefix}_accuracy.png'), bbox_inches='tight')
        plt.close()
        print('saved learning curve plots to:', self.history_destination_folder_path)
