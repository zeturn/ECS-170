import copy
import csv
import os
import random
import re
import time
from collections import Counter

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[.,!?;:]")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class WordLanguageModel(nn.Module):
    def __init__(self, vocab_size, cell_type='gru', embed_dim=128, hidden_dim=256,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.cell_type = cell_type.lower()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        rnn_cls = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[self.cell_type]
        self.rnn = rnn_cls(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.rnn(embedded, hidden)
        logits = self.fc(self.dropout(output[:, -1, :]))
        return logits, hidden


class TextGenerationExperiment:
    def __init__(self, data_path, output_dir, cell_type='gru', sequence_length=8,
                 max_vocab_size=5000, embed_dim=128, hidden_dim=256, num_layers=2,
                 dropout=0.3, batch_size=256, learning_rate=0.001, max_epoch=60,
                 seed=2):
        self.data_path = data_path
        self.output_dir = output_dir
        self.cell_type = cell_type.lower()
        self.sequence_length = sequence_length
        self.max_vocab_size = max_vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.seed = seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.word_to_id = {}
        self.id_to_word = {}
        self.history = []
        self.model = None

    def load_jokes(self):
        jokes = []
        with open(self.data_path, newline='', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                joke = row.get('Joke', '').strip()
                tokens = tokenize(joke)
                if len(tokens) >= self.sequence_length + 2:
                    jokes.append(tokens + ['<eos>'])
        return jokes

    def build_vocab(self, jokes):
        counter = Counter(token for joke in jokes for token in joke)
        vocab = ['<pad>', '<unk>', '<eos>'] + [
            w for w, _ in counter.most_common(self.max_vocab_size - 3) if w != '<eos>'
        ]
        self.word_to_id = {w: i for i, w in enumerate(vocab)}
        self.id_to_word = {i: w for w, i in self.word_to_id.items()}

    def encode(self, tokens):
        unk = self.word_to_id['<unk>']
        return [self.word_to_id.get(token, unk) for token in tokens]

    def make_dataset(self, jokes):
        X, y = [], []
        for joke in jokes:
            ids = self.encode(joke)
            for i in range(0, len(ids) - self.sequence_length):
                X.append(ids[i:i + self.sequence_length])
                y.append(ids[i + self.sequence_length])
        return np.asarray(X, dtype=np.int64), np.asarray(y, dtype=np.int64)

    def train(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        jokes = self.load_jokes()
        self.build_vocab(jokes)
        X, y = self.make_dataset(jokes)
        loader = DataLoader(
            TensorDataset(torch.LongTensor(X), torch.LongTensor(y)),
            batch_size=self.batch_size,
            shuffle=True,
        )

        print('device:', self.device)
        if self.device.type == 'cuda':
            print('gpu:', torch.cuda.get_device_name(0))
        print('generation model:', self.cell_type.upper())
        print('jokes:', len(jokes))
        print('training windows:', len(y))
        print('vocab size:', len(self.word_to_id))
        print('sequence length:', self.sequence_length)

        self.model = WordLanguageModel(
            vocab_size=len(self.word_to_id),
            cell_type=self.cell_type,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        loss_function = nn.CrossEntropyLoss()
        best_loss = float('inf')
        best_state = None

        self.history = []
        for epoch in range(1, self.max_epoch + 1):
            start_time = time.time()
            self.model.train()
            total_loss, total = 0.0, 0
            num_batches = len(loader)
            print(f'[{self.cell_type.upper()}-GEN] Epoch {epoch}/{self.max_epoch} started ({num_batches} batches)')

            for batch_idx, (batch_X, batch_y) in enumerate(loader, start=1):
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                logits, _ = self.model(batch_X)
                loss = loss_function(logits, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()

                total_loss += loss.item() * batch_X.shape[0]
                total += batch_X.shape[0]
                if batch_idx == 1 or batch_idx % 25 == 0 or batch_idx == num_batches:
                    running_loss = total_loss / total
                    pct = 100.0 * batch_idx / num_batches
                    print(
                        f'[{self.cell_type.upper()}-GEN] Epoch {epoch}/{self.max_epoch} '
                        f'batch {batch_idx:03d}/{num_batches} ({pct:5.1f}%) '
                        f'loss={running_loss:.4f}',
                        flush=True,
                    )

            epoch_loss = total_loss / total
            self.history.append({'epoch': epoch, 'train_loss': epoch_loss})
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = copy.deepcopy(self.model.state_dict())
            print(
                f'Epoch {epoch:02d}: train_loss={epoch_loss:.4f} '
                f'best_loss={best_loss:.4f} time={time.time() - start_time:.1f}s'
            )

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f'restored best {self.cell_type.upper()} generation checkpoint with loss={best_loss:.4f}')

    def generate(self, start_words, max_new_words=30, temperature=0.8, top_k=20):
        self.model.eval()
        tokens = tokenize(start_words)
        if len(tokens) < 3:
            raise ValueError('Please provide at least three starting words.')
        generated = tokens[:]
        unk = self.word_to_id['<unk>']

        with torch.no_grad():
            for _ in range(max_new_words):
                context = generated[-self.sequence_length:]
                ids = [self.word_to_id.get(token, unk) for token in context]
                if len(ids) < self.sequence_length:
                    ids = [0] * (self.sequence_length - len(ids)) + ids
                x = torch.LongTensor([ids]).to(self.device)
                logits, _ = self.model(x)
                logits = logits[0] / temperature
                if top_k:
                    values, indices = torch.topk(logits, k=min(top_k, logits.numel()))
                    probs = torch.softmax(values, dim=0)
                    next_id = indices[torch.multinomial(probs, 1).item()].item()
                else:
                    probs = torch.softmax(logits, dim=0)
                    next_id = torch.multinomial(probs, 1).item()
                next_word = self.id_to_word.get(next_id, '<unk>')
                if next_word == '<eos>':
                    break
                if next_word in ['<pad>', '<unk>']:
                    continue
                generated.append(next_word)
                if next_word in ['.', '!', '?'] and len(generated) >= len(tokens) + 8:
                    break
        return self.detokenize(generated)

    @staticmethod
    def detokenize(tokens):
        text = ' '.join(tokens)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = text.replace(" n't", "n't").replace(" 's", "'s").replace(" 're", "'re")
        return text[:1].upper() + text[1:]

    def save_outputs(self, start_prompts):
        os.makedirs(self.output_dir, exist_ok=True)
        csv_path = os.path.join(self.output_dir, f'{self.cell_type}_generation_learning_curve.csv')
        with open(csv_path, 'w') as f:
            f.write('epoch,train_loss\n')
            for row in self.history:
                f.write(f"{row['epoch']},{row['train_loss']}\n")

        epochs = [row['epoch'] for row in self.history]
        losses = [row['train_loss'] for row in self.history]
        plt.figure()
        plt.plot(epochs, losses)
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title(f'{self.cell_type.upper()} Text Generation Loss')
        plt.savefig(os.path.join(self.output_dir, f'{self.cell_type}_generation_loss.png'), bbox_inches='tight')
        plt.close()

        samples_path = os.path.join(self.output_dir, f'{self.cell_type}_generated_samples.txt')
        with open(samples_path, 'w', encoding='utf-8') as f:
            for prompt in start_prompts:
                sample = self.generate(prompt)
                f.write(f'Prompt: {prompt}\n')
                f.write(f'Generated: {sample}\n\n')
                print('Prompt:', prompt)
                print('Generated:', sample)
        print('saved generation curve to:', csv_path)
        print('saved generation samples to:', samples_path)
