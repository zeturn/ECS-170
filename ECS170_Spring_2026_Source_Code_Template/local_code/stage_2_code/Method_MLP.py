'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class Method_MLP(method, nn.Module):
    data = None
    max_epoch = 200
    learning_rate = 1e-3
    batch_size = 64
    hidden_dims = [128, 64]
    dropout = 0.2
    optimizer_name = 'adam'
    loss_name = 'cross_entropy'
    weight_decay = 0.0
    verbose_step = 20
    learning_curve = None
    label_to_index = None
    index_to_label = None

    def __init__(
        self,
        mName,
        mDescription,
        hidden_dims=None,
        max_epoch=200,
        learning_rate=1e-3,
        batch_size=64,
        dropout=0.2,
        optimizer_name='adam',
        loss_name='cross_entropy',
        weight_decay=0.0,
        verbose_step=20
    ):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        if hidden_dims is not None:
            self.hidden_dims = hidden_dims
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout = dropout
        self.optimizer_name = optimizer_name
        self.loss_name = loss_name
        self.weight_decay = weight_decay
        self.verbose_step = verbose_step
        self.learning_curve = {'epoch': [], 'train_loss': [], 'train_accuracy': []}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def build_network(self, input_dim, output_dim):
        layers = []
        previous_dim = input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, output_dim))
        self.model_layers = nn.Sequential(*layers)
        self.model_layers.to(self.device)

    def forward(self, x):
        return self.model_layers(x)

    def _create_optimizer(self):
        optimizer_name = self.optimizer_name.lower()
        if optimizer_name == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)
        if optimizer_name == 'adamw':
            return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def _create_loss_function(self):
        if self.loss_name.lower() == 'cross_entropy_label_smoothing':
            return nn.CrossEntropyLoss(label_smoothing=0.1)
        return nn.CrossEntropyLoss()

    def fit(self, X, y):
        X_np = np.array(X, dtype=np.float32)
        y_np = np.array(y)

        label_values = sorted(np.unique(y_np).tolist())
        self.label_to_index = {label: idx for idx, label in enumerate(label_values)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}
        y_encoded = np.array([self.label_to_index[label] for label in y_np], dtype=np.int64)

        self.build_network(input_dim=X_np.shape[1], output_dim=len(label_values))

        features = torch.FloatTensor(X_np)
        labels = torch.LongTensor(y_encoded)
        dataset = TensorDataset(features, labels)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = self._create_optimizer()
        loss_function = self._create_loss_function()

        self.learning_curve = {'epoch': [], 'train_loss': [], 'train_accuracy': []}

        for epoch in range(self.max_epoch):
            nn.Module.train(self, True)
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                logits = self.forward(batch_X)
                train_loss = loss_function(logits, batch_y)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                epoch_loss += train_loss.item() * batch_X.size(0)
                batch_prediction = torch.argmax(logits, dim=1)
                epoch_correct += (batch_prediction == batch_y).sum().item()
                epoch_total += batch_y.size(0)

            mean_loss = epoch_loss / epoch_total
            mean_accuracy = epoch_correct / epoch_total
            self.learning_curve['epoch'].append(epoch + 1)
            self.learning_curve['train_loss'].append(mean_loss)
            self.learning_curve['train_accuracy'].append(mean_accuracy)

            if self.verbose_step > 0 and (epoch + 1) % self.verbose_step == 0:
                print('Epoch:', epoch + 1, 'Train Accuracy:', mean_accuracy, 'Train Loss:', mean_loss)
    
    def test(self, X):
        nn.Module.eval(self)
        X_test = torch.FloatTensor(np.array(X, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            logits = self.forward(X_test)
            pred_label_index = torch.argmax(logits, dim=1).cpu().numpy()
        pred_label = np.array([self.index_to_label[idx] for idx in pred_label_index])
        return pred_label
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.fit(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {
            'pred_y': pred_y,
            'true_y': np.array(self.data['test']['y']),
            'learning_curve': self.learning_curve,
            'train_size': len(self.data['train']['y']),
            'test_size': len(self.data['test']['y'])
        }
            