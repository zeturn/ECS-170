'''
CNN model for ECS 170 Stage 3 handwritten digit classification.
This model expects grayscale input with shape N x 1 x H x W and predicts 10 classes.
'''

from local_code.base_class.method import method
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt


class Method_CNN(method, nn.Module):
    data = None
    num_classes = 10
    history_destination_folder_path = '../../result/stage_3_result/'

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.prefix = mName.lower().replace(" ", "_") # used for saving history
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.max_epoch = None
        self.learning_rate = None
        self.batch_size = None

        # Three convolutional layers with increasing channel sizes
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25) # Avoid overfitting
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Fully connected layers for classification
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, self.num_classes)
        self.history = []

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = F.relu(self.conv3(x))

        x = self.adaptive_pool(x)
        x = x.reshape(x.shape[0], -1) # Flatten for fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    def _prepare_X(self, X):

        # 转换为张量 float32
        X = torch.FloatTensor(np.array(X, dtype=np.float32))
        if X.max() > 1.0:
            X = X / 255.0

        # 如果输入是 3D（例如 N x H x W），在第二个维度插入通道维度，变为 N x 1 x H x W（假设灰度图像）
        if len(X.shape) == 3:
            X = X.unsqueeze(1)
        
        # 确保通道在前
        if len(X.shape) == 4 and X.shape[-1] in [1, 3] and X.shape[1] not in [1, 3]:
            X = X.permute(0, 3, 1, 2)
        return X

    def _prepare_y(self, y):
        y_tensor = torch.LongTensor(np.array(y, dtype=np.int64).reshape(-1))
        if y_tensor.min() > 0:
            y_tensor = y_tensor - y_tensor.min()
        return y_tensor

    def _accuracy(self, logits, y):
        pred = logits.max(1)[1]
        return (pred == y).float().mean().item()

    def train_model(self, X_train, y_train, X_test, y_test):

        if self.learning_rate is None:
            raise ValueError("learning_rate not set")
        if self.max_epoch is None:
            raise ValueError("max_epoch not set")
        if self.batch_size is None:
            raise ValueError("batch_size not set")

        X_train = self._prepare_X(X_train)
        y_train = self._prepare_y(y_train)
        X_test = self._prepare_X(X_test)
        y_test = self._prepare_y(y_test)

        # 超参数 Check
        print(">>> learning rate actually used:", self.learning_rate)
        print(">>> batch size actually used:", self.batch_size)
        print(">>> max_epoch actually used:", self.max_epoch)


        # 如果不是 1 通道，重新定义 conv1
        in_channels = X_train.shape[1]
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, padding=2).to(self.device)

        found_classes = int(torch.max(torch.cat([y_train, y_test])).item()) + 1
        if found_classes != self.num_classes:
            self.num_classes = found_classes
            self.fc2 = nn.Linear(128, self.num_classes).to(self.device)
        # 如果数据集中类别数不等于默认的 self.num_classes，重新创建 self.fc2 输出层

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()

        self.to(self.device)
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=self.batch_size, shuffle=False)

        self.history = []
        for epoch in range(1, self.max_epoch + 1): # 从1开始计数
            self.train()
            total_loss = 0.0
            total_count = 0
            correct_train = 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                y_pred = self.forward(batch_X) # 前向传播得到 logits
                train_loss = loss_function(y_pred, batch_y)
                train_loss.backward() 
                optimizer.step()    # 参数更新

                total_loss += train_loss.item() * batch_X.shape[0]
                total_count += batch_X.shape[0]
                pred_labels = y_pred.max(1)[1]
                correct_train += (pred_labels == batch_y).sum().item()

            avg_loss = total_loss / total_count
            train_acc = correct_train / total_count

            # 评估
            self.eval()
            correct_test = 0
            with torch.no_grad():
                for test_X, test_y in test_loader:
                    test_X, test_y = test_X.to(self.device), test_y.to(self.device)
                    test_logits = self.forward(test_X)
                    pred_test = test_logits.max(1)[1]
                    correct_test += (pred_test == test_y).sum().item()

            test_acc = correct_test / len(y_test)
            row = {'epoch': epoch, 'train_loss': avg_loss, 'train_accuracy': train_acc, 'test_accuracy': test_acc}
            self.history.append(row)
            print('Epoch:', epoch, 'Loss:', round(avg_loss, 4), 'Train Acc:', round(train_acc, 4), 'Test Acc:',
              round(test_acc, 4))


        self._save_history()


    def test(self, X):
        X = self._prepare_X(X)
        self.eval()
        with torch.no_grad():
            y_pred = self.forward(X)
        return y_pred.max(1)[1].numpy()


    def run(self):
        print('method running...')
        print('--start training...')
        self.train_model(
            self.data['train']['X'],
            self.data['train']['y'],
            self.data['test']['X'],
            self.data['test']['y']
        )
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        true_y = np.array(self.data['test']['y'], dtype=np.int64).reshape(-1)
        return {'pred_y': pred_y, 'true_y': true_y, 'history': self.history}


    def _save_history(self):
        os.makedirs(self.history_destination_folder_path, exist_ok=True)
        csv_path = os.path.join(self.history_destination_folder_path, f'{self.prefix}_learning_curve.csv')

        with open(csv_path, 'w') as f:
            f.write('epoch,train_loss,train_accuracy,test_accuracy\n')
            for row in self.history:
                f.write(str(row['epoch']) + ',' + str(row['train_loss']) + ',' + str(row['train_accuracy']) + ',' + str(
                    row['test_accuracy']) + '\n')
        print('saved learning curve data to:', csv_path)

        try:
            import matplotlib.pyplot as plt

            epochs = [row['epoch'] for row in self.history]
            losses = [row['train_loss'] for row in self.history]
            plt.figure()
            plt.plot(epochs, losses)
            plt.xlabel('Epoch')
            plt.ylabel('Training Loss')
            plt.title('CNN Training Loss')
            plt.savefig(os.path.join(self.history_destination_folder_path, 'cnn_training_loss.png'), bbox_inches='tight')
            plt.close()

            train_acc = [row['train_accuracy'] for row in self.history]
            test_acc = [row['test_accuracy'] for row in self.history]
            plt.figure()
            plt.plot(epochs, train_acc, label='Train Accuracy')
            plt.plot(epochs, test_acc, label='Test Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('CNN Accuracy')
            plt.legend()
            plt.savefig(os.path.join(self.history_destination_folder_path, 'cnn_accuracy.png'), bbox_inches='tight')
            plt.close()
            print('saved learning curve plots to:', self.history_destination_folder_path)
        except Exception as e:
            print('plot saving skipped:', e)
