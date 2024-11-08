import torch
from torch.nn import Module
from torch.nn import Sequential, MaxPool1d, Conv1d, Flatten, Linear
from torch.nn import ReLU, Sigmoid, Softmax
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, confusion_matrix

class LUCID(Module):
    def __init__(self, dim_x, dim_y):
        self.model = Sequential(
            Conv1d(),
            ReLU(),
            MaxPool1d(),
            Flatten(),
            Linear(),
            Sigmoid()
        )
        self.loss = CrossEntropyLoss()
        self.optim = Adam()

    def forward(self, x):
        return self.model(x)
    
    def train(self):
        pass

class BasicMLP(Module):
    def __init__(self, dim_x, dim_h, dim_y):
        self.model = Sequential(
            Linear(dim_x, dim_h),
            ReLU(),
            Linear(dim_h, dim_y),
            Softmax()
        )
        self.loss = CrossEntropyLoss()
        self.optim = Adam()

    def forward(self, x):
        return self.model(x)
    
    def train(self, train_data, epoch):
        self.train()

        total_step = len(train_data)

        for e in range(epoch):
            for i, (x, y) in enumerate(train_data):
                b_x = Variable(x)
                b_y = Variable(y)
                output = self(b_x)
                loss = self.loss(output, b_y)

                self.optim.zero_grad()

                self.loss.backward()
                self.optim.step()

                if (i+1) % 100 == 0:
                    acc = accuracy_score(output, b_y)
                    tn, fp, fn, tp = confusion_matrix(output, b_y).ravel()
                    print(f'Epoch {e} Step {i}/{total_step}: Acc: {acc} / TN: {tn} / FP: {fp} / FN: {fn} / TP: {tp}')

    def eval(self, test_data):
        with torch.no_grad():
            x, y = next(iter(test_data))
            out = self(x)
            acc = accuracy_score(x, y)
            tn, fp, fn, tp = confusion_matrix(x, y).ravel()
            print(f'Acc: {acc} / TN: {tn} / FP: {fp} / FN: {fn} / TP: {tp}')
    
class LSTMAutoencoder(Module):
    def __init__(self, dim_x, dim_y):
        pass

    def forward(self, x):
        pass