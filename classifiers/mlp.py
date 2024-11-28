import torch
from torch.nn import Module
from torch.nn import Sequential, MaxPool1d, Conv1d, Flatten, Linear
from torch.nn import ReLU, Sigmoid, Softmax
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np

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
    
    def train(self, x_train, y_train, epoch=100):
        self.train()

        total_step = len(x_train)

        for e in range(epoch):
            for i, x in enumerate(x_train):
                b_x = Variable(x)
                b_y = Variable(y_train[i])
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
            acc = accuracy_score(out, y)
            tn, fp, fn, tp = confusion_matrix(out, y).ravel()
            print(f'Acc: {acc} / TN: {tn} / FP: {fp} / FN: {fn} / TP: {tp}')

def one_hot_encode(y):
    return np.array([[1, 0] if i == 0 else [0, 1] for i in y])

if __name__ == "__main__":
    dataset = np.loadtxt('dataset.csv', delimiter=',', dtype=np.int64)
    X = dataset[:, :-1]
    y = one_hot_encode(dataset[:, -1:])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    X_train = torch.tensor(X_train)
    X_test = torch.tensor(X_test)
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)

    model = BasicMLP(6, 10, 2)
    model.train(X_train, y_train)
    model.eval(X_test, y_test)