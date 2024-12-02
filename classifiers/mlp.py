import torch
from torch.nn import Module
from torch.nn import Sequential, MaxPool1d, Conv1d, Flatten, Linear, Dropout
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
        super(BasicMLP, self).__init__()
        self.model = Sequential(
            Linear(dim_x, dim_h),
            Dropout(p=0.5),
            Linear(dim_h, dim_h),
            Dropout(p=0.5),
            Linear(dim_h, dim_y),
            Dropout(p=0.5),
            Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)
    
def train(model, x_train, y_train, epoch=100, lr=0.1):
    criterion = CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for e in range(epoch):
        output = model(x_train)
        loss = criterion(output, y_train)

        optim.zero_grad()

        loss.backward()
        optim.step()

        output_np = output.detach().numpy().argmax(axis=1)
        y_train_np = y_train.detach().numpy().argmax(axis=1)
        acc = accuracy_score(output_np, y_train_np)
        tn, fp, fn, tp = confusion_matrix(output_np, y_train_np).ravel()
        print(f'Epoch {e}: Acc: {acc} / TN: {tn} / FP: {fp} / FN: {fn} / TP: {tp}, Loss: {loss.item()}')

def eval(model, X_train, y_train):
    model.eval()

    with torch.no_grad():
        out = model(X_train).argmax(axis=1)
        acc = accuracy_score(out, y_train.argmax(axis=1))
        tn, fp, fn, tp = confusion_matrix(out, y_train.argmax(axis=1)).ravel()
        print(f'Acc: {acc} / TN: {tn} / FP: {fp} / FN: {fn} / TP: {tp}')

def one_hot_encode(y):
    return np.array([[1, 0] if i == 0 else [0, 1] for i in y])

if __name__ == "__main__":
    dataset = np.loadtxt('data/result/result_0.csv', delimiter=',', dtype=str)[1:]
    dataset[:, 1] = np.where(dataset[:, 1] == 'TCP', 0,
        np.where(dataset[:, 1] == 'UDP', -1, 1))
    dataset = dataset.astype(float)
    X = dataset[:, :-1]
    y = one_hot_encode(dataset[:, -1:])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    X_train = torch.tensor(X_train, dtype=torch.float32, requires_grad=True)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32, requires_grad=True)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    model = BasicMLP(6, 10, 2)
    train(model, X_train, y_train)
    eval(model, X_test, y_test)