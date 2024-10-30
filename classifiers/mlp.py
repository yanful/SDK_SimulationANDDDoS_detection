import torch
from torch.nn import Module
from torch.nn import Sequential, MaxPool1d, Conv1d, Flatten, Linear
from torch.nn import ReLU, Sigmoid
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

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
    
class LSTMAutoencoder(Module):
    def __init__(self, dim_x, dim_y):
        pass

    def forward(self, x):
        pass