import torch
from torch import nn

class NeuralNet(nn.Module):
    def __init__(self, mconfig):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(**mconfig['l1_kwargs'])
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(**mconfig['l2_kwargs'])

    def forward(self, X):
        out = self.l1(X)
        out = self.relu(out)
        out = self.l2(out)
        return out
    