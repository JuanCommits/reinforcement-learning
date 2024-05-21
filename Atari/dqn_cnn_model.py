import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

class DQN_CNN_Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN_CNN_Model, self).__init__()
        self.l1 = Linear(input_dim, 64)
        self.l2 = Linear(64, 128)
        self.l3 = Linear(128, 128)
        self.l4 = Linear(128, 64)
        self.out = Linear(64, output_dim)

    def forward(self, x):
        pred = torch.relu(self.l1(x))
        pred = torch.relu(self.l2(pred))
        pred = torch.relu(self.l3(pred))
        pred = torch.relu(self.l4(pred))
        return self.out(pred)