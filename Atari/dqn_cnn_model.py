import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

class DQN_CNN_Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN_CNN_Model, self).__init__()
        self.l1 = Linear(input_dim, 128)
        self.l2 = Linear(128, 128)
        self.out = Linear(128, output_dim)

    def forward(self, x):
        pred = torch.relu(self.l1(x))
        pred = torch.relu(self.l2(pred))
        return self.out(pred)