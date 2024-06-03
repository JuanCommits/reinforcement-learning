import torch.nn as nn
import torch.nn.functional as F

class DQN_Model(nn.Module):
    def __init__(self,  env_inputs, n_actions):
        super(DQN_Model, self).__init__()
        self.l1 = nn.Linear(env_inputs, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, n_actions)

    def forward(self, env_input):
        x = F.relu(self.l1(env_input))
        x = F.relu(self.l2(x))
        return self.l3(x)