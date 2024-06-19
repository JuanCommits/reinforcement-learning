import torch.nn as nn
import torch.nn.functional as F

class ActorModel(nn.Module):
    def __init__(self,  env_inputs, n_actions):
        super(ActorModel, self).__init__()
        self.layer_norm = nn.LayerNorm(128)
        self.l1 = nn.Linear(env_inputs, 128)
        self.l2 = nn.Linear(128, n_actions)

    def forward(self, env_input):
        x = F.relu(self.l1(env_input))
        self.layer_norm(x)
        x = self.l2(x)
        return F.softmax(x, dim=1)
    
class CriticModel(nn.Module):
    def __init__(self,  env_inputs):
        super(CriticModel, self).__init__()
        self.l1 = nn.Linear(env_inputs, 128)
        self.l2 = nn.Linear(128, 1)

    def forward(self, env_input):
        x = F.relu(self.l1(env_input))
        return self.l2(x)