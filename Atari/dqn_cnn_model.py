import torch.nn as nn
import torch.nn.functional as F

class DQN_CNN_Model(nn.Module):
    def __init__(self,  env_inputs, n_actions, is_actor=False):
        super(DQN_CNN_Model, self).__init__()
        self.is_actor = is_actor
        in_channels, in_height, in_width = env_inputs
        self.first_layer = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.second_layer = nn.Conv2d(32, 64, kernel_size=6, stride=2)

        self.out_layer = nn.Sequential(
            nn.Linear(4096, 256),
            nn.Linear(256, n_actions)
        )

    def forward(self, env_input):
        x = self.first_layer(env_input)
        x = self.second_layer(x)
        x = self.out_layer(x.flatten(1))
        if self.is_actor:
            return F.softmax(x, dim=1)
        return x