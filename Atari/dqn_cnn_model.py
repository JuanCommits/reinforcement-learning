import torch.nn as nn
import torch.nn.functional as F

class DQN_CNN_Model(nn.Module):
    def __init__(self,  env_inputs, n_actions):
        super(DQN_CNN_Model, self).__init__()
        in_channels, in_height, in_width = env_inputs
        self.first_layer = nn.Conv2d(in_channels, 16, kernel_size=8, stride=4)
        self.second_layer = nn.Conv2d(16, 16, kernel_size=6, stride=3)

        self.out_layer = nn.Sequential(
            nn.Linear(400, 128),
            nn.Linear(128, n_actions)
        )

    def forward(self, env_input):
        x = self.first_layer(env_input)
        x = self.second_layer(x)
        x = self.out_layer(x.flatten(1))
        return x