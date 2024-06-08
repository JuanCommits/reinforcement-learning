import torch.nn as nn
import torch.nn.functional as F

class DQN_CNN_Model(nn.Module):
    def __init__(self,  env_inputs, n_actions):
        super(DQN_CNN_Model, self).__init__()
        in_channels, in_height, in_width = env_inputs
        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=8, stride=4),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU()
        )

        self.out_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.Linear(128, n_actions)
        )

    def forward(self, env_input):
        x = self.first_layer(env_input)
        x = self.out_layer(x)
        return x