import torch.nn as nn
import torch.nn.functional as F

class DQN_CNN_Model(nn.Module):
    def __init__(self,  env_inputs, n_actions):
        super(DQN_CNN_Model, self).__init__()
        in_channels, in_height, in_width = env_inputs
        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d(2, 2)

        self.out_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 256),
            nn.Linear(256, n_actions)
        )

    def forward(self, env_input):
        x = self.first_layer(env_input)
        x = self.pool(x)
        x = self.out_layer(x)
        return x