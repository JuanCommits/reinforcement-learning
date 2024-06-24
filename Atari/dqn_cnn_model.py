import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN_CNN_Model(nn.Module):
    def __init__(self, env_inputs, n_actions):
        super(DQN_CNN_Model, self).__init__()
        in_channels, in_height, in_width = env_inputs
        self.first_layer = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        out_height = (in_height - 8) // 4 + 1
        out_width = (in_width - 8) // 4 + 1

        self.ln = nn.LayerNorm([32, out_height, out_width])
        self.second_layer = nn.Conv2d(32, 64, kernel_size=6, stride=2)

        self.out_layer = nn.Sequential(nn.Linear(4096, 256), nn.Linear(256, n_actions))

    def forward(self, env_input):
        x = self.first_layer(env_input)
        x = self.second_layer(self.ln(x))
        x = self.out_layer(x.flatten(1))
        return x


class DQN_CNN_Model_Paper(nn.Module):
    def __init__(self, env_inputs, n_actions):
        super(DQN_CNN_Model_Paper, self).__init__()

        in_channels, in_height, in_width = env_inputs
        self.l1 = nn.Conv2d(in_channels, 16, kernel_size=8, stride=4)
        self.l2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        self.fc1 = nn.Linear(self.feat_size(env_inputs), 256)
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)

    def feat_size(self, shape):
        return self.l2(self.l1(torch.zeros(1, *shape))).view(1, -1).size(1)
