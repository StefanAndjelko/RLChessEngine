import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess

from helper import board_to_tensor

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(13, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 4608)

    def forward(self, x):
        x = x.view(-1, 13, 8, 8)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

board = chess.Board()
tensor = board_to_tensor(board)
p_net = PolicyNetwork()
print(p_net(tensor).size())


