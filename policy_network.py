import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess

from helper import board_to_tensor

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(8 * 8 * 13, 256)
        self.fc2 = nn.Linear(256, 4608)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return torch.reshape(x, (-1,))

board = chess.Board()
tensor = board_to_tensor(board)
p_net = PolicyNetwork()
print(p_net(tensor).size())


