import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess

from helper import board_to_tensor

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=8, action_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# board = chess.Board()
# tensor = board_to_tensor(board)
# p_net = PolicyNetwork()
# print(p_net(tensor).size())


