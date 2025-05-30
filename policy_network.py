import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess

from helper import encode_board, generate_random_kqk_position

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=8, action_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_dim)
    
    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32, device=next(self.parameters()).device)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.out(x)

        return x

# board = generate_random_kqk_position()
# tensor = encode_board(board)
# p_net = PolicyNetwork()
# print(p_net(tensor).size())


