import chess
import numpy as np
import torch
import torch.nn.functional as F

from helper import move_index_mapper, board_to_tensor
from policy_network import PolicyNetwork

def game_episode(p_net, move_index, index_move):
    board = chess.Board()

    end = False
    sar_history = []
    log_probs = []
    black = False
    # print(board)
    while not end:
        turn = board.fullmove_number
        # if black:
        #     print(f"Move: {turn}")
        black = ~black
        tensor_board = board_to_tensor(board)
        action_logits = p_net(tensor_board)

        legal_moves = list(board.generate_legal_moves())
        legal_move_indices = [move_index[move] for move in legal_moves]
        legal_logits = action_logits[legal_move_indices]

        action_probs = F.softmax(legal_logits, dim=0)
        action_distribution = torch.distributions.Categorical(action_probs)
        action_ix = action_distribution.sample()
        log_prob = action_distribution.log_prob(action_ix)

        action = legal_moves[action_ix.item()]
        board.push(action)

        sar_history.append((board.fen(), action, 0))
        log_probs.append(log_prob)

        # print(board)
        # end = True

        if board.is_game_over(claim_draw=True):
            reward = board.result(claim_draw=True)
            print(reward)

            if reward == "1-0":
                reward = 1
            elif reward == "0-1":
                reward = -1
            else:
                reward = 0
            sar_history.append((board.fen(), action, reward))
            print(reward)
            end = True
            print(board)
    
    return sar_history, log_probs

def reinforce():
    p_net = PolicyNetwork()
    move_index, index_move = move_index_mapper()

    lengths = []
    rewards = []

    gamma = 0.99
    lr_p_net = 2**-13
    optimizer = torch.optim.Adam(p_net.parameters(), lr=lr_p_net)

    for i in range(10):
        sar_history, log_probs = game_episode(p_net, move_index, index_move)

        final_reward = sar_history[-1][2]

        loss = 0
        for log_prob in log_probs:
            loss += -log_prob * final_reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Episode {i + 1}, Reward: {final_reward}, Loss: {loss.item():.4f}")

        if i == 9:
            board = chess.Board()
            board.set_fen(sar_history[-1][0])

if __name__ == "__main__":
    reinforce()



