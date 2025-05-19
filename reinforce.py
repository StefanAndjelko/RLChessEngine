import chess
import numpy as np
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm

from helper import move_index_mapper, board_to_tensor
from policy_network import PolicyNetwork

def game_episode(p_net, move_index, index_move):
    board = chess.Board()

    end = False
    sa_history = []
    log_probs = []
    white = True
    reward = None
    # print(board)
    while not end:
        turn = board.fullmove_number
        
        legal_moves = list(board.generate_legal_moves())

        action = None
        if white:
            tensor_board = board_to_tensor(board)
            action_logits = p_net(tensor_board).squeeze(0)

            legal_move_indices = [move_index[move] for move in legal_moves]
            legal_logits = action_logits[legal_move_indices]

            action_probs = F.softmax(legal_logits, dim=0)
            action_distribution = torch.distributions.Categorical(action_probs)
            action_ix = action_distribution.sample()
            log_prob = action_distribution.log_prob(action_ix)

            action = legal_moves[action_ix.item()]
            log_probs.append(log_prob)
        else:
            action = random.choice(legal_moves)

        sa_history.append((board.fen(), action))
        board.push(action)

        white = ~white

        # print(board)
        # end = True

        if board.is_game_over(claim_draw=True):
            reward = board.result(claim_draw=True)
            # print(reward)

            if reward == "1-0":
                reward = 1
            elif reward == "0-1":
                reward = -1
            else:
                reward = 0

            # print(reward)
            end = True
            # print(board)
    
    return sa_history, log_probs, reward

def reinforce():
    p_net = PolicyNetwork()
    move_index, index_move = move_index_mapper()

    lengths = []
    rewards = []

    lr_p_net = 2**-13
    optimizer = torch.optim.Adam(p_net.parameters(), lr=lr_p_net)

    num_episodes = 1000
    win_counter = 0
    loss_counter = 0
    for i in tqdm(range(num_episodes)):
        sa_history, log_probs, final_reward = game_episode(p_net, move_index, index_move)

        if final_reward == 1:
            win_counter += 1
        elif final_reward == -1:
            loss_counter += 1

        loss = sum(-log_prob * final_reward for log_prob in log_probs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(f"Episode {i + 1}, Reward: {final_reward}, Loss: {loss.item():.4f}")

        if i == num_episodes - 1:
            board = chess.Board()
            board.set_fen(sa_history[-1][0])
            torch.save(p_net.state_dict(), "model_weights.pth")
            print(f"Played {num_episodes} games. Win: {win_counter} | Loss: {loss_counter}")

if __name__ == "__main__":
    reinforce()



