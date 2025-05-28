import chess
import numpy as np
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm

from helper import generate_random_kqk_position, select_action, encode_board
from policy_network import PolicyNetwork

def game_episode(p_net, move_index, index_move):
    board = generate_random_kqk_position()

    end = False
    sar_history = []
    log_probs = []
    white = True
    reward = None
    # print(board)
    while not end:
        turn = board.fullmove_number
        
        legal_moves = list(board.generate_legal_moves())

        action = None
        if white:
            state = encode_board(board)
            action, probabilities = select_action(state, board)
            print(action)
            print(probabilities)
            end = True
        else:
            action = random.choice(legal_moves)

        reward = -0.01
        sar_history.append((board.fen(), action, reward))
        board.push(action)

        white = ~white

        # print(board)
        # end = True

        if board.is_game_over(claim_draw=True):
            game_result = board.result(claim_draw=True)

            if game_result == "1-0":
                reward = 1
            else:
                reward = -1

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



