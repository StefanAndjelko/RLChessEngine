import chess
import numpy as np
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm

from helper import generate_random_kqk_position, generate_simple_positions, select_action, encode_board, action_index_to_move
from policy_network import PolicyNetwork

def game_episode(p_net, iteration):
    board1, board2, board3 = generate_simple_positions()
    board = None
    if iteration < 10000:
        # board = random.choice([board1, board2, board3])
        board = random.choice([board1, board2, board3])
    elif iteration < 15000:
        if random.uniform(0, 1) < 0.5:
            board = random.choice([board1, board2])
        else:
            board = board3
    elif iteration < 20000:
        if random.uniform(0, 1) < 0.2:
            board = random.choice([board1, board2])
        else:
            board = board3
    elif iteration < 25000:
        if random.uniform(0, 1) < 0.05:
            board = random.choice([board1, board2])
        else:
            board = board3
    else:
        board = generate_random_kqk_position()

    log_probs = []
    rewards = []
    sa_history = []
    
    prev_king_mobility = 8
    
    while True:
        state = encode_board(board)
        
        action_idx, probability = select_action(p_net, state, board)
        log_probs.append(torch.log(probability))
        action = action_index_to_move(board, action_idx)
        board.push(action)
        sa_history.append((state, action))
        
        if board.is_game_over(claim_draw=True):
            result = board.result(claim_draw=True)
            reward = 1.0 if result == "1-0" else -1.0
            rewards.append(reward)
            return sa_history, log_probs, rewards
        
        legal_moves = list(board.generate_legal_moves())
        capturing_moves = [move for move in legal_moves if board.is_capture(move)]
        if capturing_moves:
            board.push(capturing_moves[0])
        else:
            board.push(random.choice(legal_moves))
        
        if board.is_game_over(claim_draw=True):
            result = board.result(claim_draw=True)
            reward = 1.0 if result == "1-0" else -1.0
            rewards.append(reward)
            return sa_history, log_probs, rewards
        
        reward = 0
        
        # king_sq = board.king(chess.BLACK)
        # king_file, king_rank = chess.square_file(king_sq), chess.square_rank(king_sq)
        
        # if min(7 - king_file, king_file) == 0 or min(7 - king_rank, king_rank) == 0:
        #     reward += 0.002
        
        rewards.append(reward)

def reinforce():
    p_net = PolicyNetwork()

    lengths = []
    rewards = []

    lr_p_net = 2**-13
    gamma = 1
    optimizer = torch.optim.Adam(p_net.parameters(), lr=lr_p_net)

    num_episodes = 50000
    win_counter = 0
    won_games = []
    for i in tqdm(range(num_episodes)):
        sa_history, log_probs, rewards = game_episode(p_net, i)
        if (len(log_probs) != len(rewards)):
            print("not same length")
            exit(0)
        
        if rewards[-1] == 1:
            win_counter += 1
            won_games.append([sa[1] for sa in sa_history])
        
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns)
        
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        loss = []
        entropy_bonus = 0
        for log_prob, g in zip(log_probs, returns):
            loss.append(-log_prob * g)

        
        optimizer.zero_grad()
        total_loss = torch.stack(loss).sum()
        total_loss.backward()
        optimizer.step()
        
        if i % 1000 == 0:
            print()
            print(f"Won {win_counter} games out of 1000")
            print()
            win_counter = 0

    torch.save(p_net.state_dict(), "chess_model.pth")
    print(won_games[-1])

if __name__ == "__main__":
    reinforce()