import chess
import numpy as np
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm

from helper import generate_random_kqk_position, generate_easy_kqk_position, select_action, encode_board, action_index_to_move
from policy_network import PolicyNetwork

def game_episode(p_net, iteration):
    board = None
    if iteration < 10000:
        board = generate_easy_kqk_position()
    elif iteration < 20000:
        board = generate_easy_kqk_position() if random.random() < 0.7 else generate_random_kqk_position()
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
        black_move = random.choice(legal_moves)
        board.push(black_move)
        
        if board.is_game_over(claim_draw=True):
            result = board.result(claim_draw=True)
            reward = 1.0 if result == "1-0" else -1.0
            rewards.append(reward)
            return sa_history, log_probs, rewards
        
        reward = -0.5
        
        king_sq = board.king(chess.BLACK)
        king_file, king_rank = chess.square_file(king_sq), chess.square_rank(king_sq)

        safe_squares = 0
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1), (1,-1), (-1,-1)]:
            target_file = king_file + dx
            target_rank = king_rank + dy
            if 0 <= target_file < 8 and 0 <= target_rank < 8:
                target_sq = chess.square(target_file, target_rank)
                if board.is_legal(chess.Move(king_sq, target_sq)):
                    safe_squares += 1
        
        mobility_bonus = (prev_king_mobility - safe_squares) * 0.1
        reward += mobility_bonus
        prev_king_mobility = safe_squares
        
        if min(7 - king_file, king_file) == 0 or min(7 - king_rank, king_rank) == 0:
            reward += 0.3
        
        rewards.append(reward)

def reinforce():
    p_net = PolicyNetwork()

    lengths = []
    rewards = []

    lr_p_net = 2**-12
    gamma = 1
    optimizer = torch.optim.Adam(p_net.parameters(), lr=lr_p_net)

    num_episodes = 10000
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
            entropy_bonus -= 0.01 * torch.exp(log_prob) * log_prob

        
        optimizer.zero_grad()
        total_loss = torch.stack(loss).sum() + entropy_bonus
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



