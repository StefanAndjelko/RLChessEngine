import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# def move_index_mapper():
#     moves = []
#     for from_square in chess.SQUARES:
#         for to_square in chess.SQUARES:
#             moves.append(chess.Move(from_square, to_square))

#             rank_from = chess.square_rank(from_square)
#             rank_to = chess.square_rank(to_square)

#             if (rank_from == 6 and rank_to == 7) or (rank_from == 1 and rank_to == 0):
#                 pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
#                 for piece in pieces:
#                     moves.append(chess.Move(from_square, to_square, promotion=piece))

#     move_to_index = {move: idx for idx, move in enumerate(moves)}
#     index_to_move = {idx: move for move, idx in move_to_index.items()}
    
#     return move_to_index, index_to_move

# def one_hot_encode_peice(piece):
#     pieces = list('rnbqkpRNBQKP.')
#     arr = np.zeros(len(pieces))
#     piece_to_index = {p: i for i, p in enumerate(pieces)}
#     index = piece_to_index[piece]
#     arr[index] = 1
#     return arr

# def encode_board(board):
#     board_str = str(board)
#     board_str = board_str.replace(' ', '')
#     board_list = []
#     for row in board_str.split('\n'):
#         row_list = []
#         for piece in row:
#             row_list.append(one_hot_encode_peice(piece))
#         board_list.append(row_list)

#     return np.array(board_list)

# def board_to_tensor(board):
#     encoded = encode_board(board)
#     tensor = torch.tensor(encoded, dtype=torch.float32).view(-1)

#     return tensor.unsqueeze(0)

def encode_board(board):
    print(board)
    safe_squares = 0
    directions = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1), (1,-1), (-1,-1)]

    for sq in board.pieces(chess.QUEEN, chess.WHITE):
        w_q_x, w_q_y = (int(chess.square_file(sq)) / 7, int(chess.square_rank(sq)) / 7)

    for sq in board.pieces(chess.KING, chess.WHITE):
        w_k_x, w_k_y = (int(chess.square_file(sq)) / 7, int(chess.square_rank(sq)) / 7)

    safe_squares = 0
    for sq in board.pieces(chess.KING, chess.BLACK):
        b_k_x, b_k_y = (int(chess.square_file(sq)) / 7, int(chess.square_rank(sq)) / 7)
        for dx, dy in directions:
            target_file = chess.square_file(sq) + dx
            target_rank = chess.square_rank(sq) + dy
            if 0 <= target_file < 8 and 0 <= target_rank < 8:
                target_square = chess.square(target_file, target_rank)
                
                if not board.is_attacked_by(chess.WHITE, target_square):
                    safe_squares += 1

    safe_squares /= 8

    distance = max(abs(w_q_x * 7 - b_k_x * 7), abs(w_q_y * 7 - b_k_y * 7))
    distance /= 7
     
    return [w_q_x, w_q_y, w_k_x, w_k_y, b_k_x, b_k_y, distance, safe_squares]

def kings_not_adjacent(wk_square, bk_square):
    return not chess.SquareSet(chess.BB_KING_ATTACKS[wk_square] & (1 << bk_square))

def is_legal_endgame(wk, wq, bk):
    board = chess.Board(None)
    board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(wq, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))

    if not kings_not_adjacent(wk, bk):
        return False

    board.turn = chess.WHITE
    if board.is_check():
        return False

    return True

def generate_random_kqk_position():
    squares = list(chess.SQUARES)
    while True:
        wk = random.choice(squares)
        remaining = list(set(squares) - {wk})
        wq = random.choice(remaining)
        remaining = list(set(remaining) - {wq})
        bk = random.choice(remaining)

        if is_legal_endgame(wk, wq, bk):
            board = chess.Board(None)
            board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(wq, chess.Piece(chess.QUEEN, chess.WHITE))
            board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
            board.turn = chess.WHITE
            return board

def is_valid_queen_move(from_sq, dir_idx, dist, board):
    dx, dy = DIRECTIONS[dir_idx]
    f, r = chess.square_file(from_sq), chess.square_rank(from_sq)
    
    # Check path is clear
    for step in range(1, dist + 1):
        target_f = f + dx * step
        target_r = r + dy * step
        
        # Check if off-board
        if not (0 <= target_f < 8 and 0 <= target_r < 8):
            return False
            
        target_sq = chess.square(target_f, target_r)
        
        # Stop at first obstacle
        if step < dist and board.piece_at(target_sq):
            return False
            
        # Final square can be enemy king (capture) or empty
        if step == dist:
            return board.is_legal(chess.Move(from_sq, target_sq))
    
    return True

def is_valid_king_move(from_sq, dir_idx, board):
    dx, dy = DIRECTIONS[dir_idx]
    f, r = chess.square_file(from_sq), chess.square_rank(from_sq)
    target_f, target_r = f + dx, r + dy
    
    if not (0 <= target_f < 8 and 0 <= target_r < 8):
        return False
        
    target_sq = chess.square(target_f, target_r)
    return board.is_legal(chess.Move(from_sq, target_sq))

def select_action(state, board):
    logits = model(state)
    mask = torch.zeros(64)  # 56 queen + 8 king moves
    
    # Queen moves (indices 0-55)
    queen_sq = board.queen_square
    for dir_idx in range(8):
        for dist in range(1, 8):  # Distance 1-7
            action_idx = dir_idx * 7 + (dist - 1)
            if is_valid_queen_move(queen_sq, dir_idx, dist, board):
                mask[action_idx] = 1
    
    # King moves (indices 56-63)
    king_sq = board.king_square
    for dir_idx in range(8):
        action_idx = 56 + dir_idx
        if is_valid_king_move(king_sq, dir_idx, board):
            mask[action_idx] = 1
    
    # Apply mask and sample
    masked_logits = logits - 1e8 * (1 - mask)
    probs = F.softmax(masked_logits, dim=-1)
    action = torch.multinomial(probs, 1).item()
    return action, probs

# board = generate_random_kqk_position()
# print(encode_board(board))