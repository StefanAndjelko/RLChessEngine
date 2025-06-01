import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

DIRECTIONS = [(0, 1), (1, 1), (1, 0), (1, -1),
              (0, -1), (-1, -1), (-1, 0), (-1, 1)]

def encode_board(board):
    safe_squares = 0

    white_queen = list(board.pieces(chess.QUEEN, chess.WHITE))[0]
    white_king = list(board.pieces(chess.KING, chess.WHITE))[0]
    black_king = list(board.pieces(chess.KING, chess.BLACK))[0]

    w_q_x, w_q_y = (int(chess.square_file(white_queen)) / 7, int(chess.square_rank(white_queen)) / 7)
    w_k_x, w_k_y = (int(chess.square_file(white_king)) / 7, int(chess.square_rank(white_king)) / 7)
    b_k_x, b_k_y = (int(chess.square_file(black_king)) / 7, int(chess.square_rank(black_king)) / 7)

    safe_squares = 0
    for dx, dy in DIRECTIONS:
        target_file = chess.square_file(black_king) + dx
        target_rank = chess.square_rank(black_king) + dy
        if 0 <= target_file < 8 and 0 <= target_rank < 8:
            target_square = chess.square(target_file, target_rank)
            
            if not board.is_attacked_by(chess.WHITE, target_square):
                safe_squares += 1

    safe_squares /= 8
    
    bk_wq_distance = max(abs(w_q_x * 7 - b_k_x * 7), abs(w_q_y * 7 - b_k_y * 7)) / 7
    bk_wk_distance = max(abs(w_k_x * 7 - b_k_x * 7), abs(w_k_y * 7 - b_k_y * 7)) / 7
    wk_wq_distnace= max(abs(w_q_x * 7 - w_k_x * 7), abs(w_q_y * 7 - w_k_y * 7)) / 7
     
    return np.array([w_q_x, w_q_y, w_k_x, w_k_y, b_k_x, b_k_y, bk_wq_distance, bk_wk_distance, wk_wq_distnace, safe_squares], dtype='f')

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

        board = chess.Board(None)
        board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(wq, chess.Piece(chess.QUEEN, chess.WHITE))
        board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        board.turn = chess.WHITE

        if board.is_valid():
            return board

def generate_trivial_kqk_position():
    boards = []
    board = chess.Board(None)
    board.set_piece_at(chess.B5, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.C7, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.A8, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE
    boards.append(board)
    board = chess.Board(None)
    board.set_piece_at(chess.B4, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.C2, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE
    boards.append(board)
    board = chess.Board(None)
    board.set_piece_at(chess.G4, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.F2, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.H1, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE
    boards.append(board)
    board = chess.Board(None)
    board.set_piece_at(chess.G5, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.F7, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE
    boards.append(board)

    return random.choice(boards)

def generate_easy_kqk_position():
    boards = []
    board = chess.Board(None)
    board.set_piece_at(chess.E5, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.H6, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.G8, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE
    boards.append(board)
    board = chess.Board(None)
    board.set_piece_at(chess.D4, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.A3, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.B1, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE
    boards.append(board)
    board = chess.Board(None)
    board.set_piece_at(chess.E4, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.H3, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.G1, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE
    boards.append(board)
    board = chess.Board(None)
    board.set_piece_at(chess.D5, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.A6, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.B8, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE
    boards.append(board)

    return random.choice(boards)

def generate_medium_kqk_position():
    boards = []
    board = chess.Board(None)
    board.set_piece_at(chess.G3, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.F7, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.H5, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE
    boards.append(board)
    board = chess.Board(None)
    board.set_piece_at(chess.F7, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.B6, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.D8, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE
    boards.append(board)
    board = chess.Board(None)
    board.set_piece_at(chess.B6, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.C2, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.A4, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE
    boards.append(board)
    board = chess.Board(None)
    board.set_piece_at(chess.F2, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.B3, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.D1, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE
    boards.append(board)

    return random.choice(boards)

def generate_simple_positions():
    return generate_trivial_kqk_position(), generate_easy_kqk_position(), generate_medium_kqk_position()

def is_valid_queen_move(from_sq, dir_idx, dist, board):
    dx, dy = DIRECTIONS[dir_idx]
    f, r = chess.square_file(from_sq), chess.square_rank(from_sq)
    
    for step in range(1, dist + 1):
        target_f = f + dx * step
        target_r = r + dy * step
        
        if not (0 <= target_f < 8 and 0 <= target_r < 8):
            return False
            
        target_sq = chess.square(target_f, target_r)
        
        if step < dist and board.piece_at(target_sq):
            return False
            
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

def select_action(p_net, state, board):
    # print(f"STATE: {state}")
    logits = p_net(state)
    # print(f"LOGITS: {logits}")
    mask = torch.zeros(64)
    
    queen_sq = list(board.pieces(chess.QUEEN, chess.WHITE))[0]
    king_sq = list(board.pieces(chess.KING, chess.WHITE))[0]
    for dir_idx in range(8):
        for dist in range(1, 8):
            action_idx = dir_idx * 7 + (dist - 1)
            if is_valid_queen_move(queen_sq, dir_idx, dist, board):
                mask[action_idx] = 1

    for dir_idx in range(8):
        action_idx = 56 + dir_idx
        if is_valid_king_move(king_sq, dir_idx, board):
            mask[action_idx] = 1
    
    masked_logits = logits.masked_fill(mask == 0, -1e8)
    probs = F.softmax(masked_logits, dim=-1)
    action = torch.multinomial(probs, 1).item()

    return action, probs[action]

def action_index_to_move(board, action_idx):
    queen_sq = list(board.pieces(chess.QUEEN, chess.WHITE))[0]
    king_sq = list(board.pieces(chess.KING, chess.WHITE))[0]

    if action_idx < 56:
        dir_idx = action_idx // 7
        dist = (action_idx % 7) + 1
        dx, dy = DIRECTIONS[dir_idx]
        f, r = chess.square_file(queen_sq), chess.square_rank(queen_sq)
        target_f = f + dx * dist
        target_r = r + dy * dist
        target_sq = chess.square(target_f, target_r)
        return chess.Move(queen_sq, target_sq)
    else:
        dir_idx = action_idx - 56
        dx, dy = DIRECTIONS[dir_idx]
        f, r = chess.square_file(king_sq), chess.square_rank(king_sq)
        target_f = f + dx
        target_r = r + dy
        target_sq = chess.square(target_f, target_r)
        return chess.Move(king_sq, target_sq)

# board = generate_random_kqk_position()
# print(encode_board(board))