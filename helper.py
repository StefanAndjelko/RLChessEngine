import chess
import numpy as np
import torch

def move_index_mapper():
    moves = []
    for from_square in chess.SQUARES:
        for to_square in chess.SQUARES:
            moves.append(chess.Move(from_square, to_square))

            rank_from = chess.square_rank(from_square)
            rank_to = chess.square_rank(to_square)

            if (rank_from == 6 and rank_to == 7) or (rank_from == 1 and rank_to == 0):
                pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
                for piece in pieces:
                    moves.append(chess.Move(from_square, to_square, promotion=piece))

    move_to_index = {move: idx for idx, move in enumerate(moves)}
    index_to_move = {idx: move for move, idx in move_to_index.items()}
    
    return move_to_index, index_to_move

def one_hot_encode_peice(piece):
    pieces = list('rnbqkpRNBQKP.')
    arr = np.zeros(len(pieces))
    piece_to_index = {p: i for i, p in enumerate(pieces)}
    index = piece_to_index[piece]
    arr[index] = 1
    return arr

def encode_board(board):
    board_str = str(board)
    board_str = board_str.replace(' ', '')
    board_list = []
    for row in board_str.split('\n'):
        row_list = []
        for piece in row:
            row_list.append(one_hot_encode_peice(piece))
        board_list.append(row_list)

    return np.array(board_list)

def board_to_tensor(board):
    encoded = encode_board(board)
    tensor = torch.tensor(encoded, dtype=torch.float32).view(-1)

    return tensor.unsqueeze(0)