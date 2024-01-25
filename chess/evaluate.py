import chess


def evaluate(board):
    piece_values = {
        "P": 1,
        "N": 3,
        "B": 3,
        "R": 5,
        "Q": 9,
        "K": 0,
        "p": -1,
        "n": -3,
        "b": -3,
        "r": -5,
        "q": -9,
        "k": 0,
    }
    total_value = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            total_value += piece_values[piece.symbol()]
    return total_value
