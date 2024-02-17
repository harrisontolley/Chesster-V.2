import chess
import time

from chess.evaluation.evaluate import evaluate


def minimax(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        return evaluate(board)

    if maximizing_player:
        max_eval = float("-inf")
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float("inf")
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


def find_best_move(board, depth):
    best_move = None
    best_value = float("-inf")
    for move in board.legal_moves:
        board.push(move)
        move_value = minimax(board, depth - 1, float("-inf"), float("inf"), False)
        board.pop()
        if move_value > best_value:
            best_value = move_value
            best_move = move
    return best_move