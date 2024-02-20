import chess
import time
import random
from ..evaluation.evaluate import Evaluation


class Search:
    posInf = 999999999999
    negInf = -posInf

    def __init__(self):
        self.eval = Evaluation()
        self.board = None

    def minimax(
        self, board, depth, is_maximizing, alpha=-float("inf"), beta=float("inf")
    ):
        if board.is_checkmate():
            return (
                (self.negInf + depth, None)
                if is_maximizing
                else (self.posInf - depth, None)
            )
        elif board.is_stalemate() or board.is_insufficient_material() or depth == 0:
            return (self.eval.evaluate(board), None)

        self.board = board

        legal_moves = list(board.legal_moves)
        if len(legal_moves) == 1:
            return (0, legal_moves[0])

        if is_maximizing:
            max_eval = -float("inf")
            best_move = None
            for move in board.legal_moves:
                board.push(move)
                eval, _ = self.minimax(board, depth - 1, False, alpha, beta)
                board.pop()

                if eval > max_eval:
                    max_eval = eval
                    best_move = move

                alpha = max(alpha, eval)
                if beta <= alpha:
                    break

            return (max_eval, best_move)
        else:
            min_eval = float("inf")
            best_move = None
            for move in board.legal_moves:
                board.push(move)
                eval, _ = self.minimax(board, depth - 1, True, alpha, beta)
                board.pop()

                if eval < min_eval:
                    min_eval = eval
                    best_move = move

                beta = min(beta, eval)
                if beta <= alpha:
                    break

            return (min_eval, best_move)
