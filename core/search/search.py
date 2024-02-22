import chess
import time
import random
from ..evaluation.evaluate import Evaluation
from .transposition_table import TranspositionTable


class Search:
    posInf = 999999999999
    negInf = -posInf

    def __init__(self):
        self.eval = Evaluation()
        self.board = None
        self.transposition_table = TranspositionTable()
        self.tt_hits = 0  # track the number of times the transposition table is used for performance evaluation

    def minimax(
        self, board, depth, is_maximizing, alpha=-float("inf"), beta=float("inf")
    ):
        current_hash = self.transposition_table.generate_initial_hash(board)

        # Use the transposition table for pruning, not for directly returning moves
        if current_hash in self.transposition_table.table:
            self.tt_hits += 1
            stored_score = self.transposition_table.table[current_hash]
            if is_maximizing and stored_score <= alpha:
                return stored_score, None  # Prune
            if not is_maximizing and stored_score >= beta:
                return stored_score, None  # Prune

        if depth == 0 or board.is_game_over():
            score = self.eval.evaluate(board)
            self.transposition_table.table[current_hash] = score
            return score, None

        if is_maximizing:
            max_eval = self.negInf
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
            # Optionally update the transposition table here
            return max_eval, best_move
        else:
            min_eval = self.posInf
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
            # Optionally update the transposition table here
            return min_eval, best_move
