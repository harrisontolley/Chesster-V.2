import chess
import time

from ..evaluation.evaluate import Evaluation


class Search:
    def __init__(self):
        self.eval = Evaluation()

    def basic_search(self, board):
        """
        Finds the best move for the given color on 1 ply.
        """
        best_move = None
        best_score = -9999

        for move in board.legal_moves:
            board.push(move)
            score = -self.eval.evaluate(board)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

        return best_move
