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

    def minimax(
        self, board, depth, is_maximizing, alpha=-float("inf"), beta=float("inf")
    ):
        if board.is_game_over():
            # Assign checkmate and stalemate scores
            if board.is_checkmate():
                return (-float("inf"), None) if is_maximizing else (float("inf"), None)
            else:
                return (0, None)  # Draw or stalemate

        if depth == 0 or board.is_game_over():
            return self.eval.evaluate(board), None

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
            return max_eval, best_move
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
            return min_eval, best_move
