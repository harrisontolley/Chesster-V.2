import chess
from ..evaluation.evaluate import Evaluation
from .transposition_table import TranspositionTable


class Search:
    def __init__(self):
        self.eval = Evaluation()
        self.posInf = (
            Evaluation.CHECKMATE_SCORE
        )  # Assuming CHECKMATE_SCORE is a large positive number representing checkmate
        self.negInf = -self.posInf
        self.tt = TranspositionTable()

    def order_moves(self, board):
        scored_moves = []
        for move in board.legal_moves:
            score = 0
            # Score capturing moves
            if board.is_capture(move):
                capture_piece_type = board.piece_type_at(move.to_square)
                move_piece_type = board.piece_type_at(move.from_square)
                score += 10 * (
                    self.get_piece_value(capture_piece_type)
                    - self.get_piece_value(move_piece_type)
                )
            # Score promoting a pawn
            if move.promotion:
                score += 9 * self.get_piece_value(
                    move.promotion
                )  # Queen promotion is most common

            scored_moves.append((score, move))

        # Sort moves based on score, highest first
        scored_moves.sort(reverse=True, key=lambda x: x[0])

        # Return a list of moves, ordered by score
        ordered_moves = [move for _, move in scored_moves]
        return ordered_moves

    def get_piece_value(self, piece_type):
        if piece_type == chess.PAWN:
            return Evaluation.PAWN_VALUE
        if piece_type == chess.KNIGHT:
            return Evaluation.KNIGHT_VALUE
        if piece_type == chess.BISHOP:
            return Evaluation.BISHOP_VALUE
        if piece_type == chess.ROOK:
            return Evaluation.ROOK_VALUE
        if piece_type == chess.QUEEN:
            return Evaluation.QUEEN_VALUE
        return 0

    def search(self, board, depth):
        legal_moves = self.order_moves(board)
        best_move = None
        best_score = self.negInf
        for move in legal_moves:
            board.push(move)
            score = -self.minimax(board, depth, -1000000, 1000000)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def minimax(self, board, depth, alpha, beta):
        if depth == 0:
            return self.eval.evaluate(board)

        legal_moves = self.order_moves(board)
        if len(legal_moves) == 0:
            if board.is_checkmate() or board.is_check():
                return self.negInf
            return 0

        for move in legal_moves:
            board.push(move)
            eval = -self.minimax(board, depth - 1, -beta, -alpha)
            board.pop()
            if eval >= beta:
                return beta
            alpha = max(alpha, eval)

        return alpha
