import chess
from ..evaluation.evaluate import Evaluation
from .transposition_table import TranspositionTable
import copy


class Search:
    def __init__(self):
        self.eval = Evaluation()
        self.posInf = Evaluation.CHECKMATE_SCORE
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
        board_hash = self.tt.generate_initial_hash(board)  # Initialize board hash
        best_move = None
        best_score = self.negInf
        legal_moves = self.order_moves(board)
        for move in legal_moves:
            board_before_move = copy.deepcopy(board)
            board.push(move)
            new_hash = self.tt.update_hash(board_hash, move, board_before_move, board)
            score = -self.minimax(board, depth - 1, -1000000, 1000000, new_hash)
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def minimax(self, board, depth, alpha, beta, board_hash):
        # Check transposition table first
        if board_hash in self.tt.table:
            return self.tt.table[board_hash]

        if depth == 0 or board.is_game_over():
            score = self.eval.evaluate(board)
            self.tt.table[board_hash] = score
            return score

        legal_moves = self.order_moves(board)
        for move in legal_moves:
            board_before_move = copy.deepcopy(board)
            board.push(move)
            new_hash = self.tt.update_hash(board_hash, move, board_before_move, board)
            score = -self.minimax(board, depth - 1, -beta, -alpha, new_hash)
            board.pop()
            if score >= beta:
                self.tt.table[board_hash] = beta  # Store cutoff score
                return beta
            if score > alpha:
                alpha = score
        self.tt.table[board_hash] = alpha  # Store exact score or best score found
        return alpha
