import random
import chess


class TranspositionTable:
    def __init__(self):
        self.table = {}
        self.zobrist_table = [
            [random.getrandbits(64) for _ in range(12)] for _ in range(64)
        ]  # 6 piece types * 2 colors
        self.zobrist_castling = [random.getrandbits(64) for _ in range(4)]
        self.zobrist_enpassant = [random.getrandbits(64) for _ in range(8)]
        self.black_to_move = random.getrandbits(64)

    def piece_to_index(self, piece):
        """Map a chess.Piece to an index [0, 11]."""
        if piece is None:
            return None
        index = piece.piece_type - 1  # Piece types start at 1
        if piece.color == chess.BLACK:
            index += 6  # Offset for black pieces
        return index

    def generate_initial_hash(self, board):
        hash_code = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                piece_index = self.piece_to_index(piece)
                hash_code ^= self.zobrist_table[square][piece_index]

        if board.turn == chess.BLACK:
            hash_code ^= self.black_to_move

        # Castling rights
        castling_rights = [
            board.has_kingside_castling_rights(chess.WHITE),
            board.has_queenside_castling_rights(chess.WHITE),
            board.has_kingside_castling_rights(chess.BLACK),
            board.has_queenside_castling_rights(chess.BLACK),
        ]
        for i, has_right in enumerate(castling_rights):
            if has_right:
                hash_code ^= self.zobrist_castling[i]

        # En passant
        # Corrected to remove the incorrect check for en passant capture
        if board.ep_square is not None:
            file = chess.square_file(board.ep_square)
            hash_code ^= self.zobrist_enpassant[file]

        return hash_code

    def update_hash(self, hash_code, move, board_before_move, board_after_move):
        """
        Function to update the hash after a move has been made.
        """
        # XOR out the piece from starting square
        start_piece = board_before_move.piece_at(move.from_square)
        if start_piece is not None:
            start_piece_index = self.piece_to_index(start_piece)
            hash_code ^= self.zobrist_table[move.from_square][start_piece_index]

        # XOR out any captured piece (if applicable)
        if board_before_move.is_capture(move):
            end_piece = board_before_move.piece_at(move.to_square)
            if end_piece is not None:
                end_piece_index = self.piece_to_index(end_piece)
                hash_code ^= self.zobrist_table[move.to_square][end_piece_index]

        # XOR in the piece to the ending square (consider promotion)
        promoted_piece = move.promotion
        end_piece_index = self.piece_to_index(
            start_piece
            if not promoted_piece
            else chess.Piece(promoted_piece, start_piece.color)
        )
        hash_code ^= self.zobrist_table[move.to_square][end_piece_index]

        # Update side to move
        hash_code ^= self.black_to_move

        # Update castling rights if they have changed
        previous_castling = [
            board_before_move.has_kingside_castling_rights(chess.WHITE),
            board_before_move.has_queenside_castling_rights(chess.WHITE),
            board_before_move.has_kingside_castling_rights(chess.BLACK),
            board_before_move.has_queenside_castling_rights(chess.BLACK),
        ]
        current_castling = [
            board_after_move.has_kingside_castling_rights(chess.WHITE),
            board_after_move.has_queenside_castling_rights(chess.WHITE),
            board_after_move.has_kingside_castling_rights(chess.BLACK),
            board_after_move.has_queenside_castling_rights(chess.BLACK),
        ]
        for i in range(4):
            if previous_castling[i] != current_castling[i]:
                hash_code ^= self.zobrist_castling[i]

        # Update en passant square if it has changed
        if board_before_move.ep_square is not None:
            file = chess.square_file(board_before_move.ep_square)
            hash_code ^= self.zobrist_enpassant[file]

        if board_after_move.ep_square is not None:
            file = chess.square_file(board_after_move.ep_square)
            hash_code ^= self.zobrist_enpassant[file]

        return hash_code
