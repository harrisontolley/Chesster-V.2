import chess


def chess_position_to_vector(board: chess.Board) -> list:
    # Initialize a 768-length vector with all zeros
    vector = [0] * 768

    # Iterate over all squares on the chessboard
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            # Determine color (0 for white, 1 for black) and adjust for piece type and color
            color = int(piece.color)
            piece_type = (
                piece.piece_type - 1
            )  # Adjust because piece types in python-chess start at 1

            # Calculate base index for this piece type and color
            base_index = color * 6 * 64 + piece_type * 64

            # Determine the adjusted square index
            # In Rust, square^56 is used to flip the square index for black's perspective
            # In Python, we achieve the same by using chess.square_mirror for black
            if color == chess.BLACK:
                square = chess.square_mirror(square)

            # Calculate the final index for the vector and set it to 1
            vector_index = base_index + square
            vector[vector_index] = 1

    return vector


def validate_vector(vector, board):
    assert sum(vector) == len(
        board.piece_map()
    ), "Vector sum does not match piece count on board."
