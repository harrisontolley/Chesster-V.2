import chess


def chess_position_to_vector(board: chess.Board) -> list:
    # Initialize a 768-length vector with all zeros
    vector = [0] * 768

    # Iterate over all squares on the chessboard
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            # Determine color (0 for white, 1 for black)
            color = int(piece.color)
            # Map the piece type to an index (Pawn: 0, Knight: 1, ..., King: 5)
            piece_index = piece.piece_type - 1
            # Calculate the base index for this piece type and color
            base_index = color * 6 * 64 + piece_index * 64
            # Adjust the index based on the square, flipping the row index for black
            if color == chess.WHITE:
                vector_index = base_index + square
            else:
                vector_index = base_index + (square ^ 56)

            # Set the corresponding vector element to 1
            vector[vector_index] = 1

    return vector
