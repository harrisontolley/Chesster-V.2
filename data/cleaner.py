import chess
import torch
import os
import pickle  # Import pickle module
import time  # Import time module for timekeeping

MAPPING = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
    chess.BLACK: 6,
}

INVERSE_SQUARES = {
    chess.H8: chess.A1,
    chess.G8: chess.B1,
    chess.F8: chess.C1,
    chess.E8: chess.D1,
    chess.D8: chess.E1,
    chess.C8: chess.F1,
    chess.B8: chess.G1,
    chess.A8: chess.H1,
    chess.H7: chess.A2,
    chess.G7: chess.B2,
    chess.F7: chess.C2,
    chess.E7: chess.D2,
    chess.D7: chess.E2,
    chess.C7: chess.F2,
    chess.B7: chess.G2,
    chess.A7: chess.H2,
    chess.H6: chess.A3,
    chess.G6: chess.B3,
    chess.F6: chess.C3,
    chess.E6: chess.D3,
    chess.D6: chess.E3,
    chess.C6: chess.F3,
    chess.B6: chess.G3,
    chess.A6: chess.H3,
    chess.H5: chess.A4,
    chess.G5: chess.B4,
    chess.F5: chess.C4,
    chess.E5: chess.D4,
    chess.D5: chess.E4,
    chess.C5: chess.F4,
    chess.B5: chess.G4,
    chess.A5: chess.H4,
    chess.H4: chess.A5,
    chess.G4: chess.B5,
    chess.F4: chess.C5,
    chess.E4: chess.D5,
    chess.D4: chess.E5,
    chess.C4: chess.F5,
    chess.B4: chess.G5,
    chess.A4: chess.H5,
    chess.H3: chess.A6,
    chess.G3: chess.B6,
    chess.F3: chess.C6,
    chess.E3: chess.D6,
    chess.D3: chess.E6,
    chess.C3: chess.F6,
    chess.B3: chess.G6,
    chess.A3: chess.H6,
    chess.H2: chess.A7,
    chess.G2: chess.B7,
    chess.F2: chess.C7,
    chess.E2: chess.D7,
    chess.D2: chess.E7,
    chess.C2: chess.F7,
    chess.B2: chess.G7,
    chess.A2: chess.H7,
    chess.H1: chess.A8,
    chess.G1: chess.B8,
    chess.F1: chess.C8,
    chess.E1: chess.D8,
    chess.D1: chess.E8,
    chess.C1: chess.F8,
    chess.B1: chess.G8,
    chess.A1: chess.H8,
}


zero_vector = [0] * 768


# def fen_to_vector(fen):
#     board = chess.Board(fen)
#     vector = zero_vector.copy()

#     for square in chess.SQUARES:
#         piece = board.piece_at(square)
#         if piece:
#             piece_type = MAPPING[piece.piece_type]
#             color_offset = 0 if piece.color == chess.WHITE else MAPPING[chess.BLACK]
#             index = (piece_type + color_offset) * 64 + square
#             vector[index] = 1

#     return vector


def side_to_move_to_vector(fen: str, color_to_move: chess.Color) -> torch.Tensor:
    board = chess.Board(fen)
    vector = zero_vector.copy()

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_type = MAPPING[piece.piece_type]
            color_offset = 0 if piece.color == color_to_move else MAPPING[chess.BLACK]
            # Adjust index calculation based on color and whether it is the color to move
            index_square = (
                square
                if piece.color == color_to_move
                else INVERSE_SQUARES.get(square, square)
            )
            index = (piece_type + color_offset) * 64 + index_square
            vector[index] = 1

    return torch.tensor(vector, dtype=torch.float32)


# def white_to_move_to_vector(fen: str) -> torch.Tensor:
#     board = chess.Board(fen)
#     vector = zero_vector.copy()

#     for square in chess.SQUARES:
#         piece = board.piece_at(square)
#         if piece:
#             piece_type = MAPPING[piece.piece_type]
#             color_offset = 0 if piece.color == chess.WHITE else MAPPING[chess.BLACK]
#             index = (piece_type + color_offset) * 64 + square
#             vector[index] = 1

#     return torch.tensor(vector, dtype=torch.float32)


# def black_to_move_to_vector(fen: str) -> torch.Tensor:
#     board = chess.Board(fen)
#     vector = zero_vector.copy()

#     for square in chess.SQUARES:
#         piece = board.piece_at(square)
#         if piece:
#             piece_type = MAPPING[piece.piece_type]
#             color_offset = 0 if piece.color == chess.BLACK else MAPPING[chess.BLACK]
#             index = (piece_type + color_offset) * 64 + INVERSE_SQUARES[square]
#             vector[index] = 1

#     return torch.tensor(vector, dtype=torch.float32)


def read_fens_and_convert_in_batches(
    input_file, output_file, batch_size=10000, limit=None
):
    """
    Reads FENs from input_file, converts them to vectors, and stores them in batches.
    Optionally stops after converting a specified limit of vectors.

    Args:
        input_file (str): Path to the file containing FEN strings.
        output_file (str): Path to the file where the batches will be stored.
        batch_size (int): The number of vectors to process before saving a batch.
        limit (int, optional): The maximum number of vectors to process. If None, processes all vectors.
    """
    start_time = time.time()  # Start timekeeping

    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        try:
            with open(output_file, "rb") as f:
                data = pickle.load(f)  # Attempt to load existing data
        except EOFError:
            print(
                "Output file exists but is corrupt or empty. Initializing new data structure."
            )
            data = []  # Initialize as empty list if file is corrupt or causes EOFError
    else:
        print(
            "Output file does not exist or is empty. Initializing new data structure."
        )
        data = []

    total_lines_processed = 0
    vectors_processed = 0  # Initialize counter for processed vectors
    with open(input_file, "r") as file:
        vectors = []
        scores = []
        for line in file:
            if limit is not None and vectors_processed >= limit:
                print(f"Reached limit of {limit} vectors. Stopping.")
                break  # Break out of the loop if the limit is reached

            parts = line.strip().split("|")
            if len(parts) < 3:
                print("Skipping line due to unexpected format:", line)
                continue

            fen = parts[0].strip()

            stm = parts[0].split(" ")[1]
            if stm == "w":
                stm = chess.WHITE
            else:
                stm = chess.BLACK

            score = float(parts[1].strip())  # Convert score to float

            vector = side_to_move_to_vector(fen, stm)
            vectors.append(vector)
            scores.append(score)
            total_lines_processed += 1
            vectors_processed += 1  # Increment counter for processed vectors

            if len(vectors) >= batch_size:
                data.append({"vectors": vectors, "scores": scores})
                with open(output_file, "wb") as f:
                    pickle.dump(data, f)
                print(
                    f"Processed and saved batch of {len(vectors)} vectors. Total lines processed so far: {total_lines_processed}"
                )
                vectors = []
                scores = []

        # Save any remaining positions not forming a complete batch
        if vectors:
            data.append({"vectors": vectors, "scores": scores})
            with open(output_file, "wb") as f:
                pickle.dump(data, f)
            print(
                f"Processed and saved final batch of {len(vectors)} vectors. Total lines processed: {total_lines_processed}"
            )

    end_time = time.time()  # End timekeeping
    print(f"Finished processing. Time taken: {end_time - start_time:.2f} seconds.")


# Usage example
# input_filename = "./data/70mildepth3.txt"
# output_file = "./data/batches_data.pkl"  # Change to a single output file
# batch_size = 50000  # Adjust based on your memory capacity
# read_fens_and_convert_in_batches(
#     input_filename, output_file, batch_size=batch_size, limit=100000
# )
