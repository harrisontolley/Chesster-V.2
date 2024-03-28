import chess
import torch
import os
import pickle  # Import pickle module

MAPPING = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
    chess.BLACK: 6,
}

zero_vector = [0] * 768


def fen_to_vector(fen):
    board = chess.Board(fen)
    vector = zero_vector.copy()

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_type = MAPPING[piece.piece_type]
            color_offset = 0 if piece.color == chess.WHITE else MAPPING[chess.BLACK]
            index = (piece_type + color_offset) * 64 + square
            vector[index] = 1

    return vector


def read_fens_and_convert_in_batches(input_file, output_file, batch_size=10000):
    if not os.path.exists(output_file):
        with open(output_file, "wb") as f:
            pickle.dump([], f)  # Initialize file with an empty list

    with open(input_file, "r") as file:
        vectors = []
        scores = []
        for line in file:
            parts = line.strip().split("|")
            if len(parts) < 3:
                print("Skipping line due to unexpected format:", line)
                continue
            fen = parts[0].strip()
            score = float(parts[1].strip())  # Convert score to float

            board = chess.Board(fen)
            # Negate the score if it's black's turn to move
            if board.turn == chess.BLACK:
                score = -score

            vector = fen_to_vector(fen)
            vectors.append(vector)
            scores.append(score)

            if len(vectors) >= batch_size:
                with open(output_file, "rb") as f:
                    data = pickle.load(f)
                data.append({"vectors": vectors, "scores": scores})
                with open(output_file, "wb") as f:
                    pickle.dump(data, f)
                vectors = []
                scores = []

        # Save any remaining positions not forming a complete batch
        if vectors:
            with open(output_file, "rb") as f:
                data = pickle.load(f)
            data.append({"vectors": vectors, "scores": scores})
            with open(output_file, "wb") as f:
                pickle.dump(data, f)


# Usage example
input_filename = "./data/70mildepth3.txt"
output_file = "./data/batches_data.pkl"  # Change to a single output file
batch_size = 500000  # Adjust based on your memory capacity
read_fens_and_convert_in_batches(input_filename, output_file, batch_size=batch_size)
