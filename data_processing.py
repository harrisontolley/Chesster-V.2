"""
File that preprocesses the data into a form that can be used by the model trainer.
"""
import chess.pgn
import numpy as np
import time

from youtube import split_dims, stockfish


def process_pgn_file(pgn_file_path, max_games=500, analysis_depth=3):
    board_matrices = []
    evaluations = []
    transposition_table = {}  # Transposition table to store board evaluations
    game_count = 0
    start = time.time()

    with open(pgn_file_path) as pgn:
        while game_count < max_games:
            print(f"Processing game {game_count + 1}...")
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                board_fen = board.fen()
                if board_fen in transposition_table:
                    # Use the stored evaluation if available
                    evaluation = transposition_table[board_fen]
                else:
                    try:
                        evaluation = stockfish(board, analysis_depth)
                        if evaluation is None:
                            evaluation = 0  # Default value for problematic positions
                        transposition_table[board_fen] = evaluation
                    except Exception as e:
                        print("Error during analysis:", e)
                        evaluation = 0

                board_matrix = split_dims(board)
                board_matrices.append(board_matrix)
                evaluations.append(evaluation)

            game_count += 1

    end = time.time()
    print(f"Processed {game_count} games in {end - start} seconds.")
    print(f"Average time per game: {(end - start) / game_count} seconds.")
    return np.array(board_matrices), np.array(evaluations)


# Main script
print("Processing PGN file...")
start_time = time.time()

pgn_file_path = "./data/dec_2017_5gb.pgn"  # Update with your PGN file path
board_matrices, evaluations = process_pgn_file(pgn_file_path)

# Normalize evaluations
normalized_evaluations = (evaluations - np.min(evaluations)) / (
    np.max(evaluations) - np.min(evaluations)
)

# Save the processed data
np.savez("chess_data.npz", b=board_matrices, v=normalized_evaluations)

end_time = time.time()
print(f"Processing completed in {end_time - start_time} seconds.")
