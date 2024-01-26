"""
File that preprocesses the data into a form that can be used by the model trainer.
"""
import chess.pgn
import numpy as np
import time
import concurrent.futures
import threading

from youtube import split_dims, stockfish


def process_game(game, analysis_depth, transposition_table, lock):
    board_matrices = []
    evaluations = []

    if game is None or len(list(game.mainline_moves())) == 0:
        return board_matrices, evaluations

    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
        board_fen = board.fen()

        with lock:  # Lock the following block of code
            if board_fen in transposition_table:
                evaluation = transposition_table[board_fen]
                print("Found in transposition table:", evaluation)
            else:
                try:
                    evaluation = stockfish(board, analysis_depth)
                    if evaluation is None:
                        evaluation = 0
                    transposition_table[board_fen] = evaluation
                    print("Evaluated:", evaluation)
                except Exception as e:
                    print("Error during analysis:", e)
                    evaluation = 0

        board_matrix = split_dims(board)
        board_matrices.append(board_matrix)
        evaluations.append(evaluation)

    return board_matrices, evaluations


def process_pgn_file(pgn_file_path, max_games=5000, analysis_depth=3, max_workers=10):
    games_data = []
    transposition_table = {}
    lock = threading.Lock()

    with open(pgn_file_path) as pgn:
        games = [
            chess.pgn.read_game(pgn)
            for _ in range(max_games)
            if chess.pgn.read_game(pgn) is not None
        ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_game = {
            executor.submit(
                process_game, game, analysis_depth, transposition_table, lock
            ): game
            for game in games
        }

        processed_games = 0
        for future in concurrent.futures.as_completed(future_to_game):
            game_data = future.result()
            if game_data[0]:  # Only add data if it's not empty
                games_data.append(game_data)

            processed_games += 1
            print(f"Processed {processed_games}/{max_games} games.")

    # Ensure that we have data to concatenate
    if games_data:
        board_matrices = np.concatenate([data[0] for data in games_data])
        evaluations = np.concatenate([data[1] for data in games_data])
    else:
        board_matrices = np.array([])
        evaluations = np.array([])

    return board_matrices, evaluations


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
