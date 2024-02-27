import chess
from core.evaluation import Evaluation
from core.search.search import Search
import time
import requests


def query_tablebase(fen):  # Query the tablebase when 7 pieces or less
    url = "http://tablebase.lichess.ovh/standard"  # table base api
    params = {"fen": fen.replace(" ", "_")}
    response = requests.get(url, params=params)
    if response.ok:
        data = response.json()
        if "moves" in data and data["moves"]:
            best_move_uci = data["moves"][0]["uci"]
            return best_move_uci
        else:
            print("No moves available or not a tablebase position.")
            return None
    else:
        print("Error:", response.status_code)
        return None


def main():
    board = chess.Board()
    search = Search()

    move_count = 0  # Just to track the number of moves
    total_time = 0
    print(board)
    while not board.is_game_over():
        print("Bot is thinking...")
        start = time.time()

        if (
            board.piece_map().__len__() <= 7
        ):  # Query the tablebase when 7 pieces or less
            fen = board.fen()
            best_move = query_tablebase(fen)
        else:
            best_move = search.search(board, 4)

        end = time.time()

        # Make the best move
        if best_move:
            move = (
                chess.Move.from_uci(best_move)
                if isinstance(best_move, str)
                else best_move
            )
            board.push(move)
            move_count += 1
            total_time += end - start
            print(f"Move {move_count}: {move.uci()} | Time taken: {end - start:.2f}s")
        else:
            print("No valid move found by bot.")
            break

        print(board)

    # Game result
    average_time_per_move = total_time / move_count if move_count else 0
    print(f"Game over. Result: {board.result()}")
    print(f"Total moves: {move_count}")
    print(f"Average time per move: {average_time_per_move:.2f}s")
    print(f"Board fen: {board.fen()}")


if __name__ == "__main__":
    main()
