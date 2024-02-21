import chess
from core.evaluation import Evaluation
from core.search.search import Search
import time
import requests

# def main():
#     board = chess.Board()
#     search = Search()

#     valid_white_turn = ["w", "white", "White", "WHITE"]
#     valid_black_turn = ["b", "black", "Black", "BLACK"]

#     # Prompt for user color
#     user_color_input = input("Enter the color of the player (white/black): ").lower()
#     while user_color_input not in valid_white_turn + valid_black_turn:
#         user_color_input = input(
#             "Invalid color. Please enter 'white' or 'black': "
#         ).lower()

#     # Assign colors
#     user_color = chess.WHITE if user_color_input in valid_white_turn else chess.BLACK
#     bot_color = chess.BLACK if user_color == chess.WHITE else chess.WHITE

#     print(f"You are playing as {'white' if user_color == chess.WHITE else 'black'}.")

#     # Game loop
#     while not board.is_game_over():
#         print(board)
#         if board.turn == user_color:
#             move_san = input("Enter your move: ")
#             print(board.legal_moves)
#             try:
#                 move = board.parse_san(move_san)
#                 if move in board.legal_moves:
#                     board.push(move)
#                 else:
#                     raise ValueError
#             except ValueError:
#                 print("Invalid move. Please try again.")
#                 continue
#         else:
#             print("Bot is thinking...")
#             start = time.time()
#             _, move = search.minimax(board, 5, board.turn == bot_color)
#             end = time.time()
#             print(f"Time taken: {end - start:.2f}s")
#             if move:
#                 board.push(move)
#                 print(f"Bot plays: {move.uci()}")
#             else:
#                 print("No valid move found by bot.")

#     # Game result
#     print(f"Game over. Result: {board.result()}")


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

    # Game loop for bot vs. bot
    move_count = 0  # Just to track the number of moves
    total_time = 0
    print(board)
    while not board.is_game_over():
        print("Bot is thinking...")
        start = time.time()
        move = None

        if len(board.piece_map()) <= 7:
            best_move_uci = query_tablebase(board.fen())
            if best_move_uci:
                move = chess.Move.from_uci(best_move_uci)
            else:
                print("Failed to retrieve tablebase information.")

        if move is None:  # If tablebase move is not available
            _, move = search.minimax(board, 5, board.turn == chess.WHITE)

        end = time.time()

        if move:
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
