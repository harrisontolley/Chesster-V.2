import chess
from core.evaluation import Evaluation
from core.search.search import Search
import time


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


def main():

    # board = chess.Board("8/8/8/8/3K4/2Q5/k7/8 w - - 0 1")
    # eval = Evaluation()

    # for move in board.legal_moves:
    #     board.push(move)
    #     print(board.fen(), eval.evaluate(board), move)
    #     print("EVAL STATISTICS")
    #     print(f"Material Score: {eval.whiteEval.material_score}")
    #     print(f"Mop Up Score: {eval.whiteEval.mop_up_score}")
    #     print(f"Piece Square Score: {eval.whiteEval.piece_square_score}")
    #     print(f"Pawn Score: {eval.whiteEval.pawn_score}")
    #     print("Pawn Shield Score: ", eval.whiteEval.pawn_shield_score)
    #     print()
    #     board.pop()

    board = chess.Board("8/8/8/8/3K4/2Q5/k7/8 w - - 0 1")

    search = Search()

    # Game loop for bot vs. bot
    move_count = 0  # Just to track the number of moves
    while not board.is_game_over():
        print(board)
        print("Bot is thinking...")
        start = time.time()

        _, move = search.minimax(board, 3, board.turn == chess.WHITE)
        end = time.time()

        if move:
            board.push(move)
            move_count += 1
            print(f"Move {move_count}: {move.uci()} | Time taken: {end - start:.2f}s")
        else:
            print("No valid move found by bot.")
            break

    # Game result
    print(f"Game over. Result: {board.result()}")
    print(f"Total moves: {move_count}")
    print(f"Board fen: {board.fen()}")


if __name__ == "__main__":
    main()
