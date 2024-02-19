import chess
from core.evaluation import Evaluation
from core.search.search import Search


def main():
    board = chess.Board()
    search = Search()

    valid_white_turn = ["w", "white", "White", "WHITE"]
    valid_black_turn = ["b", "black", "Black", "BLACK"]

    # Prompt for user color
    user_color_input = input("Enter the color of the player (white/black): ").lower()
    while user_color_input not in valid_white_turn + valid_black_turn:
        user_color_input = input(
            "Invalid color. Please enter 'white' or 'black': "
        ).lower()

    # Assign colors
    user_color = chess.WHITE if user_color_input in valid_white_turn else chess.BLACK
    bot_color = chess.BLACK if user_color == chess.WHITE else chess.WHITE

    print(f"You are playing as {'white' if user_color == chess.WHITE else 'black'}.")

    # Game loop
    while not board.is_game_over():
        print(board)
        if board.turn == user_color:
            move_san = input("Enter your move: ")
            print(board.legal_moves)
            try:
                move = board.parse_san(move_san)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    raise ValueError
            except ValueError:
                print("Invalid move. Please try again.")
                continue
        else:
            print("Bot is thinking...")
            _, move = search.minimax(board, 5, board.turn == bot_color)
            if move:
                board.push(move)
                print(f"Bot plays: {move.uci()}")
            else:
                print("No valid move found by bot.")

    # Game result
    print(f"Game over. Result: {board.result()}")


if __name__ == "__main__":
    main()
