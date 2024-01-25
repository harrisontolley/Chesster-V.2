import time
import random

from youtube import random_board, stockfish


print("generating boards...")
start = time.time()
boards = []
for _ in range(20):
    new_board = random_board(random.randrange(1, 50, 1))
    boards.append(new_board)
end = time.time()
print("generated 10 boards in", end - start, "seconds")


max_board = None
min_board = None

boards_eval = {}

print("evaluating boards...")
start = time.time()
for board in boards:
    score = stockfish(board, 10)
    print(score)

    board_str = str(board)
    if score is not None:
        if board_str not in boards_eval:
            boards_eval[board_str] = score

        if max_board is None or score > boards_eval[str(max_board)]:
            max_board = board

        if min_board is None or score < boards_eval[str(min_board)]:
            min_board = board


end = time.time()
print("evaluated boards in", end - start, "seconds")

print("max board:", boards_eval[str(max_board)])
print(max_board)
print("min board:", boards_eval[str(min_board)])
print(min_board)
