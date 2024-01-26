import time
import chess
import chess.engine
import random
import numpy

stockfish_path = "./stockfish/stockfish.exe"


# this function will create our x (board)
def random_board(max_depth=200):
    board = chess.Board()
    depth = random.randrange(0, max_depth)

    for _ in range(depth):
        all_moves = list(board.legal_moves)
        random_move = random.choice(all_moves)
        board.push(random_move)

        if board.is_game_over():
            break

    return board


def stockfish(board, depth):
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as sf:
        result = sf.analyse(board, chess.engine.Limit(depth=depth))
        score = result["score"].white().score()
        if score is None:
            print("SCORE IS NONE BOZO")
            print(board)
            print(
                board.is_game_over(),
                board.is_checkmate(),
                board.is_stalemate(),
                board.is_insufficient_material(),
                board.is_seventyfive_moves(),
                board.is_fivefold_repetition(),
                board.is_variant_end(),
                board.is_valid(),
            )
        return score


# # board = random_board(500)
# # print(board)
# # print(stockfish(board, 10))

squares_index = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}


def square_to_index(square):
    letter = chess.square_name(square)
    return 8 - int(letter[1]), squares_index[letter[0]]


def split_dims(board):
    # this is the 3d array that will be returned
    board3d = numpy.zeros((14, 8, 8), dtype=numpy.uint8)

    # here we add the pieces to the board
    for piece in chess.PIECE_TYPES:
        for square in board.pieces(piece, chess.WHITE):
            idx = numpy.unravel_index(square, (8, 8))
            board3d[piece - 1][7 - idx[0]][idx[1]] = 1
        for square in board.pieces(piece, chess.BLACK):
            idx = numpy.unravel_index(square, (8, 8))
            board3d[piece + 5][7 - idx[0]][idx[1]] = 1

    # add attacks and valid moves too
    # so the network knows what is being attacked and what moves are valid
    aux = board.turn
    board.turn = chess.WHITE

    for move in board.legal_moves:
        i, j = square_to_index(move.from_square)
        board3d[12][i][j] = 1
    board.turn = chess.BLACK
    for move in board.legal_moves:
        i, j = square_to_index(move.from_square)
        board3d[13][i][j] = 1
    board.turn = aux

    return board3d


import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.utils as utils


def build_model(conv_size, conv_depth):
    board3d = layers.Input(shape=(14, 8, 8))  # 14 planes of 8x8

    x = board3d
    for _ in range(conv_depth):
        x = layers.Conv2D(
            filters=conv_size, kernel_size=3, padding="same", activation="relu"
        )(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(1, activation="sigmoid")(x)

    return models.Model(inputs=board3d, outputs=x)


model = build_model(32, 4)
print(model.summary())
# utils.plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)


def build_model_residual(conv_size, conv_depth):
    board3d = layers.Input(shape=(14, 8, 8))  # 14 planes of 8x8

    # adding the conv layers
    x = layers.Conv2D(
        filters=conv_size, kernel_size=3, padding="same", data_format="channels_first"
    )(board3d)
    for _ in range(conv_depth):
        previous_x = x
        x = layers.Conv2D(
            filters=conv_size,
            kernel_size=3,
            padding="same",
            data_format="channels_first",
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(
            filters=conv_size,
            kernel_size=3,
            padding="same",
            data_format="channels_first",
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, previous_x])
        x = layers.Activation("relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1, activation="sigmoid")(x)

    return models.Model(inputs=board3d, outputs=x)


import tensorflow.keras.callbacks as callbacks


def get_dataset():
    container = numpy.load("chess_data.npz")
    b, v = container["b"], container["v"]
    v = numpy.asarray(v / abs(v).max() / 2 + 0.5, dtype=numpy.float32)  # -1..1 -> 0..1
    return b, v


# x_train, y_train = get_dataset()
# # print(x_train.shape, y_train.shape)

# model.compile(optimizer="adam", loss="mean_squared_error")
# model.summary()
# model.fit(
#     x_train,
#     y_train,
#     batch_size=2048,
#     epochs=20,
#     verbose=1,
#     validation_split=0.1,
#     callbacks=[
#         callbacks.ReduceLROnPlateau(monitor="loss", patience=10),
#         callbacks.EarlyStopping(monitor="loss", patience=15, min_delta=0.001),
#     ],
# )
# model.save("model.keras")


# def minimax_eval(board):
#     board3d = split_dims(board)
#     board3d = numpy.expand_dims(board3d, 0)
#     return model.predict(board3d, verbose=0)[0][0]


# def minimax(board, depth, alpha, beta, maximising_player):
#     if depth == 0 or board.is_game_over():
#         return minimax_eval(board)

#     if maximising_player:
#         max_eval = -numpy.inf
#         for move in board.legal_moves:
#             board.push(move)
#             eval = minimax(board, depth - 1, alpha, beta, False)
#             board.pop()
#             max_eval = max(max_eval, eval)
#             alpha = max(alpha, eval)
#             if beta <= alpha:
#                 break
#         return max_eval
#     else:
#         min_eval = numpy.inf
#         for move in board.legal_moves:
#             board.push(move)
#             eval = minimax(board, depth - 1, alpha, beta, True)
#             board.pop()
#             min_eval = min(min_eval, eval)
#             beta = min(beta, eval)
#             if beta <= alpha:
#                 break
#         return min_eval


# # function that gets the move from the neural network
# def get_ai_move(board, depth):
#     max_move = None
#     max_eval = -numpy.inf

#     for move in board.legal_moves:
#         board.push(move)
#         eval = minimax(board, depth, -numpy.inf, numpy.inf, False)
#         board.pop()
#         if eval > max_eval:
#             max_eval = eval
#             max_move = move

#     return max_move


# board = chess.Board()

# start = time.time()
# with chess.engine.SimpleEngine.popen_uci(stockfish_path) as sf:
#     while not board.is_game_over():
#         # AI Move
#         ai_move = get_ai_move(board, 1)
#         board.push(ai_move)
#         # print(f"\n{board}")
#         if board.is_game_over():
#             break

#         # Stockfish Move
#         result = sf.analyse(board, chess.engine.Limit(time=1))
#         stockfish_move = result.get("pv")[
#             0
#         ]  # Get the first move from the principal variation
#         board.push(stockfish_move)
#         print(f"\n{board}")
#         print("-----------------------------------------")

# end = time.time()
# print("Time taken:", end - start)
# print("Result:", board.result())
