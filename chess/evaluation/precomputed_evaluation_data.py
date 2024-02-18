import chess

class PrecomputedEvaluationData:
    PawnShieldSquaresWhite = [[] for _ in range(64)]
    PawnShieldSquaresBlack = [[] for _ in range(64)]

    @staticmethod
    def create_pawn_shield():
        for squareIndex in range(64):
            file = squareIndex % 8
            rank = squareIndex // 8
            white_shield, black_shield = [], []

            for file_offset in range(-1, 2):
                if 0 <= file + file_offset < 8:
                    if rank + 1 < 8:
                        white_shield.append(chess.square(file + file_offset, min(rank + 1, 7)))
                    if rank + 2 < 8:
                        white_shield.append(chess.square(file + file_offset, min(rank + 2, 7)))
                    if rank - 1 >= 0:
                        black_shield.append(chess.square(file + file_offset, max(rank - 1, 0)))
                    if rank - 2 >= 0:
                        black_shield.append(chess.square(file + file_offset, max(rank - 2, 0)))

            PrecomputedEvaluationData.PawnShieldSquaresWhite[squareIndex] = white_shield
            PrecomputedEvaluationData.PawnShieldSquaresBlack[squareIndex] = black_shield


PrecomputedEvaluationData.create_pawn_shield()