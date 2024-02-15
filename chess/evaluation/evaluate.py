import chess


class Evaluation:
    PAWN_VALUE = 100
    KNIGHT_VALUE = 300
    BISHOP_VALUE = 320
    ROOK_VALUE = 500
    QUEEN_VALUE = 900

    passed_pawn_bonuses = [0, 120, 80, 50, 30, 15, 15]
    isolated_pawn_penalty_by_count = [0, -10, -25, -50, -75, -75, -75, -75, -75]
    king_pawn_shield_scores = [4, 7, 4, 3, 6, 3]

    endgame_material_start = ROOK_VALUE * 2 + BISHOP_VALUE + KNIGHT_VALUE

    class EvaluationData:
        def __init__(self):
            self.material_score = 0
            self.mop_up_score = 0
            self.piece_square_score = 0
            self.pawn_score = 0
            self.pawn_shield_score = 0

        def sum(self):
            return (
                self.material_score
                + self.mop_up_score
                + self.piece_square_score
                + self.pawn_score
                + self.pawn_shield_score
            )

    class MaterialInfo:
        def __init__(
            self,
            num_pawns,
            num_knights,
            num_bishops,
            num_queens,
            num_rooks,
            my_pawns,
            enemy_pawns,
        ):
            self.num_pawns = num_pawns
            self.num_bishops = num_bishops
            self.num_queens = num_queens
            self.num_rooks = num_rooks
            self.pawns = my_pawns
            self.enemy_pawns = enemy_pawns

            self.num_majors = num_rooks + num_queens
            self.num_minors = num_bishops + num_knights

            self.material_score = (
                num_pawns * self.PAWN_VALUE
                + num_knights * self.KNIGHT_VALUE
                + num_bishops * self.BISHOP_VALUE
                + num_rooks * self.ROOK_VALUE
                + num_queens * self.QUEEN_VALUE
            )

            queen_endgame_weight = 45
            rook_endgame_weight = 20
            bishop_endgame_weight = 10
            knight_endgame_weight = 10
            endgame_start_weight = (
                2 * rook_endgame_weight
                + 2 * bishop_endgame_weight
                + 2 * knight_endgame_weight
                + queen_endgame_weight
            )
            endgame_weight_sum = (
                num_queens * queen_endgame_weight
                + num_rooks * rook_endgame_weight
                + num_bishops * bishop_endgame_weight
                + num_knights * knight_endgame_weight
            )

            self.endgameT = 1 - min(1, endgame_weight_sum / float(endgame_start_weight))

    def __init__(self):
        self.board = None  # Placeholder for board instance
        self.white_eval = self.EvaluationData()
        self.black_eval = self.EvaluationData()

    def get_board(self):
        return self.board

    def evaluate(self, board):
        self.board = board
        self.white_eval = self.EvaluationData()
        self.black_eval = self.EvaluationData()

        # calculate white material

    def get_material_info(self, color):
        """
        Returns the sum of material for the given color
        """
        num_pawns = self.get_board().pieces(chess.PAWN, color)
        num_knights = self.get_board().pieces(chess.KNIGHT, color)
        num_bishops = self.get_board().pieces(chess.BISHOP, color)
        num_queens = self.get_board().pieces(chess.QUEEN, color)
        num_rooks = self.get_board().pieces(chess.ROOK, color)
        my_pawns = self.get_board().pawns(color)
        enemy_pawns = self.get_board().pawns(not color)

        return self.MaterialInfo(
            num_pawns,
            num_knights,
            num_bishops,
            num_queens,
            num_rooks,
            my_pawns,
            enemy_pawns,
        )


test = chess.Board()

print(len(test.pieces(chess.KNIGHT, chess.WHITE)))
