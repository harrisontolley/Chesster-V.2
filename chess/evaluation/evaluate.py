import chess
from piece_square_tables import PieceSquareTables

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

    def __init__(self):
        self.board = None  # Placeholder for board instance
        self.whiteEval = self.EvaluationData()
        self.blackEval = self.EvaluationData()
        self.piece_square_tables = PieceSquareTables()

    def get_board(self):
        return self.board

    def evaluate(self, board):
        self.board = board
        self.whiteEval = self.EvaluationData()
        self.blackEval = self.EvaluationData()

        # calculate material
        whiteMaterialInfo = self.get_material_info(chess.WHITE)
        blackMaterialInfo = self.get_material_info(chess.BLACK)

        self.whiteEval.material_score = whiteMaterialInfo.material_score
        self.blackEval.material_score = blackMaterialInfo.material_score

        self.whiteEval.piece_square_score = self.evaluate_piece_square_tables(chess.WHITE, blackMaterialInfo.endgameT)
        self.blackEval.piece_square_score = self.evaluate_piece_square_tables(chess.BLACK, whiteMaterialInfo.endgameT)

    def get_material_info(self, color):
        """
        Returns the sum of material for the given color
        """
        num_pawns = len(self.get_board().pieces(chess.PAWN, color))
        num_knights = len(self.get_board().pieces(chess.KNIGHT, color))
        num_bishops = len(self.get_board().pieces(chess.BISHOP, color))
        num_queens = len(self.get_board().pieces(chess.QUEEN, color))
        num_rooks = len(self.get_board().pieces(chess.ROOK, color))
        my_pawns = len(self.get_board().pieces(chess.PAWN, color))
        enemy_pawns = len(self.get_board().pieces(chess.PAWN, not color))

        return self.MaterialInfo(
            num_pawns,
            num_knights,
            num_bishops,
            num_queens,
            num_rooks,
            my_pawns,
            enemy_pawns,
        )

    def evaluate_piece_square_tables(self, color, endgameT):
        value = 0
        colourIndex = 0 if color == chess.WHITE else 6

        # add the piece square table for the pieces for the given color
        value += self.evaluate_piece_square_table(self.piece_square_tables.tables[colourIndex + chess.KNIGHT], color, chess.KNIGHT)
        value += self.evaluate_piece_square_table(self.piece_square_tables.tables[colourIndex + chess.BISHOP], color, chess.BISHOP)
        value += self.evaluate_piece_square_table(self.piece_square_tables.tables[colourIndex + chess.ROOK], color, chess.ROOK)
        value += self.evaluate_piece_square_table(self.piece_square_tables.tables[colourIndex + chess.QUEEN], color, chess.QUEEN)

        # interpolate between early and late game 
        pawnEarlyGame = self.evaluate_piece_square_table(self.piece_square_tables.tables[colourIndex + chess.PAWN], color, chess.PAWN)
        pawnsLateGame = self.evaluate_piece_square_table(self.piece_square_tables.tables[13 + (not color)], color, chess.PAWN)
        value += pawnEarlyGame * (1 - endgameT) + pawnsLateGame * endgameT

        kingEarlyGame = self.evaluate_piece_square_table(self.piece_square_tables.tables[colourIndex + chess.KING], color, chess.KING)
        kingLateGame = self.evaluate_piece_square_table(self.piece_square_tables.tables[15 + (not color)], color, chess.KING)
        value += kingEarlyGame * (1 - endgameT) + kingLateGame * endgameT

        return value

    def evaluate_piece_square_table(self, table, color, piece):
        value = 0
        for square in self.get_board().pieces(piece, color):
            value += self.piece_square_tables.read(table, color, square)
            print("adding value: ", self.piece_square_tables.read(table, color, square))
            print("from piece: ", piece, " at square: ", square, " for color: ", color)
            print("value: ", value)
            print("--------------------")
        return value


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
                num_pawns * Evaluation.PAWN_VALUE
                + num_knights * Evaluation.KNIGHT_VALUE
                + num_bishops * Evaluation.BISHOP_VALUE
                + num_rooks * Evaluation.ROOK_VALUE
                + num_queens * Evaluation.QUEEN_VALUE
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

