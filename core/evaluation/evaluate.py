import chess
from .piece_square_tables import PieceSquareTables
from .precomputed_evaluation_data import PrecomputedEvaluationData
from .precomputed_move_data import PrecomputedMoveData


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

    CHECKMATE_SCORE = 999999999999

    def __init__(self):
        self.board = None  # Placeholder for board instance
        self.whiteEval = self.EvaluationData()
        self.blackEval = self.EvaluationData()

        self.piece_square_tables = PieceSquareTables()
        self.precomputed_move_data = PrecomputedMoveData()

    def get_board(self):
        return self.board

    def evaluate(self, board):
        if board.is_checkmate():
            # Return a positive score if the opponent is checkmated,
            # negative if the side to move is checkmated
            return (
                Evaluation.CHECKMATE_SCORE
                if board.turn == chess.WHITE
                else -Evaluation.CHECKMATE_SCORE
            )
        elif (
            board.is_stalemate()
            or board.is_insufficient_material()
            or board.is_seventyfive_moves()
            or board.is_fivefold_repetition()
        ):
            # Return 0 for a draw
            return 0

        self.board = board
        self.whiteEval = self.EvaluationData()
        self.blackEval = self.EvaluationData()

        # calculate material
        whiteMaterialInfo = self.get_material_info(chess.WHITE)
        blackMaterialInfo = self.get_material_info(chess.BLACK)

        # plain material score advantage
        self.whiteEval.material_score = whiteMaterialInfo.material_score
        self.blackEval.material_score = blackMaterialInfo.material_score

        # score based on position of pieces
        self.whiteEval.piece_square_score = self.evaluate_piece_square_tables(
            chess.WHITE, blackMaterialInfo.endgameT
        )
        self.blackEval.piece_square_score = self.evaluate_piece_square_tables(
            chess.BLACK, whiteMaterialInfo.endgameT
        )

        # mop up score for both colors
        # encouraging the engine to centralise kings in winning endgames
        self.whiteEval.mop_up_score = self.mop_up_eval(
            chess.WHITE, whiteMaterialInfo, blackMaterialInfo
        )
        self.blackEval.mop_up_score = self.mop_up_eval(
            chess.BLACK, blackMaterialInfo, whiteMaterialInfo
        )

        # evaluate pawn structure for both colors
        self.whiteEval.pawn_score = self.evaluate_pawns(chess.WHITE)
        self.blackEval.pawn_score = self.evaluate_pawns(chess.BLACK)

        # evaluate king pawn shield for both colors
        self.whiteEval.pawn_shield_score = self.evaluate_king_pawn_shield(
            chess.WHITE, blackMaterialInfo, self.blackEval.piece_square_score
        )
        self.blackEval.pawn_shield_score = self.evaluate_king_pawn_shield(
            chess.BLACK, whiteMaterialInfo, self.whiteEval.piece_square_score
        )

        if board.turn == chess.WHITE:
            perspective = 1
        else:
            perspective = -1

        evalSum = self.whiteEval.sum() - self.blackEval.sum()
        return evalSum * perspective

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
        value += self.evaluate_piece_square_table(
            self.piece_square_tables.tables[colourIndex + chess.KNIGHT],
            color,
            chess.KNIGHT,
        )
        value += self.evaluate_piece_square_table(
            self.piece_square_tables.tables[colourIndex + chess.BISHOP],
            color,
            chess.BISHOP,
        )
        value += self.evaluate_piece_square_table(
            self.piece_square_tables.tables[colourIndex + chess.ROOK], color, chess.ROOK
        )
        value += self.evaluate_piece_square_table(
            self.piece_square_tables.tables[colourIndex + chess.QUEEN],
            color,
            chess.QUEEN,
        )

        # interpolate between early and late game
        pawnEarlyGame = self.evaluate_piece_square_table(
            self.piece_square_tables.tables[colourIndex + chess.PAWN], color, chess.PAWN
        )
        pawnsLateGame = self.evaluate_piece_square_table(
            self.piece_square_tables.tables[13 + (not color)], color, chess.PAWN
        )
        value += int(pawnEarlyGame * (1 - endgameT))
        value += int(pawnsLateGame * endgameT)

        kingEarlyGame = self.evaluate_piece_square_table(
            self.piece_square_tables.tables[colourIndex + chess.KING], color, chess.KING
        )
        kingLateGame = self.evaluate_piece_square_table(
            self.piece_square_tables.tables[15 + (not color)], color, chess.KING
        )

        value += int(kingEarlyGame * (1 - endgameT))
        value += int(kingLateGame * endgameT)

        return value

    def evaluate_piece_square_table(self, table, color, piece):
        value = 0
        for square in self.get_board().pieces(piece, color):
            value += self.piece_square_tables.read(table, color, square)
        return value

    def mop_up_eval(
        self,
        color,
        myMaterial: "Evaluation.MaterialInfo",
        enemyMaterial: "Evaluation.MaterialInfo",
    ):
        """
        Returns the mop up score for the given color
        """
        if (
            myMaterial.material_score
            > enemyMaterial.material_score + self.PAWN_VALUE * 2
        ) and enemyMaterial.endgameT > 0:
            mopUpScore = 0

            if color == chess.WHITE:
                friendlyKingSquare = self.get_board().king(chess.WHITE)
                enemyKingSquare = self.get_board().king(chess.BLACK)
            else:
                friendlyKingSquare = self.get_board().king(chess.BLACK)
                enemyKingSquare = self.get_board().king(chess.WHITE)

            mopUpScore += (
                14
                - self.precomputed_move_data.orthogonal_distance[friendlyKingSquare][
                    enemyKingSquare
                ]
            ) * 4

            # Centralise the king in the endgame
            mopUpScore += (
                self.precomputed_move_data.centre_manhattan_distance[enemyKingSquare]
                * 10
            )

            return int(mopUpScore * enemyMaterial.endgameT)
        return 0

    def evaluate_pawns(self, color):
        """
        Returns the pawn score for the given color
        """
        value = 0
        pawns = self.get_board().pieces(chess.PAWN, color)

        num_isolated_pawns = 0

        for pawn_square in pawns:
            if self.is_passed_pawn(pawn_square, color):
                # passed pawn bonus
                if color == chess.WHITE:
                    value += self.passed_pawn_bonuses[
                        7 - chess.square_rank(pawn_square)
                    ]
                else:
                    value += self.passed_pawn_bonuses[chess.square_rank(pawn_square)]

            if self.is_isolated_pawn(pawn_square, color):
                # isolated pawn penalty
                num_isolated_pawns += 1

        value += self.isolated_pawn_penalty_by_count[num_isolated_pawns]

        return value

    def is_passed_pawn(self, square, color):
        """
        Returns True if the pawn on the given square is a passed pawn for the given color
        """
        opposing_color = not color
        pawn_rank = chess.square_rank(square)
        pawn_file = chess.square_file(square)

        # assume passed unless proven otherwise
        passed = True

        end_rank = 7 if color == chess.WHITE else 0
        rank_range = (
            range(pawn_rank + 1, end_rank)
            if color == chess.WHITE
            else range(pawn_rank - 1, end_rank, -1)
        )

        # loop through ranks and files to determine if opposing pawn is blocking
        for rank in rank_range:
            for file in [pawn_file - 1, pawn_file, pawn_file + 1]:
                if file >= 0 and file <= 7:

                    target_sqare = chess.square(file, rank)
                    if self.get_board().piece_at(target_sqare) == chess.Piece(
                        chess.PAWN, opposing_color
                    ):
                        passed = False
                        break
            if not passed:
                break

        return passed

    def is_isolated_pawn(self, square, color):
        """
        Returns True if the pawn on the given square is an isolated pawn for the given color
        """
        pawn_file = chess.square_file(square)

        # assume isolated unless proven otherwise
        isolated = True

        for file in [pawn_file - 1, pawn_file + 1]:
            if file >= 0 and file <= 7:
                adjacent_pawn = self.get_board().pieces(
                    chess.PAWN, color
                ) & chess.SquareSet(chess.BB_FILES[file])

                if adjacent_pawn:
                    isolated = False
                    break
        return isolated

    def evaluate_king_pawn_shield(
        self, color, enemy_material: "Evaluation.MaterialInfo", enemy_piece_square_score
    ):
        """
        Returns the king pawn shield score for the given color
        """
        if enemy_material.endgameT >= 1:
            return 0

        penalty = 0
        uncastled_king_penalty = 0
        is_white = color == chess.WHITE
        king_square = self.get_board().king(color)
        friendly_pawn = chess.Piece(chess.PAWN, color)

        # Check if the king is present on the board - purely for testing purposes
        if king_square is None:
            return 0

        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)

        if king_file <= 2 or king_file >= 5:
            squares = (
                PrecomputedEvaluationData.PawnShieldSquaresWhite[king_square]
                if is_white
                else PrecomputedEvaluationData.PawnShieldSquaresBlack[king_square]
            )

            for i, shield_square in enumerate(squares):
                if self.get_board().piece_at(shield_square) != friendly_pawn:
                    penalty += self.king_pawn_shield_scores[
                        min(i, len(self.king_pawn_shield_scores) - 1)
                    ]  # Apply penalty based on shield position

            penalty = penalty**2

        else:
            enemy_development_score = max(
                0, min((enemy_piece_square_score + 10) / 130, 1)
            )
            uncastled_king_penalty = 50 * enemy_development_score

        open_file_against_king_penalty = 0

        if enemy_material.num_rooks > 1 or (
            enemy_material.num_rooks > 0 and enemy_material.num_queens > 0
        ):

            clamped_king_file = max(1, min(king_file, 6))
            my_pawns = enemy_material.enemy_pawns

            for attack_file in range(clamped_king_file - 1, clamped_king_file + 2):
                if color == chess.WHITE:
                    if all(
                        not self.get_board().piece_at(chess.square(attack_file, rank))
                        for rank in range(king_rank, 8)
                    ):
                        if attack_file == king_file:
                            open_file_against_king_penalty += 25
                        else:
                            open_file_against_king_penalty += 15

        pawn_shield_weight = 1 - enemy_material.endgameT
        if len(self.get_board().pieces(chess.QUEEN, not color)) == 0:  # no enemy queens
            pawn_shield_weight *= 0.6

        return (
            -penalty - uncastled_king_penalty - open_file_against_king_penalty
        ) * pawn_shield_weight

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
