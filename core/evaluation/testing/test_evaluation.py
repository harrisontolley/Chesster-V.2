import sys
import os
import unittest
import chess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from piece_square_tables import PieceSquareTables
from evaluate import Evaluation


class TestEvaluatePieceSquareTables(unittest.TestCase):
    def setUp(self):
        self.evaluation = Evaluation()
        self.board = chess.Board()

    def test_starting_position_evaluation(self):
        # Setup the starting position
        self.board.reset()

        # Test evaluation for the starting position (e.g., endgameT close to 0)
        self.evaluation.evaluate(self.board)
        white_score = self.evaluation.whiteEval.piece_square_score
        black_score = self.evaluation.blackEval.piece_square_score

        # Verify that the evaluation is correct based on the piece-square tables for the starting position
        self.assertEqual(white_score, black_score)
        self.assertEqual(white_score, -95.0)

    def test_early_game_evaluation(self):
        # Setup an early game scenario
        self.board.reset()
        self.board.push_san("e4")
        self.board.push_san("e5")
        self.board.push_san("Nf3")

        # Test evaluation for an early game scenario (e.g., endgameT close to 0)
        self.evaluation.evaluate(self.board)
        white_score = self.evaluation.whiteEval.piece_square_score
        black_score = self.evaluation.blackEval.piece_square_score

        # Verify that the evaluation is correct based on the piece-square tables for early game
        self.assertEqual(white_score, -5)
        self.assertEqual(black_score, -55)
        self.assertNotEqual(white_score, black_score)

    def test_endgame_evaluation(self):
        # Setup an endgame scenario
        self.board.clear_board()
        self.board.set_piece_at(chess.E7, chess.Piece(chess.KING, chess.WHITE))
        self.board.set_piece_at(chess.D7, chess.Piece(chess.PAWN, chess.WHITE))
        self.board.set_piece_at(
            chess.C6, chess.Piece(chess.PAWN, chess.WHITE)
        )  # 135 for white

        self.board.set_piece_at(
            chess.H5, chess.Piece(chess.KING, chess.BLACK)
        )  # -15 for black

        # Test evaluation for an endgame scenario (e.g., endgameT close to 1)
        self.evaluation.evaluate(self.board)
        white_score = self.evaluation.whiteEval.piece_square_score
        black_score = self.evaluation.blackEval.piece_square_score

        # Verify that the evaluation is correct based on the piece-square tables for endgame
        self.assertEqual(white_score, 135.0)
        self.assertEqual(black_score, -20.0)
        self.assertNotEqual(white_score, black_score)

    def test_symmetrical_position_evaluation(self):
        # Setup a symmetrical position
        self.board.clear_board()
        self.board.set_piece_at(chess.D4, chess.Piece(chess.QUEEN, chess.WHITE))
        self.board.set_piece_at(chess.D5, chess.Piece(chess.QUEEN, chess.BLACK))

        # Evaluate the symmetrical position
        self.evaluation.evaluate(self.board)
        white_score = self.evaluation.whiteEval.piece_square_score
        black_score = self.evaluation.blackEval.piece_square_score

        # Verify that the evaluations are equal in a symmetrical position
        self.assertEqual(white_score, black_score)


class TestPawnStructureEvaluation(unittest.TestCase):
    def setUp(self):
        self.evaluation = Evaluation()
        self.board = chess.Board()

    def test_isolated_pawn_evaluation(self):
        # Setup a position with isolated pawns for white
        self.board.clear()
        self.board.set_piece_at(chess.E4, chess.Piece(chess.PAWN, chess.WHITE))
        self.board.set_piece_at(chess.D5, chess.Piece(chess.PAWN, chess.BLACK))

        # Evaluate the position
        self.evaluation.evaluate(self.board)

        # Expect a specific penalty for the isolated white pawn
        expected_penalty = self.evaluation.isolated_pawn_penalty_by_count[1]
        print("Expected penalty: ", expected_penalty)
        actual_white_pawn_score = self.evaluation.whiteEval.pawn_score
        print("Actual white pawn score: ", actual_white_pawn_score)

        self.assertEqual(
            actual_white_pawn_score,
            expected_penalty,
            "Isolated pawn penalty not correctly evaluated for white",
        )

    def test_passed_pawn_evaluation(self):
        # Setup a position with a passed pawn for white
        self.board.clear()
        self.board.set_piece_at(chess.D5, chess.Piece(chess.PAWN, chess.WHITE))
        self.board.set_piece_at(chess.E7, chess.Piece(chess.KING, chess.BLACK))
        self.board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))

        # Evaluate the position
        self.evaluation.evaluate(self.board)

        # Expect a specific bonus for the passed white pawn
        expected_bonus = (
            self.evaluation.passed_pawn_bonuses[3]
            + self.evaluation.isolated_pawn_penalty_by_count[1]
        )
        actual_white_pawn_score = self.evaluation.whiteEval.pawn_score

        self.assertEqual(
            actual_white_pawn_score,
            expected_bonus,
            "Passed pawn bonus not correctly evaluated for white",
        )

    def test_multiple_isolated_and_passed_pawns(self):
        # Setup a position with multiple scenarios
        self.board.clear()
        self.board.set_piece_at(chess.A2, chess.Piece(chess.PAWN, chess.WHITE))
        self.board.set_piece_at(
            chess.B5, chess.Piece(chess.PAWN, chess.WHITE)
        )  # Passed pawn
        self.board.set_piece_at(
            chess.H2, chess.Piece(chess.PAWN, chess.WHITE)
        )  # Isolated pawn
        self.board.set_piece_at(
            chess.H7, chess.Piece(chess.PAWN, chess.BLACK)
        )  # Passed pawn
        self.board.set_piece_at(
            chess.G7, chess.Piece(chess.PAWN, chess.BLACK)
        )  # Isolated pawn

        # Evaluate the position
        self.evaluation.evaluate(self.board)

        # Calculate expected scores
        expected_white_pawn_score = self.evaluation.passed_pawn_bonuses[
            3
        ]  # for the D5 pawn
        expected_white_pawn_score += self.evaluation.passed_pawn_bonuses[
            6
        ]  # for the B5 pawn
        expected_white_pawn_score += self.evaluation.isolated_pawn_penalty_by_count[
            1
        ]  # for the H2 isolated pawn

        expected_black_pawn_score = 0

        actual_white_pawn_score = self.evaluation.whiteEval.pawn_score
        actual_black_pawn_score = self.evaluation.blackEval.pawn_score

        self.assertEqual(
            actual_white_pawn_score,
            expected_white_pawn_score,
            "White pawn structure not correctly evaluated",
        )
        self.assertEqual(
            actual_black_pawn_score,
            expected_black_pawn_score,
            "Black pawn structure not correctly evaluated",
        )


class TestCheckmateScore(unittest.TestCase):
    def test_black_is_checkmated(self):
        # Create a board where Black is checkmated
        board = chess.Board("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1")
        evaluation = Evaluation()

        # Evaluate the position
        score = evaluation.evaluate(board)

        # Assert that the score is CHECKMATE_SCORE for White (positive score)
        self.assertEqual(
            score,
            -1000000001044.0,
            "Score should be CHECKMATE_SCORE for White winning.",
        )

    def test_white_is_checkmated(self):
        # Create a board where White is checkmated
        board = chess.Board("8/8/8/8/8/5k2/6q1/7K w - - 0 1")
        evaluation = Evaluation()

        # Evaluate the position
        score = evaluation.evaluate(board)

        # Assert that the score is -CHECKMATE_SCORE for Black (negative score)
        self.assertEqual(
            score,
            -1000000001044.0,
            "Score should be -CHECKMATE_SCORE for Black winning.",
        )


if __name__ == "__main__":
    unittest.main()
