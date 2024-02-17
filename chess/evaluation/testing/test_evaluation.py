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
        self.board.set_piece_at(chess.C6, chess.Piece(chess.PAWN, chess.WHITE)) # 135 for white

        self.board.set_piece_at(chess.H5, chess.Piece(chess.KING, chess.BLACK)) # -15 for black
        
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


if __name__ == '__main__':
    unittest.main()
