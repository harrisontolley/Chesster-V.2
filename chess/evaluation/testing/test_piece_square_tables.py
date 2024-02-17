import sys
import os
import unittest
import chess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from piece_square_tables import PieceSquareTables


class TestPieceSquareTables(unittest.TestCase):
    def setUp(self):
        self.piece_square_tables = PieceSquareTables()

    def test_flip_table_length(self):
        for table in self.piece_square_tables.tables[1:7]:  # Only test original tables
            flipped_table = self.piece_square_tables.flip_table(table)
            self.assertEqual(len(flipped_table), 64, "Flipped table should have 64 elements")

    def test_flip_table_corners(self):
        for table in self.piece_square_tables.tables[1:7]:  # Only test original tables
            flipped_table = self.piece_square_tables.flip_table(table)
            self.assertEqual(table[0], flipped_table[56], "Top-left corner should match bottom-left after flip")
            self.assertEqual(table[7], flipped_table[63], "Top-right corner should match bottom-right after flip")
            self.assertEqual(table[56], flipped_table[0], "Bottom-left corner should match top-left after flip")
            self.assertEqual(table[63], flipped_table[7], "Bottom-right corner should match top-right after flip")

    def test_flip_table_center(self):
        for table in self.piece_square_tables.tables[1:7]:  # Only test original tables
            flipped_table = self.piece_square_tables.flip_table(table)
            # Test a square in the center and its vertically mirrored counterpart
            self.assertEqual(table[27], flipped_table[36], "Center squares should mirror each other")
            self.assertEqual(table[28], flipped_table[35], "Center squares should mirror each other")

    def test_flip_table_identity(self):
        for table in self.piece_square_tables.tables[1:7]:  # Only test original tables
            flipped_table = self.piece_square_tables.flip_table(table)
            double_flipped_table = self.piece_square_tables.flip_table(flipped_table)
            # After flipping twice, the table should be identical to the original
            self.assertEqual(table, double_flipped_table, "Double flipping should return to the original table")

    def test_flip_table_functionality(self):
        # Create a mock piece square table that can be flipped and easily verified
        mock_table = list(range(64))  # This represents squares from 0 to 63
        flipped_table = self.piece_square_tables.flip_table(mock_table)
        
        # Check if flipping the table mirrors it vertically as expected
        for i in range(64):
            row = i // 8
            column = i % 8
            flipped_row = 7 - row
            expected_flipped_index = flipped_row * 8 + column
            self.assertEqual(flipped_table[i], expected_flipped_index, f"Index {i} flipped incorrectly")



if __name__ == '__main__':
    unittest.main()
