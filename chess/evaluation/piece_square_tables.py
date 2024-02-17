import chess

class PieceSquareTables:
    # Define piece square tables
    PAWNS = [
        0,   0,   0,   0,   0,   0,   0,   0,
        50,  50,  50,  50,  50,  50,  50,  50,
        10,  10,  20,  30,  30,  20,  10,  10,
        5,   5,  10,  25,  25,  10,   5,   5,
        0,   0,   0,  20,  20,   0,   0,   0,
        5,  -5, -10,   0,   0, -10,  -5,   5,
        5,  10,  10, -20, -20,  10,  10,   5,
        0,   0,   0,   0,   0,   0,   0,   0
    ]

    PAWNSENDGAME = [
        0,   0,   0,   0,   0,   0,   0,   0,
        80,  80,  80,  80,  80,  80,  80,  80,
        50,  50,  50,  50,  50,  50,  50,  50,
        30,  30,  30,  30,  30,  30,  30,  30,
        20,  20,  20,  20,  20,  20,  20,  20,
        10,  10,  10,  10,  10,  10,  10,  10,
        10,  10,  10,  10,  10,  10,  10,  10,
        0,   0,   0,   0,   0,   0,   0,   0
    ]

    KNIGHTS = [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50,
    ]

    BISHOPS = [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20,
    ]

    ROOKS =  [
        0,  0,  0,  0,  0,  0,  0,  0,
        5, 10, 10, 10, 10, 10, 10,  5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        0,  0,  0,  5,  5,  0,  0,  0
    ]

    QUEENS = [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
         -5,  0,  5,  5,  5,  5,  0, -5,
          0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20    
    ]

    KINGSTART = [
        -80, -70, -70, -70, -70, -70, -70, -80, # 63
        -60, -60, -60, -60, -60, -60, -60, -60, # 55
        -40, -50, -50, -60, -60, -50, -50, -40, # 47
        -30, -40, -40, -50, -50, -40, -40, -30, # 39
        -20, -30, -30, -40, -40, -30, -30, -20, # 31
        -10, -20, -20, -20, -20, -20, -20, -10, # 23
        20,  20,  -5,  -5,  -5,  -5,  20,  20, # 15
        20,  30,  10,   0,   0,  10,  30,  20 # 7
    ]

    KINGENDGAME = [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -5,   0,   5,   5,   5,   5,   0,  -5,
        -10, -5,   20,  30,  30,  20,  -5, -10,
        -15, -10,  35,  45,  45,  35, -10, -15,
        -20, -15,  30,  40,  40,  30, -15, -20,
        -25, -20,  20,  25,  25,  20, -20, -25,
        -30, -25,   0,   0,   0,   0, -25, -30,
        -50, -30, -30, -30, -30, -30, -30, -50
    ]

    def __init__(self):
        self.tables = [[] for _ in range(17)]
        self.tables[1] = self.PAWNS
        self.tables[2] = self.KNIGHTS
        self.tables[3] = self.BISHOPS
        self.tables[4] = self.ROOKS
        self.tables[5] = self.QUEENS
        self.tables[6] = self.KINGSTART
        self.tables[7] = self.flip_table(self.PAWNS)
        self.tables[8] = self.flip_table(self.KNIGHTS)
        self.tables[9] = self.flip_table(self.BISHOPS)
        self.tables[10] = self.flip_table(self.ROOKS)
        self.tables[11] = self.flip_table(self.QUEENS)
        self.tables[12] = self.flip_table(self.KINGSTART)
        self.tables[13] = self.PAWNSENDGAME
        self.tables[14] = self.flip_table(self.PAWNSENDGAME)
        self.tables[15] = self.KINGENDGAME
        self.tables[16] = self.flip_table(self.KINGENDGAME)

    # Flips a piece square table for black pieces
    def flip_table(self, table):
        flippedTable = [0 for _ in range(64)]
        for idx in range(64):
            file = chess.square_file(idx)
            rank = chess.square_rank(idx)
            rank = 7 - rank
            flippedIdx = chess.square(file, rank)
            flippedTable[flippedIdx] = table[idx]
        return flippedTable

    def read(self, table, colour, squareIndex):
        if colour == chess.WHITE:
            helper = PieceSquareHelper()
            file = helper.file_index(squareIndex)
            rank = helper.rank_index(squareIndex)
            rank = 7 - rank
            squareIndex = helper.index_from_coord(file, rank)
            return table[squareIndex]
        else:
            return table[63 - squareIndex]


class PieceSquareHelper():
    def file_index(self, squareIndex):
        return squareIndex % 8

    def rank_index(self, squareIndex):
        return squareIndex // 8

    def index_from_coord(self, file, rank):
        return rank * 8 + file