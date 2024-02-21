class PrecomputedMoveData:
    def __init__(self):
        # Initialize the list with zeros for all 64 squares
        self.centre_manhattan_distance = [0] * 64
        self.orthogonal_distance = [[0 for _ in range(64)] for _ in range(64)]
        self.king_distance = [[0 for _ in range(64)] for _ in range(64)]
        self.initiate_values()

    def initiate_values(self):
        for squareA in range(64):
            rankA, fileA = divmod(squareA, 8)

            # Calculate the Manhattan distance from the center for each square
            file_distance_from_center = max(3 - fileA, fileA - 4)
            rank_distance_from_center = max(3 - rankA, rankA - 4)
            self.centre_manhattan_distance[squareA] = (
                file_distance_from_center + rank_distance_from_center
            )

            for squareB in range(64):
                rankB, fileB = divmod(squareB, 8)
                # Calculate orthogonal and king distances
                self.orthogonal_distance[squareA][squareB] = abs(rankA - rankB) + abs(
                    fileA - fileB
                )
                self.king_distance[squareA][squareB] = max(
                    abs(rankA - rankB), abs(fileA - fileB)
                )
