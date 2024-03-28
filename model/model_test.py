import sys
import os

# Add the parent directory to sys.path to find the data package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "")))

from model import ChessNNUE
import torch
from data.cleaner import fen_to_vector


print("Testing the model...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assume ChessNNUE is defined as before
model = ChessNNUE()

# Load the trained model weights
model.load_state_dict(
    torch.load("./model/outputs/chess_nnue_epoch_20.pth", map_location=device)
)
model = model.to(device)
print("Model loaded successfully.")

# Set the model to evaluation mode
model.eval()

# Example FEN string for a board state
fen_string = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

fen_strings = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkb1r/pp3ppp/4p3/1Np5/4B3/8/PPP2PPP/R1BQK2R b KQkq - 1 9",
    "r1b1kb1r/pp4pp/1q3p2/1Np1B3/2P1B3/8/PP3PPP/R2QK2R b KQkq - 0 12",
    "r4b1r/pp2k2p/1q2b1p1/2p1Q3/2P1B3/2N5/PP3PPP/2KR3R b - - 0 16",
    "r7/pp5p/3kb1pb/2pP4/4B3/8/PP3PPP/1K1R3R w - - 0 21",
    "8/8/3k4/3P4/4B3/8/PP3PPP/1K1R3R w - - 0 21",
    "8/4q3/3kq3/r1r1rrr1/8/8/P7/1K6 b - - 0 21",
]


for fen_string in fen_strings:
    board_vector = fen_to_vector(fen_string)
    board_tensor = (
        torch.tensor(board_vector, dtype=torch.float32).unsqueeze(0).to(device)
    )

    with torch.no_grad():
        evaluation = model(board_tensor)
        print("Evaluation Score:", evaluation.item())
