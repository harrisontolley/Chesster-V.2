import chess
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from chess_position_to_vector import chess_position_to_vector

HIDDEN_SIZE = 16
INPUT_SIZE = 768
OUTPUT_SIZE = 1
SCALE = 400


class ClippedReLU(nn.Module):
    def __init__(self):
        super(ClippedReLU, self).__init__()

    def forward(self, x):
        return torch.clamp(F.relu(x), max=1.0)


class SimpleNNUE(nn.Module):
    def __init__(self):
        super(SimpleNNUE, self).__init__()
        self.feature_transformer = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.output_layer = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        self.crelu = ClippedReLU()  # Use Clipped ReLU

    def forward(self, position_features):
        transformed_features = self.crelu(self.feature_transformer(position_features))
        output = self.output_layer(transformed_features)
        output = output * SCALE
        return output


# Example of creating a network and evaluating a dummy input
net = SimpleNNUE()


def load_model_params(model, params_path):
    params = torch.from_numpy(np.fromfile(params_path, dtype=np.float32))

    # Calculate the correct offsets for each parameter set
    start = 0
    end = INPUT_SIZE * HIDDEN_SIZE
    model.feature_transformer.weight.data = params[start:end].view(
        HIDDEN_SIZE, INPUT_SIZE
    )
    start = end
    end += HIDDEN_SIZE
    model.feature_transformer.bias.data = params[start:end]

    start = end
    end += HIDDEN_SIZE * OUTPUT_SIZE  # Corrected to match the single-layer output
    model.output_layer.weight.data = params[start:end].view(OUTPUT_SIZE, HIDDEN_SIZE)
    start = end
    end += OUTPUT_SIZE
    model.output_layer.bias.data = params[start:end]


model = SimpleNNUE()

load_model_params(model, "params.bin")

print(model)


def evaluate_position(model, position_features):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        eval = model(position_features)
        return eval.item()


print("-------------------------")
print("Chess board representation")
chess_board = chess.Board()

# convert to tensor
vector = chess_position_to_vector(chess_board)
vector_np = np.array(vector)  # Convert list to numpy array
position_features_tensor = torch.tensor(
    vector_np.reshape(1, INPUT_SIZE), dtype=torch.float32
)


evaluation = evaluate_position(model, position_features_tensor)
print(f"Evaluation: {evaluation}")
