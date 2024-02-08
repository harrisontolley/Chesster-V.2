import torch
import torch.nn as nn
import numpy as np

HIDDEN_SIZE = 16
INPUT_SIZE = 768
OUTPUT_SIZE = 1
SCALE = 400
QA = 255


class SimpleNNUE(nn.Module):
    def __init__(self):
        super(SimpleNNUE, self).__init__()
        self.feature_transformer = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.output_layer = nn.Linear(HIDDEN_SIZE * 2, OUTPUT_SIZE)
        self.crelu = nn.ReLU()  # Using standard ReLU as an approximation to CReLU

    def forward(self, us, them):
        us_transformed = self.crelu(self.feature_transformer(us))
        them_transformed = self.crelu(self.feature_transformer(them))
        concatenated = torch.cat((us_transformed, them_transformed), dim=1)
        output = self.output_layer(concatenated)
        output = output * SCALE / (QA * QA)
        return output


# Example of creating a network and evaluating a dummy input
net = SimpleNNUE()

# Dummy inputs for 'us' and 'them', replace with actual chess position features
us_features = torch.randn(1, INPUT_SIZE)
them_features = torch.randn(1, INPUT_SIZE)

output = net(us_features, them_features)
print(output)


def load_model_params(model, params_path):
    # Assume params are stored as: (feature_transformer weights)(feature_transformer biases)(output_layer weights)...
    params = torch.from_numpy(np.fromfile(params_path, dtype=np.float32))

    # Adjust indices based on your model structure
    # Since feature_transformer connects INPUT_SIZE to HIDDEN_SIZE
    start = 0
    end = INPUT_SIZE * HIDDEN_SIZE
    model.feature_transformer.weight.data = params[start:end].view(
        HIDDEN_SIZE, INPUT_SIZE
    )
    start = end
    end += HIDDEN_SIZE
    model.feature_transformer.bias.data = params[start:end]

    # Since output_layer connects HIDDEN_SIZE * 2 to OUTPUT_SIZE
    start = end
    end += HIDDEN_SIZE * 2 * OUTPUT_SIZE
    model.output_layer.weight.data = params[start:end].view(
        OUTPUT_SIZE, HIDDEN_SIZE * 2
    )
    start = end
    end += OUTPUT_SIZE
    model.output_layer.bias.data = params[start:end]


model = SimpleNNUE()
load_model_params(model, "params.bin")

print(model)


def evaluate_position(model, position_features):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Assuming position_features represents 'us', and using the same for 'them' as a placeholder
        us_features = position_features
        them_features = (
            position_features  # In a real scenario, this should be different
        )

        # Call the model with both 'us' and 'them'
        eval = model(us_features, them_features)
        return eval.item()


# position_features should be the preprocessed features from the FEN
# For demonstration, this is a placeholder
# Ensure the input tensor is correctly shaped (1, INPUT_SIZE)
position_features = np.zeros((1, INPUT_SIZE), dtype=np.float32)
position_features_tensor = torch.tensor(position_features, dtype=torch.float32)

evaluation = evaluate_position(model, position_features_tensor)
print(f"Evaluation: {evaluation}")


# Define a mapping from piece types and colors to indices
piece_type_to_index = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5}
color_to_offset = {"white": 0, "black": 6}


# Define the initial positions of pieces in a chess game
initial_positions = [
    ("a1", "R", "white"),
    ("b1", "N", "white"),
    ("c1", "B", "white"),
    ("d1", "Q", "white"),
    ("e1", "K", "white"),
    ("f1", "B", "white"),
    ("g1", "N", "white"),
    ("h1", "R", "white"),
    ("a2", "P", "white"),
    ("b2", "P", "white"),
    ("c2", "P", "white"),
    ("d2", "P", "white"),
    ("e2", "P", "white"),
    ("f2", "P", "white"),
    ("g2", "P", "white"),
    ("h2", "P", "white"),
    ("a7", "P", "black"),
    ("b7", "P", "black"),
    ("c7", "P", "black"),
    ("d7", "P", "black"),
    ("e7", "P", "black"),
    ("f7", "P", "black"),
    ("g7", "P", "black"),
    ("h7", "P", "black"),
    ("a8", "R", "black"),
    ("b8", "N", "black"),
    ("c8", "B", "black"),
    ("d8", "Q", "black"),
    ("e8", "K", "black"),
    ("f8", "B", "black"),
    ("g8", "N", "black"),
    ("h8", "R", "black"),
]


# Generate the input feature vector for the initial chess position
def generate_initial_position_vector():
    input_vector = np.zeros(
        64 * 6 * 2, dtype=np.float32
    )  # 64 squares, 6 piece types, 2 colors
    for square, piece, color in initial_positions:
        row = 8 - int(square[1])  # Row index based on square
        col = ord(square[0]) - ord("a")  # Column index based on square
        square_index = row * 8 + col
        piece_index = piece_type_to_index[piece]
        color_offset = color_to_offset[color]
        input_index = square_index * 12 + piece_index + color_offset
        input_vector[input_index] = 1
    return input_vector


# Generate and print the input vector for the default chess position
initial_position_vector = generate_initial_position_vector()
position_features_tensor = torch.tensor(
    initial_position_vector.reshape(1, INPUT_SIZE), dtype=torch.float32
)


evaluation = evaluate_position(model, position_features_tensor)
print(f"Evaluation: {evaluation}")

# Reuse the mappings from piece types and colors to indices
piece_type_to_index = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5}
color_to_offset = {"white": 0, "black": 6}

# Define the positions of pieces in our specific winning position for White
winning_positions = [
    ("e4", "K", "white"),
    ("h5", "Q", "white"),
    ("h8", "K", "black"),
]


# Generate the input feature vector for this winning position
def generate_winning_position_vector():
    input_vector = np.zeros(
        64 * 6 * 2, dtype=np.float32
    )  # 64 squares, 6 piece types, 2 colors
    for square, piece, color in winning_positions:
        row = 8 - int(square[1])  # Row index based on square
        col = ord(square[0]) - ord("a")  # Column index based on square
        square_index = row * 8 + col
        piece_index = piece_type_to_index[piece]
        color_offset = color_to_offset[color]
        input_index = square_index * 12 + piece_index + color_offset
        input_vector[input_index] = 1
    return input_vector


# Generate and print the input vector for the winning chess position
winning_position_vector = generate_winning_position_vector()

position_features_tensor = torch.tensor(
    winning_position_vector.reshape(1, INPUT_SIZE), dtype=torch.float32
)

evaluation = evaluate_position(model, position_features_tensor)
print(f"Evaluation: {evaluation}")
