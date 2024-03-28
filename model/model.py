"""
File that holds the Efficiently updatable neural network architecture for the model.
"""

import torch
import torch.nn as nn


class ChessNNUE(nn.Module):
    def __init__(self):
        super(ChessNNUE, self).__init__()
        self.input_layer = nn.Linear(768, 8)
        self.hidden_layer1 = nn.Linear(8, 8)
        self.hidden_layer2 = nn.Linear(8, 8)
        self.output_layer = nn.Linear(8, 1)
        self.clipped_relu = nn.ReLU()

    def forward(self, x):
        x = self.clipped_relu(self.input_layer(x))
        x = self.clipped_relu(self.hidden_layer1(x))
        x = self.clipped_relu(self.hidden_layer2(x))
        x = self.output_layer(x)  # Output without Clipped ReLU
        return x
