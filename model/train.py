import time
import torch
import torch.nn as nn
import torch.optim as optim
from model import ChessNNUE
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Assuming you have already defined the ChessNNUE model
    model = ChessNNUE().to(device)

    # Define a loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Number of epochs to train the model
    epochs = 1

    # Directory where batch files are stored
    batches_dir = "./data/batches"
    batch_files = sorted(
        [
            os.path.join(batches_dir, f)
            for f in os.listdir(batches_dir)
            if f.startswith("training_data_batch_")
        ]
    )

    # List to store loss for plotting
    epoch_losses = []

    # Training loop
    start = time.time()
    print("Training Started")
    for epoch in range(epochs):
        epoch_start = time.time()
        running_loss = 0.0
        for batch_file in batch_files:
            print(f"Training on batch file: {batch_file}")
            # Load training data for the current batch
            vectors_tensor, scores_tensor = torch.load(batch_file)
            vectors_tensor = vectors_tensor.to(device)
            scores_tensor = scores_tensor.to(device)

            # Use DataLoader for batching
            dataset = torch.utils.data.TensorDataset(vectors_tensor, scores_tensor)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

            for inputs, labels in dataloader:
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)

                # Calculate loss
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

        avg_epoch_loss = running_loss / len(batch_files)
        epoch_losses.append(avg_epoch_loss)

        # Save the model after every epoch
        model_save_path = f"./model/outputs/chess_nnue_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path} after epoch {epoch + 1}")

        epoch_end = time.time()
        print(f"Epoch {epoch + 1}, Loss: {avg_epoch_loss}")
        print(f"Time taken for epoch: {epoch_end - epoch_start:.2f} seconds")

    end = time.time()
    print(f"Training Finished. Time elapsed: {end - start:.2f} seconds")

    # Plotting the loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), epoch_losses, marker="o", label="Training Loss")
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("./loss_over_epochs.png")
    plt.show()


if __name__ == "__main__":
    train()
