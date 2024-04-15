"""
File to read and display data from a pickle file. Used for debugging and testing purposes.
"""

import pickle


def read_and_display_pkl_data(file_path, num_batches_to_display=1):
    """
    Reads data from a pickle file and displays the specified number of batches.

    Args:
        file_path (str): Path to the pickle file.
        num_batches_to_display (int): Number of batches to display from the data.
    """
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            print(f"Total batches in file: {len(data)}")
            for i in range(min(num_batches_to_display, len(data))):
                batch = data[i]
                print(f"\nBatch {i+1}:")
                print(f"Number of vectors in batch: {len(batch['vectors'])}")
                print("Example vector tensor size:", batch["vectors"][0].shape)
                print("Example score:", batch["scores"][0])
                print("Example tensor:", batch["vectors"][0])
    except FileNotFoundError:
        print("File not found. Please ensure the file path is correct.")
    except Exception as e:
        print("An error occurred while reading the pickle file:", str(e))


# Example usage:
file_path = "./data/batches_data.pkl"
read_and_display_pkl_data(file_path)
