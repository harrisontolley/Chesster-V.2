import random


def shuffle_large_file(input_path, output_path):
    """
    Reads the content of a large file, shuffles the lines, and writes them back to a new file.

    Args:
        input_path (str): Path to the input file containing the data to shuffle.
        output_path (str): Path to the output file to write the shuffled data.
    """
    try:
        # Read all lines from the file
        with open(input_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        # Shuffle the lines in memory
        random.shuffle(lines)

        # Write the shuffled lines to the output file
        with open(output_path, "w", encoding="utf-8") as file:
            file.writelines(lines)

        print("File has been shuffled and saved to", output_path)
    except Exception as e:
        print("An error occurred:", e)


# Example usage
input_path = "./data/70mildepth3.txt"
output_path = "./data/shuffled_70mildepth3.txt"
shuffle_large_file(input_path, output_path)
