"""
Pipeline to preprocess the converted Stockfish generated files, in .plain format, into Marlinflow legacy format (fen | eval | wdl) 
"""

# Place all files you wish to convert in the model/pipe/plain folder
# Run this script to convert all files in the plain folder to the marlinflow folder

import os


def convert_result_to_wdl(result):
    """Converts game result to WDL format."""
    if result == 1:
        return 1.0  # Win
    elif result == -1:
        return 0.0  # Loss
    else:
        return 0.5  # Draw


def process_file(input_path, output_path):
    with open(input_path, "r") as input_file, open(output_path, "w") as output_file:
        while True:
            fen_line = input_file.readline().strip()
            if not fen_line:  # End of file
                break
            move_line = input_file.readline().strip()
            score_line = input_file.readline().strip()
            ply_line = input_file.readline().strip()
            result_line = input_file.readline().strip()
            empty_line = input_file.readline().strip()  # Read the 'e' line

            # Extract the FEN string and the result
            fen = fen_line.split(" ", 1)[1]
            result = int(result_line.split(" ")[1])
            score = int(score_line.split(" ")[1])

            # Convert result to WDL format
            wdl = convert_result_to_wdl(result)

            # Write to output file
            output_file.write(f"{fen} | {score} | {wdl}\n")


def process_all_files(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".plain"):
            input_path = os.path.join(input_folder, filename)
            output_filename = filename.replace(".plain", ".txt")
            output_path = os.path.join(output_folder, output_filename)
            process_file(input_path, output_path)
            print(f"Conversion completed for {filename}")


# Paths to the input and output folders
input_folder = "model/pipe/plain"
output_folder = "model/pipe/marlinflow"

# Process all the files
process_all_files(input_folder, output_folder)
print("Conversions completed.")
