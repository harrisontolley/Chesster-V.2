"""
Pipeline to preprocess the converted Stockfish generated files, in .plain format, into Marlinflow legacy format (fen | eval | wdl) 
"""

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


# Paths
input_path = "model/pipe/1mildepth5.plain"
output_path = "model/pipe/marlinflow.txt"

# Process the file
process_file(input_path, output_path)
print("Conversion completed.")
