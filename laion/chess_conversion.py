import chess
import chess.pgn
import re
import json
from multiprocessing import Pool

input_file_path = '6gb_chess.jsonl'
output_file_path = '6gb_chess.txt'

# Compile the regex outside the loop
move_format_regex = re.compile(r'(\d+)\. ')

def process_line(line):
    data = json.loads(line)
    moves_list = data["text"]

    # Convert the list of moves to a PGN string
    board = chess.Board()
    for move_uci in moves_list:
        move = chess.Move.from_uci(move_uci)
        board.push(move)

    # Create a game from the board's moves
    game = chess.pgn.Game.from_board(board)

    # Generate the PGN string without headers
    pgn_exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
    pgn_str = str(game.accept(pgn_exporter))

    # Remove the header and result
    pgn_lines = pgn_str.splitlines()
    pgn_moves_only = ' '.join(pgn_lines[:-1])

    # Remove the spaces after the move numbers
    formatted_pgn = move_format_regex.sub(r'\1.', pgn_moves_only)

    # Trimming the string if its length exceeds 1023
    if len(formatted_pgn) > 1023:
        formatted_pgn = formatted_pgn[:1023]

    return formatted_pgn + '\n'

def main():
    with open(input_file_path, 'r') as f, open(output_file_path, 'a') as out_f:
        with Pool() as p: # Use all available CPUs
            for result in p.imap(process_line, f, chunksize=1000):  # Adjust chunksize based on your observation
                out_f.write(result)

if __name__ == "__main__":
    main()
