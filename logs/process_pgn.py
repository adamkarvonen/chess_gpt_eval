import chess.pgn
import csv
import re
import os
from tqdm import tqdm


def write_pgn_to_csv(pgn_file_path: str, csv_file_path: str, total_games_estimate: int):
    """
    Extract White ELO, Black ELO, and PGN string from a .pgn file, transform the PGN string,
    and store them in a CSV file.

    :param pgn_file_path: Path to the .pgn file.
    :param csv_file_path: Path to the output CSV file.
    :param total_games_estimate: Estimated total number of games in the PGN file.
    """
    with open(pgn_file_path, "r") as pgn_file, open(
        csv_file_path, "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["WhiteElo", "BlackElo", "Result", "transcript"])

        pbar = tqdm(total=total_games_estimate, desc="Processing games")
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break  # End of file reached

            white_elo = game.headers.get("WhiteElo", "NA")
            black_elo = game.headers.get("BlackElo", "NA")
            result = game.headers.get("Result", "NA")

            # Extracting the game moves and transforming the format
            game_moves = game.board().variation_san(game.mainline_moves())
            if len(game_moves) < 150:
                pbar.update(1)
                continue
            transformed_game_moves = re.sub(r"(\d+)\.\s+", r"\1.", game_moves)

            writer.writerow([white_elo, black_elo, result, transformed_game_moves])
            pbar.update(1)
        pbar.close()


# Path to your PGN file and output CSV file
pgn_file_path = "lichess_db_standard_rated_2017-08.pgn"
# pgn_file_path = "smaller_pgn_file.pgn"
csv_file_path = pgn_file_path.replace(".pgn", ".csv")

# Estimated total number of games based on file size
total_games_estimate = int(os.path.getsize(pgn_file_path) / (6.4 * 1024**3) * 6100000)


write_pgn_to_csv(pgn_file_path, csv_file_path, total_games_estimate)
