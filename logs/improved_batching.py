import pandas as pd
from tqdm import tqdm
import random
from pandarallel import pandarallel
import chess
from stockfish import Stockfish
from collections import deque

pandarallel.initialize(progress_bar=True)


def dedup_dataset(input_file: str):
    # Step 1: Deduplicate games
    df = pd.read_csv(input_file)

    unique_transcripts = set()
    rows_to_delete = []

    print(f"before: {len(df)}")

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        transcript = row["transcript"]
        if transcript not in unique_transcripts:
            unique_transcripts.add(transcript)
        else:
            rows_to_delete.append(index)

    df.drop(rows_to_delete, inplace=True)

    print(f"after: {len(df)}")

    df.to_csv(input_file, index=False)


# Step 2: Add player skill to beginning of every transcript
# At the end of this, we will save only the 'transcript' column to 'input_dataset.csv', as other info is no longer needed
def map_skill_to_int(skill: int) -> str:
    if skill == 20:
        return "9"
    if random.random() > 0.5:
        return "?"
    if skill == -2:
        return "0"
    # Define the original and target ranges
    original_min, original_max = -1, 19
    target_min, target_max = 1, 8

    # Calculate the total number of values in each range
    original_range = original_max - original_min
    target_range = target_max - target_min

    # Scale the original value to the target range
    scaled_value = ((skill - original_min) / original_range) * target_range + target_min

    # Round and return the scaled value, making sure it stays within the target range
    return str(min(target_max, max(target_min, round(scaled_value))))


def add_skill_to_transcript(row):
    # Split and get the number from player_one and player_two
    player_one_number = int(row["player_one"].split()[1])
    player_two_number = int(row["player_two"].split()[1])

    # Apply map_value to these numbers
    mapped_player_one = map_skill_to_int(player_one_number)
    mapped_player_two = map_skill_to_int(player_two_number)

    transcript = row["transcript"].split("\n\n")[1]

    # Prepend the transcript with the formatted string
    row["transcript"] = f"[{mapped_player_one},{mapped_player_two}]{transcript}"

    return row


# Step 3: Randomly insert centipawn
def map_eval_to_int(evaluation: dict) -> int:
    if evaluation["type"] == "mate":
        # for example, 3 would be mate in 3 for white, -2 is mate in 2 for black
        if evaluation["value"] > 0:
            return 9
        else:
            return -9

    # if not mate, must be centipawn advantage
    value = evaluation["value"]

    if value > 700:
        return 8
    elif value < -700:
        return -8
    original_min, original_max = -700, 700
    target_min, target_max = -7, 7

    # Calculate the total number of values in each range
    original_range = original_max - original_min
    target_range = target_max - target_min

    # Scale the original value to the target range
    scaled_value = ((value - original_min) / original_range) * target_range + target_min

    # Round and return the scaled value, making sure it stays within the target range
    return str(min(target_max, max(target_min, round(scaled_value))))


def game_over_to_value(board_result: str) -> int:
    result_map = {"1-0": 9, "0-1": -9, "1/2-1/2": 0}
    return result_map[board_result]


def insert_centipawn(moves_string: str, depth: int = 9, frequency: float = 0.03) -> str:
    # Create a new board
    board = chess.Board()

    mac_path = "stockfish"
    linux_path = "/usr/games/stockfish"
    # self._engine = chess.engine.SimpleEngine.popen_uci(linux_path)
    stockfish = Stockfish(mac_path)
    stockfish.set_depth(depth)
    eval_results = []

    new_moves_string = ""

    # Apply each move to the board
    for move in moves_string.split():
        # Skip move numbers
        if "." in move:
            board.push_san(move.split(".")[1])
        else:
            board.push_san(move)

        new_moves_string += move + " "
        if random.random() < frequency:
            # Check for checkmate or draw
            eval = ""
            if board.result() != "*":
                # eval_results.append(game_over_to_value(board.result()))
                eval = " <" + str(game_over_to_value(board.result())) + "> "
            else:
                stockfish.set_fen_position(board.fen())
                evaluation = stockfish.get_evaluation()

                # eval_results.append(map_eval_to_int(evaluation))
                eval = "<" + str(map_eval_to_int(evaluation)) + " "

            new_moves_string += eval
    return new_moves_string


def chunk_long_games(text: str) -> list[str] | str:
    full_chunk_size = 1023

    length = len(text)

    if length < full_chunk_size:
        return text

    header_size = 5
    chunk_size = full_chunk_size - header_size
    header = text[:header_size]

    text = text[header_size:]
    length = len(text)

    # Calculate the number of chunks needed
    num_chunks = (length + chunk_size - 1) // chunk_size

    chunks = []

    for i in range(num_chunks):
        start_index = length - (i + 1) * chunk_size
        end_index = length - i * chunk_size
        start_index = max(start_index, 0)  # Ensure the start index is not negative

        chunk = header + text[start_index:end_index]
        chunks.append(chunk)

    # Reverse the list to maintain the chronological order
    chunks.reverse()

    if len(chunks[0]) < 511:
        chunks.pop(0)

    return chunks


def create_batched_dataset(input_file: str, output_filename: str):
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Prepare the new dataset for blocks
    blocks = []
    remaining_games = deque(
        df["transcript"].tolist()
    )  # Use deque for efficient pops from the left

    original_length = len(remaining_games)  # Store the original length

    # Block size limit
    block_size = 1024

    # Initialize the progress bar
    with tqdm(total=original_length, desc="Processing") as pbar:
        while remaining_games:
            block = ";"
            # Select the next game
            next_game = remaining_games.pop()
            block += next_game
            while len(block) < block_size and remaining_games:
                # if len(df) > 21:
                #     random_idx = random.randint(0, 20)
                #     next_game = remaining_games[random_idx]
                #     remaining_games[random_idx] = ""
                # else:
                next_game = remaining_games.popleft()
                block += ";" + next_game
                if len(block) > block_size:
                    # If the game makes the block too long, re-add it to the dataset
                    if len(remaining_games) > 100:
                        remaining_games.insert(99, next_game)
                    else:
                        break
                    break

            if len(block) >= block_size:
                # Add the block to the blocks list
                blocks.append(block[:block_size])

            # Update the progress bar
            pbar.update(original_length - len(remaining_games) - pbar.n)

    del df
    # Create a new DataFrame for the blocks
    blocks_df = pd.DataFrame(blocks, columns=["transcript"])

    # Save the blocks to a new CSV file
    blocks_df.to_csv(output_filename, index=False)


def process_dataset(input_file: str):
    output_filename = input_file.replace(".csv", "_blocks.csv")
    processed_filename = input_file.replace(".csv", "_processed.csv")
    print(f"Deduplicating {input_file}")
    dedup_dataset(input_file)
    df = pd.read_csv(input_file)
    print(f"Adding skill to {input_file}")
    df = df.parallel_apply(add_skill_to_transcript, axis=1)
    df["transcript"].to_csv("input_dataset.csv", index=False)
    df = pd.read_csv("input_dataset.csv")
    print(f"Inserting centipawns in {input_file}")
    df["transcript"] = df["transcript"].parallel_apply(
        lambda x: insert_centipawn(x, depth=10, frequency=0.03)
    )
    print(f"Chunking long games in {input_file}")
    df["transcript"] = df["transcript"].parallel_apply(chunk_long_games)
    df = df.explode("transcript")

    df["length"] = df["transcript"].parallel_apply(len)
    print(f"Sorting file by game length")
    df.sort_values(by="length", inplace=True)
    df.to_csv(processed_filename, index=False)
    print(f"Creating batched {output_filename} from {processed_filename}")
    create_batched_dataset(processed_filename, output_filename)
    print(f"{input_file} complete")


input_files = [
    "pool3_dataset.csv",
    "linux_dataset.csv",
    "pool1_dataset.csv",
    "pool2_dataset.csv",
    "mac_dataset.csv",
]

for file in input_files:
    process_dataset(file)

# test_file = "test_input_dataset.csv"
# process_dataset(test_file)
