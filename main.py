import openai
import chess
import chess.engine
import os
import csv
import random
import time

import gpt_query

from typing import Optional, Tuple


# Define base Player class
class Player:
    def get_move(self, board: chess.Board, game_state: str, temperature: float) -> str:
        raise NotImplementedError

    def get_config(self) -> dict:
        raise NotImplementedError


class GPTPlayer(Player):
    def __init__(self, model: str):
        self.model = model

    def get_move(
        self, board: chess.Board, game_state: str, temperature: float
    ) -> Optional[str]:
        response = get_gpt_response(game_state, self.model, temperature)
        return get_move_from_gpt_response(response)

    def get_config(self) -> dict:
        return {"model": self.model}


class StockfishPlayer(Player):
    def __init__(self, skill_level: int, play_time: float):
        self._skill_level = skill_level
        self._play_time = play_time
        # If getting started, you need to run brew install stockfish
        self._engine = chess.engine.SimpleEngine.popen_uci("stockfish")

    def get_move(
        self, board: chess.Board, game_state: str, temperature: float
    ) -> Optional[str]:
        self._engine.configure({"Skill Level": self._skill_level})
        result = self._engine.play(board, chess.engine.Limit(time=self._play_time))
        if result.move is None:
            return None
        return board.san(result.move)

    def get_config(self) -> dict:
        return {"skill_level": self._skill_level, "play_time": self._play_time}

    def close(self):
        self._engine.quit()


def get_gpt_response(game_state: str, model: str, temperature: float) -> Optional[str]:
    response = gpt_query.get_gpt_response(game_state, model, temperature)
    return response


def get_move_from_gpt_response(response: Optional[str]) -> Optional[str]:
    if response is None:
        return None

    # Parse the response to get only the first move
    moves = response.split()
    first_move = moves[0] if moves else None

    return first_move


def record_results(
    board: chess.Board,
    player_one: Player,
    player_two: Player,
    game_state: str,
    player_one_illegal_moves: int,
    player_two_illegal_moves: int,
    total_time: float,
):
    unique_game_id = generate_unique_game_id()

    player_one_title, player_two_title = get_player_titles(player_one, player_two)

    info_dict = {
        "game_id": unique_game_id,  # Storing the unique game ID
        "transcript": game_state,
        "result": board.result(),
        "player_one": player_one_title,
        "player_two": player_two_title,
        "player_one_score": board.result().split("-")[0],
        "player_two_score": board.result().split("-")[1],
        "player_one_illegal_moves": player_one_illegal_moves,
        "player_two_illegal_moves": player_two_illegal_moves,
        "game_title": f"{player_one_title} vs. {player_two_title}",
        "number_of_moves": board.fullmove_number,
        "time_taken": total_time,
    }

    csv_file_path = "logs/games.csv"

    # Determine if we need to write headers (in case the file doesn't exist yet)
    write_headers = not os.path.exists(csv_file_path)

    # Append the results to the CSV file
    with open(csv_file_path, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=info_dict.keys())
        if write_headers:
            writer.writeheader()
        writer.writerow(info_dict)

    with open("game.txt", "w") as f:
        f.write(game_state)


def generate_unique_game_id() -> str:
    timestamp = int(time.time())
    random_num = random.randint(1000, 9999)  # 4-digit random number
    return f"{timestamp}-{random_num}"


def get_player_titles(player_one: Player, player_two: Player) -> Tuple[str, str]:
    player_one_config = player_one.get_config()
    player_two_config = player_two.get_config()

    # For player one
    if "model" in player_one_config:
        player_one_title = player_one_config["model"]
    else:
        player_one_title = f"Stockfish {player_one_config['skill_level']}"

    # For player two
    if "model" in player_two_config:
        player_two_title = player_two_config["model"]
    else:
        player_two_title = f"Stockfish {player_two_config['skill_level']}"

    return (player_one_title, player_two_title)


def get_legal_move(
    player: Player, board: chess.Board, game_state: str, max_attempts: int = 5
) -> Tuple[Optional[str], Optional[chess.Move], int]:
    """Request a move from the player and ensure it's legal."""
    move_san = None
    move_uci = None

    for attempt in range(max_attempts):
        move_san = player.get_move(
            board, game_state, ((attempt / max_attempts) * 0.5) + 0.3
        )
        try:
            move_uci = board.parse_san(move_san)
        except Exception as e:
            print(f"Error parsing move {move_san}: {e}")
            continue

        if move_uci in board.legal_moves:
            if not move_san.startswith(" "):
                move_san = " " + move_san
            return move_san, move_uci, attempt
        print(f"Illegal move: {move_san}")

    # If we reach here, the player has made illegal moves for all attempts.
    print(f"{player} provided illegal moves for {max_attempts} attempts.")
    return (
        None,
        None,
        max_attempts,
    )  # Optionally, handle the situation differently, e.g., end the game, etc.


def play_game(player_one: Player, player_two: Player):
    for _ in range(5):  # Play 10 games
        with open("gpt_inputs/prompt.txt", "r") as f:
            game_state = f.read()
        board = chess.Board()
        player_one_illegal_moves = 0
        player_two_illegal_moves = 0
        start_time = time.time()
        while not board.is_game_over():
            with open("game.txt", "w") as f:
                f.write(game_state)
            current_move_num = str(board.fullmove_number) + "."
            game_state += "\n" + current_move_num
            print(f"{current_move_num}", end=" ")

            (
                player_one_move_san,
                player_one_move_uci,
                player_one_attempts,
            ) = get_legal_move(player_one, board, game_state)

            player_one_illegal_moves += player_one_attempts

            if player_one_move_san is None or player_one_move_uci is None:
                print("Game over: 5 consecutive Illegal moves from player one")
                print(board)
                break

            board.push(player_one_move_uci)  # Apply UCI move to the board

            game_state += player_one_move_san
            print(f"{player_one_move_san}", end=" ")

            if board.is_game_over():
                print(f"Game over with result: {board.result()}")
                print(board)
                break

            (
                player_two_move_san,
                player_two_move_uci,
                player_two_attempts,
            ) = get_legal_move(player_two, board, game_state)

            player_two_illegal_moves += player_two_attempts

            if player_two_move_san is None or player_two_move_uci is None:
                print("Game over: 5 consecutive Illegal moves from player one")
                print(board)
                break

            board.push(player_two_move_uci)
            game_state += player_two_move_san
            print(f"{player_two_move_san}")

            if board.is_game_over():
                print(f"Game over with result: {board.result()}")
                print(board)
                break

        end_time = time.time()
        total_time = end_time - start_time
        record_results(
            board,
            player_one,
            player_two,
            game_state,
            player_one_illegal_moves,
            player_two_illegal_moves,
            total_time,
        )
    if isinstance(player_one, StockfishPlayer):
        player_one.close()
    if isinstance(player_two, StockfishPlayer):
        player_two.close()

        # print(game_state)


if __name__ == "__main__":
    with open("gpt_inputs/api_key.txt", "r") as f:
        openai.api_key = f.read().strip()

    player_one = GPTPlayer(model="gpt-3.5-turbo-instruct")
    # player_one = GPTPlayer(model="gpt-4")
    player_two = StockfishPlayer(skill_level=5, play_time=0.1)
    # player_two = GPTPlayer(model="gpt-3.5-turbo-instruct")
    play_game(player_one, player_two)
