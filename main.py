import openai
import chess
import chess.engine
import json
import os
import csv

import gpt_query


# Define base Player class
class Player:
    def get_move(self, board: chess.Board, game_state: str, temperature: float) -> str:
        raise NotImplementedError

    def get_config(self) -> dict:
        raise NotImplementedError


class GPTPlayer(Player):
    def __init__(self, model: str):
        self.model = model

    def get_move(self, board: chess.Board, game_state: str, temperature: float) -> str:
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

    def get_move(self, board: chess.Board, game_state: str, temperature: float) -> str:
        self._engine.configure({"Skill Level": self._skill_level})
        result = self._engine.play(board, chess.engine.Limit(time=self._play_time))
        return board.san(result.move)

    def get_config(self) -> dict:
        return {"skill_level": self._skill_level, "play_time": self._play_time}

    def close(self):
        self._engine.quit()


def get_gpt_response(game_state: str, model: str, temperature: float) -> str:
    response = gpt_query.get_gpt_response(game_state, model, temperature)
    return response


def get_move_from_gpt_response(response: str) -> str:
    if response is None:
        return None

    # Parse the response to get only the first move
    moves = response.split()
    first_move = moves[0] if moves else None

    return first_move


def record_results(info_dict: dict, game_state: str):
    # find an unused filename
    idx = 1
    while os.path.exists(f"logs/game{idx}.json"):
        idx += 1

    # write the results to a file
    with open(f"logs/game{idx}.json", "w") as f:
        json.dump(info_dict, f, indent=4)

    with open("game.txt", "w") as f:
        f.write(game_state)


def generate_game_id(info_dict: dict) -> str:
    # Generate the game title
    player_one = (
        info_dict["model_one"] or f"stockfish {info_dict['stockfish_level_one']}"
    )
    player_two = (
        info_dict["model_two"] or f"stockfish {info_dict['stockfish_level_two']}"
    )

    return f"{player_one} / {player_two}"


def get_legal_move(
    player: Player, board: chess.Board, game_state: str, max_attempts: int = 5
) -> (str, chess.Move, int):
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

            if player_one_move_san is None:
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

            if player_two_move_san is None:
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

        player_one_config = player_one.get_config()
        player_two_config = player_two.get_config()

        game_data = {
            "transcript": game_state,
            "result": board.result(),
            "model_one": player_one_config.get("model"),
            "model_two": player_two_config.get("model"),
            "stockfish_level_one": player_one_config.get("skill_level"),
            "stockfish_level_two": player_two_config.get("skill_level"),
            "stockfish_time_one": player_one_config.get("play_time"),
            "stockfish_time_two": player_two_config.get("play_time"),
        }
        record_results(game_data, game_state)
    if isinstance(player_one, StockfishPlayer):
        player_one.close()
    if isinstance(player_two, StockfishPlayer):
        player_two.close()

        # print(game_state)


if __name__ == "__main__":
    with open("gpt_inputs/api_key.txt", "r") as f:
        openai.api_key = f.read().strip()

    # player_one = GPTPlayer(model="gpt-3.5-turbo-instruct")
    player_one = GPTPlayer(model="gpt-4")
    # player_two = StockfishPlayer(skill_level=5, play_time=0.1)
    player_two = GPTPlayer(model="gpt-3.5-turbo-instruct")
    play_game(player_one, player_two)
