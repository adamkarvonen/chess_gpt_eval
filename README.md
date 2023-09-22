**Overview**

There has recently been some buzz about the ability of GPT-3.5-turbo-instruct's chess playing ability. I wanted to take a more rigourous look and created this repo. Using this repo, you can play two LLM's versus each other, and LLM vs stockfish, or stockfish vs stockfish. The primary goal is to test and record the performance of these players against one another in various configurations. Illegal moves, resignations, and game states are all tracked and recorded for later analysis. Per move, a model gets 5 illegal moves before resignation of the round.

**Setup**

- Install the necessary libraries using pip.
- Copy paste your OpenAI API key in `gpt_inputs/api_key.txt`.
- If you plan on using StockfishPlayer, ensure Stockfish is installed on your system and accessible from your system path. On Mac, this is done with `brew install stockfish`.

**Game Recording**

- record_results(): This function logs the game's outcome, various statistics, and game states into a CSV file.
- Additionally, the entire game state is written to a game.txt file, although it gets overwritten every round.
- There is always a transcript of the most recent GPT API call and response in `gpt_outputs/transcript.txt`.

**How to Use**

- Set the desired players by instantiating them at the bottom of `main.py`.
- As an example, to pit `gpt-3.5-turbo-instruct` against Stockfish level 5 in a match of 15 rounds, do the following before running the file:

```
num_games = 15
player_one = GPTPlayer(model="gpt-3.5-turbo-instruct")
player_two = StockfishPlayer(skill_level=5, play_time=0.1)
play_game(player_one, player_two, num_games)
```

