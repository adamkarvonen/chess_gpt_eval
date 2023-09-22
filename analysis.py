import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("logs/games.csv")

# Convert the player_one_score column to numeric type (if it's not already)
df["player_one_score"] = pd.to_numeric(df["player_one_score"], errors="coerce")

# Compute average score of player_one grouped by game_title
average_scores = df.groupby("game_title")["player_one_score"].mean()

# Display the result
print(average_scores)

average_scores.sort_values(ascending=False).plot(kind="bar", figsize=(10, 5))
plt.title("Average Player One Score by Game Title")
plt.ylabel("Average Score")
plt.xlabel("Game Title")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
