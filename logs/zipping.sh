#!/bin/bash

# Define an array of file names
files=("stockfish_dataset_blocks.csv" "lichess_6gb.csv" "lichess_6gb_results.csv" "gt1_8kElo_all.csv")

# Loop through the file array
for file in "${files[@]}"
do
    # Extract the base name without the extension
    base_name=$(basename "$file" .csv)

    echo "Zipping $file..."
    zip "${base_name}.zip" "$file"
    echo "$file zipped successfully."
done

echo "All files zipped."
