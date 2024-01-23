# output_file_path = '1gb_chess.jsonl'
# with open(output_file_path, 'r') as f:
#     first_line = f.readline()
#     print(first_line.strip())

from datasets import load_dataset
import json
import os

# Initialize streaming
dataset_name = 'laion/strategic_game_chess'  # Replace with the name of your dataset
dataset = load_dataset(dataset_name, split='train', streaming=True)

output_file_path = '6gb_chess.jsonl'
size = 0

with open(output_file_path, 'w') as output_file:
    for sample in dataset:
        # print(sample)
        # Convert the sample to your desired format
        json_line = json.dumps({"text": sample["Moves"]})  # Assuming the key in the dataset is "text_field"
        
        # Write the json line to file
        output_file.write(json_line + '\n')
        
        # Monitor the file size
        size += len(json_line.encode('utf-8'))
        if size >= 6e9:  # Approximately 1GB
            break
