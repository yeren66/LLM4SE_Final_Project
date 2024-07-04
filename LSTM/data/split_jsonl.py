import json
import random

def split_jsonl(jsonl_filename, train_filename, eval_filename, test_filename, train_ratio=0.7, eval_ratio=0.1):
    # Read all data from the JSONL file
    with open(jsonl_filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Shuffle the data to ensure randomness
    random.shuffle(lines)
    
    # Calculate the number of entries for each split
    total_lines = len(lines)
    train_end = int(total_lines * train_ratio)
    eval_end = train_end + int(total_lines * eval_ratio)
    
    # Split the data
    train_data = lines[:train_end]
    eval_data = lines[train_end:eval_end]
    test_data = lines[eval_end:]
    
    # Write the split data to separate files
    with open(train_filename, 'w', encoding='utf-8') as file:
        file.writelines(train_data)
    with open(eval_filename, 'w', encoding='utf-8') as file:
        file.writelines(eval_data)
    with open(test_filename, 'w', encoding='utf-8') as file:
        file.writelines(test_data)

# File paths
jsonl_filename = 'IMDB.jsonl'  # Input JSONL file
train_filename = 'train.jsonl'   # Output training file
eval_filename = 'eval.jsonl'     # Output evaluation file
test_filename = 'test.jsonl'     # Output testing file

# Call the function to split the JSONL file
split_jsonl(jsonl_filename, train_filename, eval_filename, test_filename)
