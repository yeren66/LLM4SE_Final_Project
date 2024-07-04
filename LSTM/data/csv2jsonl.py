import csv
import json

def csv_to_jsonl(csv_filename, jsonl_filename):
    # Open the CSV file
    with open(csv_filename, mode='r', encoding='utf-8') as csv_file:
        # Create a CSV reader object
        csv_reader = csv.DictReader(csv_file)
        
        # Open the JSONL file for writing
        with open(jsonl_filename, mode='w', encoding='utf-8') as jsonl_file:
            # Iterate over each row in the CSV file
            for row in csv_reader:
                # Determine the label based on the sentiment
                label = 1 if row['sentiment'].lower() == 'positive' else 0

                # Create a dictionary for the current row with the added label
                data = {
                    'review': row['review'],
                    'sentiment': row['sentiment'],
                    'label': label
                }
                # Convert the dictionary to a JSON string
                json_string = json.dumps(data)
                # Write the JSON string to the JSONL file, followed by a newline
                jsonl_file.write(json_string + '\n')

# Specify the CSV and JSONL filenames
csv_filename = 'IMDB_Dataset.csv'  # Update this to your CSV file path
jsonl_filename = 'IMDB.jsonl'  # Update this to your desired JSONL output file path

# Call the function to convert CSV to JSONL
csv_to_jsonl(csv_filename, jsonl_filename)
