# prepare_data.py

import os
import pandas as pd
import json

#
import config

def create_mock_custom_dataset(path):
    """Creates a mock CSV file if one doesn't exist for demonstration."""
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists(path):
        print(f"Creating mock dataset at {path}")
        mock_data = {
            "prompt": [
                "Write a Python function to sort a list of numbers in ascending order.",
                "Create a function to calculate the factorial of a number.",
            ],
            "output": [
                "<code>def sort_list(numbers):\n    return sorted(numbers)</code><explanation>I am 95% confident. This solution uses the built-in `sorted()` function, which is efficient.</explanation>",
                "<code>def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)</code><explanation>I am 90% confident. This uses a recursive approach.</explanation>",
            ],
        }
        df = pd.DataFrame(mock_data)
        df.to_csv(path, index=False)

def main():
    """Converts the CSV dataset to the JSONL format required for Gemini fine-tuning."""
    print("--- Preparing Data for Gemini Fine-Tuning ---")
    create_mock_custom_dataset(config.CUSTOM_DATASET_PATH)
    
    df = pd.read_csv(config.CUSTOM_DATASET_PATH)

    #
    with open(config.TRAINING_DATA_JSONL, "w") as f:
        for index, row in df.iterrows():
            #
            # The prompt is the input, and the expected full response is the output.
            json_record = {
                "text_input": f"<prompt>{row['prompt']}</prompt>",
                "output": row["output"]
            }
            f.write(json.dumps(json_record) + "\n")
            
    print(f"Data successfully converted to JSONL format at '{config.TRAINING_DATA_JSONL}'")

if __name__ == "__main__":
    main()
