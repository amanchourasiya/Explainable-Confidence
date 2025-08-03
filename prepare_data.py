# prepare_data.py

import os
import pandas as pd
import json
from datasets import load_dataset, Dataset

#
import config

def load_and_combine_benchmarks():
    """Loads standard benchmarks and combines them into a single DataFrame."""
    print("--- Loading Standard Benchmarks ---")
    
    #
    # These datasets are available on the Hugging Face Hub.
    try:
        humaneval = load_dataset("openai_humaneval", split="test")
        mbpp = load_dataset("google/mbpp", split="test")
        truthcodebench = load_dataset("google-research-datasets/truth-code-bench", split="validation") #
    except Exception as e:
        print(f"Could not load a benchmark from Hugging Face Hub: {e}")
        print("Using empty placeholders for benchmark data.")
        return pd.DataFrame()

    #
    benchmark_prompts = []
    benchmark_prompts.extend([item['prompt'] for item in humaneval])
    benchmark_prompts.extend([item['text'] for item in mbpp])
    benchmark_prompts.extend([item['prompt'] for item in truthcodebench])
    
    #
    # For fine-tuning, you often only need the prompts. We will use a placeholder for the output.
    df = pd.DataFrame({"prompt": benchmark_prompts})
    df["output"] = "" #
    
    print(f"Loaded {len(df)} prompts from standard benchmarks.")
    return df

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
                "<code>def sort_list(numbers):\n    return sorted(numbers)</code><explanation>I am 95% confident. This solution uses the built-in `sorted()` function.</explanation>",
                "<code>def factorial(n):\n    if n == 0: return 1\n    else: return n * factorial(n-1)</code><explanation>I am 90% confident. This uses a recursive approach.</explanation>",
            ],
        }
        df = pd.DataFrame(mock_data)
        df.to_csv(path, index=False)

def main():
    """Converts all datasets to the JSONL format required for Gemini fine-tuning."""
    print("--- Preparing All Datasets for Gemini Fine-Tuning ---")
    
    #
    create_mock_custom_dataset(config.CUSTOM_DATASET_PATH)
    custom_df = pd.read_csv(config.CUSTOM_DATASET_PATH)
    print(f"Loaded {len(custom_df)} examples from custom dataset.")

    #
    benchmark_df = load_and_combine_benchmarks()
    
    #
    combined_df = pd.concat([custom_df, benchmark_df], ignore_index=True)
    print(f"Total examples for fine-tuning: {len(combined_df)}")

    #
    with open(config.TRAINING_DATA_JSONL, "w") as f:
        for index, row in combined_df.iterrows():
            json_record = {
                "text_input": f"<prompt>{row['prompt']}</prompt>",
                "output": row["output"]
            }
            f.write(json.dumps(json_record) + "\n")
            
    print(f"All data successfully converted to JSONL format at '{config.TRAINING_DATA_JSONL}'")

if __name__ == "__main__":
    main()
