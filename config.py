# config.py

#
# IMPORTANT: Replace "YOUR_API_KEY" with your actual Google AI Studio API key.
# Keep this key secure and do not commit it to public repositories.
GOOGLE_API_KEY = "YOUR_API_KEY"

#
# The specific Gemini 1.5 model to use for fine-tuning and generation.
#
MODEL_NAME = "gemini-1.5-flash-latest"

#
# The name you want to give your fine-tuned model on the Google AI platform.
TUNED_MODEL_DISPLAY_NAME = "explainable-confidence-model-v1"

#
# Local data paths
CUSTOM_DATASET_PATH = "data/custom_dataset.csv"
TRAINING_DATA_JSONL = "data/training_data.jsonl" #

#
# Fine-tuning hyperparameters
TRAINING_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 0.001
