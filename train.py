# train.py

import google.generativeai as genai
import time

#
import config

def main():
    """Uploads the dataset and starts a fine-tuning job for the Gemini model."""
    print("--- Starting Gemini Fine-Tuning Job ---")
    
    #
    genai.configure(api_key=config.GOOGLE_API_KEY)

    #
    print(f"Uploading file '{config.TRAINING_DATA_JSONL}' to Google AI...")
    training_file = genai.upload_file(path=config.TRAINING_DATA_JSONL)

    #
    print(f"Starting fine-tuning job for model: {config.MODEL_NAME}")
    job = genai.create_tuned_model(
        source_model=f"models/{config.MODEL_NAME}",
        training_data=training_file,
        id=config.TUNED_MODEL_DISPLAY_NAME,
        epoch_count=config.TRAINING_EPOCHS,
        batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
    )

    print(f"Fine-tuning job started. Job name: {job.name}")
    print("You can monitor the job's progress in the Google AI Studio.")

    #
    while job.state.name != "SUCCEEDED" and job.state.name != "FAILED":
        print(f"Job state: {job.state.name}... waiting...")
        time.sleep(60) #
        job = genai.get_tuned_model(job.name) #

    if job.state.name == "SUCCEEDED":
        print("Fine-tuning job completed successfully!")
        print(f"Your new model is ready: {job.name}")
    else:
        print("Fine-tuning job failed.")
        print(job.state)

if __name__ == "__main__":
    main()
