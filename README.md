# Explainable Confidence Framework

This repository contains the official implementation for the research paper, **"Explainable Confidence: LLMs That Justify Their Code."** This project introduces a framework to fine-tune Gemini 1.5, enabling it to generate not only code but also a reliable confidence score and a human-understandable justification for its reasoning.

The goal is to bridge the trust gap in AI-assisted software engineering by making the code generation process transparent and accountable.

## Features

  * **Modular Codebase:** The project is organized into clear, separate scripts for configuration, data preparation, training, and generation.
  * **Gemini 1.5 Integration:** Utilizes the Google Generative AI API to fine-tune and run inference on the Gemini 1.5 model.
  * **Data Preparation:** Includes scripts to process and combine standard benchmarks (**HumanEval**, **MBPP**, **TruthCodeBench**) with your custom dataset.
  * **API-Based Fine-Tuning:** The `train.py` script manages the process of uploading data and launching a fine-tuning job via the Google AI API.
  * **Structured Output:** The fine-tuned model is trained to produce structured output containing the generated code and a detailed explanation, which includes a verbalized confidence score.

-----

## Setup and Installation

### 1\. Clone the Repository

```bash
git clone https://github.com/amanchourasiya/explainable-confidence.git
cd explainable-confidence
```

### 2\. Install Dependencies

It is recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

*(You will need to create a `requirements.txt` file containing `google-generativeai`, `pandas`, `datasets`, and `scikit-learn`.)*

### 3\. Configure Your API Key

Open the `config.py` file and replace the placeholder with your Google AI Studio API key.

```python
# config.py
GOOGLE_API_KEY = "YOUR_API_KEY"
```

### 4\. Prepare Your Custom Dataset

Ensure your custom dataset of 5,000 code-explanation pairs is available at `data/custom_dataset.csv`. The CSV should have two columns: `prompt` and `output`.

-----

## How to Run

The project is designed to be run in three main steps.

### Step 1: Prepare the Data

This script will load all your datasets, combine them, and convert them into the `training_data.jsonl` file required by the Google AI API.

```bash
python prepare_data.py
```

### Step 2: Fine-Tune the Model

This script will upload the prepared data and start the fine-tuning job. You can monitor the progress in your Google AI Studio.

```bash
python train.py
```

After the job succeeds, your fine-tuned model will be available through the API, identified by the display name you set in `config.py`.

### Step 3: Generate Code and Explanations

Use this script to run inference with your newly fine-tuned model.

```bash
python generate.py
```

The script will output the generated code and the corresponding explanation for the test prompt defined within the file. You can easily modify `generate.py` to test different prompts.
