# Conditional Text Generation for Question Answering

This project focuses on building and evaluating a conditional text generation model using a fine-tuned T5 transformer. [cite_start]The model is trained on the Stanford Question Answering Dataset (SQuAD) to generate relevant questions from a given context passage[cite: 2, 4, 13].

## Overview

[cite_start]The core of this project is the fine-tuning of a pre-trained `t5-small` model for a sequence-to-sequence task[cite: 10]. Given a context, the model is trained to generate a corresponding question. The project includes scripts for training the model, generating text using various decoding strategies, and evaluating the quality of the generated questions using standard NLP metrics.

### Key Features

* [cite_start]**Model:** Fine-tunes the `t5-small` model, an encoder-decoder transformer with 60M parameters[cite: 10, 11].
* [cite_start]**Dataset:** Utilizes the Stanford Question Answering Dataset (SQuAD v1.1)[cite: 4].
* [cite_start]**Decoding Strategies:** Implements and compares three different generation techniques[cite: 63]:
    * [cite_start]Beam Search (Greedy) [cite: 25]
    * [cite_start]Top-k Sampling [cite: 25]
    * [cite_start]Top-p (Nucleus) Sampling [cite: 26]
* [cite_start]**Evaluation:** Automatically evaluates generated text against reference questions using BLEU, ROUGE, and METEOR scores[cite: 32].
* [cite_start]**Optimization:** The training and inference pipelines are optimized for Apple Silicon (MPS) devices[cite: 13].

## Project Structure

```
.
├── data/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── models/
│   └── t5-squad-final/
│       └── ... (fine-tuned model files)
├── samples/
│   ├── greedy_questions.txt
│   ├── top_k_questions.txt
│   └── top_p_questions.txt
├── train.py
├── generation.py
└── README.md
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Install dependencies:**
    This project requires Python and several libraries. Install them using pip:
    ```bash
    pip install torch transformers pandas rouge_score nltk numpy
    ```

3.  **Download NLTK data:**
    The evaluation script requires NLTK data for tokenization and METEOR scores. Run this in a Python interpreter:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('wordnet')
    ```

4.  **Prepare the Dataset:**
    * Download the SQuAD v1.1 dataset.
    * Preprocess the data into three CSV files: `train.csv`, `val.csv`, and `test.csv`.
    * [cite_start]Each CSV should contain at least an `input_text` column (formatted as `"answer_question: [context]"`) and a `target_text` column (the corresponding question)[cite: 8].
    * Place these files inside a `data/` directory in the root of the project.

## Usage

### 1. Training the Model

Run the training script to fine-tune the `t5-small` model. [cite_start]The script will handle loading the data, training for 3 epochs, and saving the best model based on validation loss[cite: 17, 20].

```bash
python train.py
```

The final model and tokenizer will be saved to `models/t5-squad-final/`.

### 2. Generating Questions

After training, run the generation script. [cite_start]This will load your fine-tuned model, generate 100 questions for each decoding strategy using contexts from your test data, and save the results[cite: 27].

```bash
python generation.py
```

[cite_start]The script will create three output files in the `samples/` directory[cite: 31]:
* `samples/greedy_questions.txt`
* `samples/top_k_questions.txt`
* `samples/top_p_questions.txt`

The script will then automatically evaluate the outputs and print the performance metrics to the console.

## Evaluation Results

The model's performance was evaluated using three decoding strategies. [cite_start]Top-K sampling was identified as the best-performing method based on an average of all metrics[cite: 52, 60].

| Method          | BLEU    | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR  |
| --------------- | ------- | ------- | ------- | ------- | ------- |
| **Greedy** | [cite_start]0.0258 [cite: 34] | [cite_start]0.1599 [cite: 35] | [cite_start]0.0256 [cite: 36] | [cite_start]0.1573 [cite: 37] | [cite_start]0.1344 [cite: 38] |
| **Top-K (Best)**| [cite_start]**0.0244** [cite: 40] | [cite_start]**0.1739** [cite: 41] | [cite_start]**0.0283** [cite: 42] | [cite_start]**0.1668** [cite: 43] | [cite_start]**0.1495** [cite: 44] |
| **Top-P** | [cite_start]0.0226 [cite: 46] | [cite_start]0.1365 [cite: 47] | [cite_start]0.0174 [cite: 48] | [cite_start]0.1320 [cite: 49] | [cite_start]0.1237 [cite: 50] |

### Challenges and Observations

* [cite_start]**Strengths:** The model successfully generated contextually relevant questions and the epoch-based evaluation strategy ensured stable training convergence[cite: 63]. [cite_start]Top-K sampling provided the best balance of quality and diversity[cite: 63].
* [cite_start]**Challenges:** The overall low metric scores suggest there is room for improvement in question quality[cite: 65]. [cite_start]Memory optimization was also a key consideration for running on an MPS device[cite: 65].
