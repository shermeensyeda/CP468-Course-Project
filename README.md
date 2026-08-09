# CP468-Course-Project
## Grammatical Error Correction: LSTM vs. LLM

### Project Overview

This project compares a traditional LSTM-based sequence-to-sequence model with a modern Large Language Model (LLM) on a grammatical error correction (GEC) task.

The goal is to transform grammatically incorrect learner-written sentences into corrected sentences. Both approaches are evaluated on the same held-out test examples to provide a fair comparison between the traditional neural architecture and the modern LLM baseline.

The project consists of two main approaches:

- **LSTM Model:** A sequence-to-sequence architecture using an LSTM encoder, attention mechanism, and LSTM decoder.
- **LLM Baseline:** A Gemini-based grammatical error correction system evaluated using two prompt variants under zero-shot and few-shot settings.

### Dataset

The project uses the **CLC FCE Dataset v1.1**, which contains English learner-written sentences and grammatical corrections.

The preprocessing pipeline converts the original data into source-target sentence pairs:

- **Source:** Original learner-written sentence
- **Target:** Grammatically corrected sentence

The processed dataset contains:

| Split | Number of Sentence Pairs |
|---|---:|
| Training | 25,282 |
| Validation | 1,924 |
| Test | 2,382 |

The original dataset partitions are preserved to prevent train/test leakage.

A word-level vocabulary is constructed using the training split only. The vocabulary includes the special tokens `<PAD>`, `<UNK>`, `<SOS>`, and `<EOS>`.
### Repository Structure

The repository is organized into separate components for data preprocessing, LSTM training, LLM prompting, evaluation, and saved outputs.

```text
CP468-Course-Project/
├── data/
│   ├── processed/
│   └── raw/
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── vocabulary.py
├── llm_baseline/
│   ├── outputs/
│   ├── prompts.py
│   └── run_baseline.py
├── checkpoints/
├── scripts/
├── requirements.txt
└── README.md
```

### Environment Setup

The project was tested using Python 3.13.3.

The required Python packages and tested versions are specified in `requirements.txt`:

```text
torch==2.13.0
numpy==2.5.1
google-genai==2.17.0
```

#### 1. Download or Clone the Repository

Clone the repository and navigate to the project root directory.

```bash
git clone https://github.com/shermeensyeda/CP468-Course-Project.git
cd CP468-Course-Project
```

#### 2. Install Dependencies

Install the required packages using:

```bash
python -m pip install -r requirements.txt
```

#### 3. Verify the Installation

The main dependencies can be verified with:

```bash
python -c "import torch; import numpy; from google import genai; print('All dependencies working')"
```

A successful installation should output:

```text
All dependencies working
```

### API Configuration

The LLM baseline uses the Google GenAI SDK and requires a Gemini API key.

The API key must be stored in the `GEMINI_API_KEY` environment variable before running the LLM baseline.

API keys should never be committed directly to the GitHub repository.

### Data Pipeline

The data pipeline loads the processed FCE sentence pairs, converts tokens to vocabulary IDs, applies padding, and creates masks for batched model training.

The pipeline can be tested from the project root using:

```bash
python -m scripts.test_data_pipeline
```

A successful test verifies:

- Dataset loading
- Vocabulary loading
- Source and target batching
- Out-of-vocabulary (OOV) handling
- Padding
- Source and target masks

The data pipeline was tested successfully with 25,282 training rows and a vocabulary size of 10,899.

### LSTM Model

The traditional baseline uses an encoder-decoder sequence-to-sequence architecture implemented in PyTorch.

The model consists of:

- Word embeddings
- LSTM encoder
- LSTM decoder
- Attention mechanism
- Teacher forcing during training
- Gradient clipping

The training script uses a fixed random seed of `42` to improve reproducibility.

To train the LSTM model from the project root, run:

```bash
python -m src.train
```

The training script automatically selects CUDA when a compatible GPU is available and otherwise uses the CPU.

During initialization, the model reports the number of trainable parameters. The integrated model contains:

```text
42,188,692 trainable parameters
```

The best validation checkpoint is saved to:

```text
checkpoints/best_lstm.pt
```

Training metadata is saved to:

```text
checkpoints/training_meta.json
```

The metadata includes the model parameter count, total training time, hardware used, and best validation loss.

### LLM Baseline

The LLM baseline uses Gemini for grammatical error correction.

Four experimental conditions are implemented:

1. Prompt V1 — Zero-shot
2. Prompt V2 — Zero-shot
3. Prompt V1 — Few-shot
4. Prompt V2 — Few-shot

Few-shot examples are selected from the training split using a fixed random seed of `42`.

The baseline can be tested on a small subset using:

```bash
python llm_baseline/run_baseline.py --limit 20
```

To run the baseline on the complete configured test set, use:

```bash
python llm_baseline/run_baseline.py
```

The outputs are stored in:

```text
llm_baseline/outputs/
```

The experiment also records token usage and estimated API cost in:

```text
llm_baseline/outputs/cost_summary.json
```

The IDs of the exact test examples used are stored in:

```text
llm_baseline/outputs/test_subset_ids.json
```

This allows the LSTM and LLM to be evaluated on the same test examples for a fair comparison.

### Reproducibility

Several measures are used to improve reproducibility across the project.

#### Fixed Random Seeds

The LSTM training pipeline uses a fixed random seed of `42` for Python, NumPy, and PyTorch. The LLM baseline also uses a seed of `42` when selecting test subsets and few-shot examples.

#### Pinned Dependencies

The required package versions are pinned in `requirements.txt` to reduce differences between environments.

```text
torch==2.13.0
numpy==2.5.1
google-genai==2.17.0
```

#### Consistent Data Splits

The original FCE training, validation, and test partitions are preserved. The vocabulary is constructed using only the training data to prevent information leakage from the validation or test sets.

#### Consistent Evaluation Examples

The LLM baseline records the IDs of the test examples used during the experiment in:

```text
llm_baseline/outputs/test_subset_ids.json
```

These IDs can be used to ensure that the LSTM and LLM are evaluated on identical examples.

#### Checkpoints and Experiment Metadata

LSTM training saves the best validation checkpoint and training metadata so that the trained model and experimental setup can be inspected and reused.

LLM outputs are saved separately for each prompting condition, along with token usage and estimated API costs.

### Evaluation

The LSTM and LLM outputs will be evaluated on the same test examples using:

- BLEU
- chrF
- Exact Match

The evaluation pipeline also includes qualitative error analysis to compare the types of grammatical corrections made by the two approaches.

Exact evaluation commands and output locations will be documented after the evaluation pipeline is finalized.
