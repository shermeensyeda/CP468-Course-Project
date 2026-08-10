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
│   ├── evaluation.py
│   ├── model.py
│   ├── train.py
│   └── vocabulary.py
├── llm_baseline/
│   ├── outputs/
│   ├── compare_metrics.py
│   ├── prompts.py
│   ├── run_baseline.py
│   └── run_lstm_baseline.py
├── checkpoints/
├── scripts/
│   └── print_eval.py
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

The number of epochs, batch size, and training-set size can also be configured using command-line arguments.

For example, a small integration test can be run using:

```bash
python -m src.train --epochs 1 --limit 20
```

The `--limit` option is intended for testing the training pipeline and should not be used to reproduce the final experimental model.

The training script automatically selects CUDA when a compatible GPU is available and otherwise uses the CPU.

During initialization, the model reports the number of trainable parameters. The integrated model contains:

```text
42,188,692 trainable parameters
```

Model checkpoints are saved to:

```text
checkpoints/best_lstm.pt
checkpoints/last_lstm.pt
```

`best_lstm.pt` stores the model with the lowest validation loss, while `last_lstm.pt` stores the model state from the final completed epoch. 

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

To reproduce our reported results specifically, use:

```bash
python llm_baseline/run_baseline.py --limit 100
```

For the reported experiment, the LLM baseline was run on a fixed subset of 100 test examples selected using a random seed of `42`. Due to API rate limits and cost considerations, this experimental subset was used instead of running the LLM baseline across the complete 2,382-example test split.

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
The LLM baseline was run on 100 selected test examples. During automatic metric calculation, examples where the source sentence and reference correction were already identical are excluded consistently across all systems. This leaves 76 non-trivial examples for the reported automatic metric comparison.

#### Checkpoints and Experiment Metadata

LSTM training saves the best validation checkpoint and training metadata so that the trained model and experimental setup can be inspected and reused.

LLM outputs are saved separately for each prompting condition, along with token usage and estimated API costs.

### Evaluation

The LSTM and LLM approaches are evaluated using two standard automatic evaluation metrics:

- **BLEU**
- **ROUGE-L F1**

The evaluation compares the LSTM model against all four Gemini prompting conditions:

1. Prompt V1 — Zero-shot
2. Prompt V2 — Zero-shot
3. Prompt V1 — Few-shot
4. Prompt V2 — Few-shot

To compute the automatic evaluation metrics, run:

```bash
python llm_baseline/compare_metrics.py
```

The resulting metrics are saved to:

```text
llm_baseline/outputs/metrics_summary.json
```

The LLM baseline was run on 100 selected test examples. For the reported automatic comparison, examples where the source sentence and reference correction were identical are excluded consistently across all systems. This leaves 76 non-trivial examples for the automatic metric comparison.

The resulting scores are:

| System | BLEU | ROUGE-L F1 |
| --- | ---: | ---: |
| LSTM | 0.1621 | 0.5312 |
| Gemini V1 Zero-shot | 0.6202 | 0.8188 |
| Gemini V2 Zero-shot | **0.7256** | **0.8826** |
| Gemini V1 Few-shot | 0.6625 | 0.8471 |
| Gemini V2 Few-shot | 0.7080 | 0.8767 |

Among the evaluated LLM conditions, Prompt V2 in the zero-shot setting achieved the highest BLEU and ROUGE-L F1 scores.

### Qualitative Error Analysis

In addition to the automatic metrics, the evaluation includes a qualitative comparison of LSTM and LLM outputs.

The analysis presents at least 10 test examples side by side, including:

- Source sentence
- Reference correction
- LSTM output
- LLM output
- Error category

The selected examples illustrate different model behaviours and failure modes rather than only successful cases. Observed LSTM failure modes include repetition, under-translation, and low-adequacy outputs.

To display the quantitative results and curated qualitative examples, run:

```bash
python scripts/print_eval.py
```

### LLM API Cost

Token usage and estimated API costs were recorded for each Gemini prompting condition.

| Condition | Input Tokens | Output Tokens | Estimated Cost (USD) |
| --- | ---: | ---: | ---: |
| V1 Zero-shot | 4,281 | 1,966 | $0.0062 |
| V2 Zero-shot | 10,681 | 1,997 | $0.0082 |
| V1 Few-shot | 20,881 | 1,998 | $0.0113 |
| V2 Few-shot | 27,281 | 2,001 | $0.0132 |

The total estimated API cost across all four experimental conditions was approximately **$0.0389 USD**.

Detailed cost information is stored in:

```text
llm_baseline/outputs/cost_summary.json
```

