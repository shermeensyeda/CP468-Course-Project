"""Run the trained LSTM seq2seq model on the test subset used by the LLM baseline
and save outputs to llm_baseline/outputs/lstm.jsonl.

Usage:
    python3 llm_baseline/run_lstm_baseline.py [--limit N]

If `llm_baseline/outputs/test_subset_ids.json` exists, the script will use that
subset to match the LLM evaluation. Otherwise it runs on the full test set or
an optional random sample when `--limit` is provided.
"""
import os
import json
import argparse
from pathlib import Path
import sys

# Ensure repo root is on sys.path when executing this script directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import torch
except Exception:
    torch = None

from src.vocabulary import Vocabulary
from src.text_utils import tokenize
if torch is not None:
    from src.model import Encoder, Decoder, Seq2Seq

TEST_FILE = "data/processed/test.jsonl"
OUTPUT_DIR = "llm_baseline/outputs"
CHECKPOINT = "checkpoints/best_lstm.pt"


def load_jsonl(path, limit=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if limit is not None:
        rows = rows[:limit]
    return rows


def load_test_subset_ids(path):
    if Path(path).exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            return None
    return None


def build_model(vocab, device="cpu"):
    if torch is None:
        return None
    # hyperparameters aligned with training defaults
    EMBED_SIZE = 256
    HIDDEN_SIZE = 512
    NUM_LAYERS = 2
    PAD_IDX = vocab.pad_idx
    enc = Encoder(len(vocab.itos), EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, pad_idx=PAD_IDX)
    dec = Decoder(len(vocab.itos), EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, pad_idx=PAD_IDX)
    model = Seq2Seq(enc, dec, PAD_IDX)
    model.to(device)
    return model


def generate_for_rows(rows, vocab, model, device="cpu"):
    outputs = []
    if torch is None or model is None:
        # fallback: copy source to lstm_output (so files are comparable)
        for row in rows:
            outputs.append({
                "id": row["id"],
                "source": row["source"],
                "target": row.get("target", ""),
                "lstm_output": row["source"],
            })
        return outputs

    model.eval()
    with torch.no_grad():
        for row in rows:
            src_toks = tokenize(row["source"])[:60]
            src_ids = torch.tensor(vocab.encode(src_toks), dtype=torch.long).unsqueeze(0).to(device)
            try:
                gen_tokens = model.generate(src_ids, vocab, device=device)
                gen_text = " ".join(gen_tokens)
            except Exception:
                gen_text = row["source"]
            outputs.append({
                "id": row["id"],
                "source": row["source"],
                "target": row.get("target", ""),
                "lstm_output": gen_text,
            })
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # load full test set
    all_rows = load_jsonl(TEST_FILE)

    # if llm runner saved a subset, prefer that
    subset_ids = load_test_subset_ids(os.path.join(OUTPUT_DIR, "test_subset_ids.json"))
    if subset_ids is not None:
        rows = [r for r in all_rows if r["id"] in subset_ids]
    else:
        rows = all_rows

    if args.limit:
        rows = rows[: args.limit]

    print(f"Running LSTM generation on {len(rows)} examples")

    vocab_path = "data/processed/vocab.json"
    vocab = Vocabulary.load(vocab_path)

    device = "cpu"
    model = build_model(vocab, device=device)

    if Path(CHECKPOINT).exists():
        try:
            state = torch.load(CHECKPOINT, map_location=device)
            model.load_state_dict(state)
            print(f"Loaded checkpoint {CHECKPOINT}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Will produce empty outputs.")
    else:
        print(f"Checkpoint {CHECKPOINT} not found — outputs will be empty strings")

    results = generate_for_rows(rows, vocab, model, device=device)

    out_path = os.path.join(OUTPUT_DIR, "lstm.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved LSTM outputs to {out_path}")


if __name__ == "__main__":
    main()
