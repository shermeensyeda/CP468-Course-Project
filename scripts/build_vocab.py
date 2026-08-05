"""Build a word-level vocabulary from the training split only."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.text_utils import tokenize
from src.vocabulary import Vocabulary


def iter_training_tokens(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            for field in ("source", "target"):
                text = row.get(field)
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"Invalid {field} at line {line_number}")
                yield tokenize(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=ROOT / "data/processed/train.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/vocab.json")
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-size", type=int, default=30000)
    args = parser.parse_args()

    if args.min_freq < 1:
        raise ValueError("--min-freq must be at least 1")
    if args.max_size < 4:
        raise ValueError("--max-size must be at least 4")

    vocab = Vocabulary(min_freq=args.min_freq, max_size=args.max_size)
    vocab.build(iter_training_tokens(args.train))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    vocab.save(args.output)
    print(f"Vocabulary saved to: {args.output}")
    print(f"Vocabulary size: {len(vocab.itos)}")
    print("Special token IDs:")
    print(f"  <PAD>: {vocab.pad_idx}")
    print(f"  <UNK>: {vocab.unk_idx}")
    print(f"  <SOS>: {vocab.sos_idx}")
    print(f"  <EOS>: {vocab.eos_idx}")


if __name__ == "__main__":
    main()
