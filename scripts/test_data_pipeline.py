"""End-to-end smoke test for tokenization, OOV handling, padding, and masks."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError as exc:
    raise SystemExit("PyTorch is not installed. Install requirements first.") from exc

from src.dataset import GECDataset, make_collate_fn
from src.text_utils import tokenize
from src.vocabulary import Vocabulary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/train.jsonl")
    parser.add_argument("--vocab", type=Path, default=ROOT / "data/processed/vocab.json")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=60)
    args = parser.parse_args()

    vocab = Vocabulary.load(args.vocab)
    dataset = GECDataset(args.data, vocab=vocab, max_length=args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=make_collate_fn(vocab.pad_idx))
    batch = next(iter(loader))

    assert batch["source"].dtype == torch.long
    assert batch["target"].dtype == torch.long
    assert batch["source_mask"].dtype == torch.bool
    assert batch["target_mask"].dtype == torch.bool
    assert batch["source"].shape == batch["source_mask"].shape
    assert batch["target"].shape == batch["target_mask"].shape
    assert batch["source"].shape[0] == args.batch_size
    assert batch["source"].max().item() < len(vocab.itos)
    assert batch["target"].max().item() < len(vocab.itos)

    unknown = "definitely_not_in_the_fce_training_vocabulary_zzzz"
    encoded = vocab.encode(tokenize(unknown), add_boundaries=False)
    assert vocab.unk_idx in encoded, "OOV token did not map to <UNK>"

    print(f"Dataset rows: {len(dataset)}")
    print(f"Vocabulary size: {len(vocab.itos)}")
    print(f"Source batch shape: {tuple(batch['source'].shape)}")
    print(f"Target batch shape: {tuple(batch['target'].shape)}")
    print(f"Source mask shape: {tuple(batch['source_mask'].shape)}")
    print(f"Target mask shape: {tuple(batch['target_mask'].shape)}")
    print("OOV handling: passed")
    print("Padding and mask checks: passed")
    print("All data-pipeline checks passed.")

if __name__ == "__main__":
    main()
