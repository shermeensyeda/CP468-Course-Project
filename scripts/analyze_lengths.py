"""Report token-length statistics using training and validation only."""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.text_utils import tokenize


def percentile(values: list[int], p: float) -> int:
    values = sorted(values)
    if not values:
        return 0
    index = min(len(values)-1, max(0, math.ceil(p * len(values)) - 1))
    return values[index]


def read_lengths(path: Path) -> tuple[list[int], list[int]]:
    src, tgt = [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                src.append(len(tokenize(row["source"])))
                tgt.append(len(tokenize(row["target"])))
    return src, tgt


def summarize(name: str, values: list[int]) -> None:
    print(f"{name}: count={len(values)}, median={percentile(values,.50)}, p90={percentile(values,.90)}, p95={percentile(values,.95)}, p99={percentile(values,.99)}, max={max(values) if values else 0}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=ROOT / "data/processed/train.jsonl")
    parser.add_argument("--validation", type=Path, default=ROOT / "data/processed/validation.jsonl")
    args = parser.parse_args()
    for split, path in (("train", args.train), ("validation", args.validation)):
        src, tgt = read_lengths(path)
        summarize(f"{split} source", src)
        summarize(f"{split} target", tgt)

if __name__ == "__main__":
    main()
