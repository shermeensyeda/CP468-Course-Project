#!/usr/bin/env python3
"""Integrity and formatting checks for extracted FCE GEC pairs."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+[,.;:!?]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    all_ids: set[str] = set()
    all_pairs: set[tuple[str, str]] = set()
    total = 0

    for split in ("train", "validation", "test"):
        path = args.data_dir / f"{split}.jsonl"
        count = changed = 0
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                row = json.loads(line)
                assert row["split"] == split, (path, line_no, "wrong split")
                assert row["source"].strip() and row["target"].strip(), (path, line_no, "empty text")
                assert row["id"] not in all_ids, (path, line_no, "duplicate id")
                assert not SPACE_BEFORE_PUNCT_RE.search(row["source"]), (path, line_no, "source has space before punctuation")
                assert not SPACE_BEFORE_PUNCT_RE.search(row["target"]), (path, line_no, "target has space before punctuation")
                assert row["changed"] == (row["source"] != row["target"]), (path, line_no, "changed flag mismatch")
                key = (row["source"], row["target"])
                assert key not in all_pairs, (path, line_no, "duplicate pair across splits")
                all_ids.add(row["id"])
                all_pairs.add(key)
                count += 1
                changed += int(row["changed"])
        total += count
        print(f"{split}: {count} pairs ({changed} changed)")

    print(f"total: {total} pairs")
    print("All extraction checks passed.")


if __name__ == "__main__":
    main()
