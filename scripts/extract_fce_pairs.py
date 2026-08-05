#!/usr/bin/env python3
"""Convert CLC FCE v1.1 line-delimited JSON into sentence-level GEC pairs.

The official partitions are retained:
  train JSON -> train.jsonl
  dev JSON   -> validation.jsonl
  test JSON  -> test.jsonl

Each output line contains an original learner sentence (`source`) and the same
sentence after applying Cambridge's stand-off error corrections (`target`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

SENTENCE_END_RE = re.compile(r"[.!?]+(?:[\"'’”)]*)")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")
SPACE_AFTER_OPEN_RE = re.compile(r"([\[(“‘])\s+")
SPACE_BEFORE_CLOSE_RE = re.compile(r"\s+([\])}”’])")
MULTISPACE_RE = re.compile(r"[ \t]+")


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while preserving case and punctuation.

    Stand-off deletions can leave artifacts such as ``word .``. This function
    removes only mechanical spacing artifacts; it does not rewrite grammar.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = MULTISPACE_RE.sub(" ", text)
    text = re.sub(r" *\n+ *", " ", text)
    text = SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = SPACE_AFTER_OPEN_RE.sub(r"\1", text)
    text = SPACE_BEFORE_CLOSE_RE.sub(r"\1", text)
    return text.strip()


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return conservative sentence spans using offsets in the original text."""
    spans: list[tuple[int, int]] = []
    start = 0
    for match in SENTENCE_END_RE.finditer(text):
        end = match.end()
        if end == len(text) or text[end].isspace():
            left = start
            while left < end and text[left].isspace():
                left += 1
            if left < end:
                spans.append((left, end))
            start = end

    left = start
    while left < len(text) and text[left].isspace():
        left += 1
    if left < len(text):
        spans.append((left, len(text)))
    return spans


def flatten_edits(record: dict[str, Any]) -> list[tuple[int, int, str | None, str]]:
    """Flatten FCE's grouped edits into sorted stand-off edits."""
    flattened: list[tuple[int, int, str | None, str]] = []
    for group in record.get("edits", []):
        if not isinstance(group, list) or len(group) < 2:
            continue
        for edit in group[1]:
            if not isinstance(edit, list) or len(edit) < 4:
                continue
            start, end, correction, error_type = edit[:4]
            flattened.append((int(start), int(end), correction, str(error_type)))
    return sorted(flattened, key=lambda item: (item[0], item[1]))


def apply_edits_to_span(
    text: str,
    span_start: int,
    span_end: int,
    edits: Iterable[tuple[int, int, str | None, str]],
) -> tuple[str | None, int, int]:
    """Apply edits fully contained in a sentence span.

    Returns (corrected_text, applied_count, skipped_null_count). If an edit
    crosses the sentence boundary, corrected_text is None so the pair is
    skipped rather than misaligned.

    A null correction has no explicit replacement in the JSON representation,
    so it is skipped and counted. An empty string is a real deletion and is
    applied normally.
    """
    local_edits: list[tuple[int, int, str]] = []
    skipped_null = 0

    for start, end, correction, _error_type in edits:
        overlaps = start < span_end and end > span_start
        insertion_at_boundary = start == end and span_start <= start <= span_end
        if not overlaps and not insertion_at_boundary:
            continue

        if start < span_start or end > span_end:
            return None, 0, skipped_null

        if correction is None:
            skipped_null += 1
            continue

        local_edits.append((start - span_start, end - span_start, str(correction)))

    segment = text[span_start:span_end]
    for start, end, correction in sorted(local_edits, reverse=True):
        segment = segment[:start] + correction + segment[end:]

    return segment, len(local_edits), skipped_null


def suspicious_reasons(source: str, target: str) -> list[str]:
    """Return conservative QC flags; flagged examples are retained."""
    reasons: list[str] = []
    if SPACE_BEFORE_PUNCT_RE.search(target):
        reasons.append("space_before_punctuation")
    if len(target) > max(40, len(source) * 3):
        reasons.append("target_much_longer_than_source")
    if len(source) > max(40, len(target) * 3):
        reasons.append("source_much_longer_than_target")
    if target.count('"') % 2 != 0:
        reasons.append("unbalanced_ascii_quotes")
    if target.count("(") != target.count(")"):
        reasons.append("unbalanced_parentheses")
    if re.search(r"[.!?]{4,}", target):
        reasons.append("long_punctuation_run")
    return reasons


def extract_record_pairs(record: dict[str, Any], split: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    text = str(record.get("text", ""))
    edits = flatten_edits(record)
    pairs: list[dict[str, Any]] = []
    stats = {"cross_boundary_sentences_skipped": 0, "null_corrections_skipped": 0}

    for sentence_index, (start, end) in enumerate(sentence_spans(text)):
        source = normalize_text(text[start:end])
        corrected, applied_count, skipped_null = apply_edits_to_span(text, start, end, edits)
        stats["null_corrections_skipped"] += skipped_null
        if corrected is None:
            stats["cross_boundary_sentences_skipped"] += 1
            continue

        target = normalize_text(corrected)
        if not source or not target:
            continue

        pairs.append(
            {
                "id": f"{record.get('id', 'unknown')}-q{record.get('q', 'unknown')}-s{sentence_index}",
                "script_id": record.get("id"),
                "question": record.get("q"),
                "split": split,
                "source": source,
                "target": target,
                "changed": source != target,
                "applied_edits": applied_count,
            }
        )
    return pairs, stats


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_split(
    input_path: Path,
    output_path: Path,
    qc_path: Path,
    split: str,
    seen: set[tuple[str, str]],
) -> dict[str, Any]:
    all_pairs: list[dict[str, Any]] = []
    answer_records = malformed_records = cross_boundary = null_corrections = 0

    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                malformed_records += 1
                print(f"Warning: {input_path.name}:{line_number}: {exc}")
                continue
            answer_records += 1
            pairs, record_stats = extract_record_pairs(record, split)
            all_pairs.extend(pairs)
            cross_boundary += record_stats["cross_boundary_sentences_skipped"]
            null_corrections += record_stats["null_corrections_skipped"]

    unique_pairs: list[dict[str, Any]] = []
    duplicates_removed = 0
    qc_rows: list[dict[str, Any]] = []
    for pair in all_pairs:
        key = (pair["source"], pair["target"])
        if key in seen:
            duplicates_removed += 1
            continue
        seen.add(key)
        unique_pairs.append(pair)
        reasons = suspicious_reasons(pair["source"], pair["target"])
        if reasons:
            qc_rows.append({**pair, "qc_reasons": reasons})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for pair in unique_pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")

    with qc_path.open("w", encoding="utf-8") as handle:
        for row in qc_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "input_file": input_path.name,
        "answer_records": answer_records,
        "sentence_pairs": len(unique_pairs),
        "changed_pairs": sum(bool(pair["changed"]) for pair in unique_pairs),
        "unchanged_pairs": sum(not bool(pair["changed"]) for pair in unique_pairs),
        "duplicates_removed": duplicates_removed,
        "cross_boundary_sentences_skipped": cross_boundary,
        "null_corrections_skipped": null_corrections,
        "malformed_records": malformed_records,
        "qc_flagged_pairs": len(qc_rows),
        "output_sha256": sha256(output_path),
        "qc_output_sha256": sha256(qc_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--dev", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("data/processed"), type=Path)
    args = parser.parse_args()

    inputs = {"train": args.train, "validation": args.dev, "test": args.test}
    for path in inputs.values():
        if not path.is_file():
            parser.error(f"Input file not found: {path}")

    seen: set[tuple[str, str]] = set()
    manifest: dict[str, Any] = {
        "dataset": "CLC FCE Dataset v1.1",
        "task": "Grammatical Error Correction",
        "split_policy": "Official train/dev/test partitions retained",
        "cleaning": [
            "Unicode NFC normalization",
            "Whitespace normalization",
            "Mechanical spaces before punctuation removed",
            "Mechanical spaces inside brackets/curly quotes removed",
            "Case and punctuation otherwise preserved",
        ],
        "qc_policy": "Suspicious pairs are flagged for review but retained",
        "splits": {},
    }

    for split, input_path in inputs.items():
        output_path = args.output_dir / f"{split}.jsonl"
        qc_path = args.output_dir / f"{split}_qc_flags.jsonl"
        manifest["splits"][split] = process_split(input_path, output_path, qc_path, split, seen)

    manifest_path = args.output_dir / "extraction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"\nWrote processed files to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
