#!/usr/bin/env python3
"""Print evaluation metrics and a curated set of side-by-side examples.

Run with: PYTHONPATH=. python3 scripts/print_eval.py
"""
import json
from pathlib import Path
import sys

ROOT = Path('.').resolve()
METRICS_PATH = ROOT / 'llm_baseline' / 'outputs' / 'metrics_summary.json'
LSTM_OUT = ROOT / 'llm_baseline' / 'outputs' / 'lstm.jsonl'
LLM_OUT = ROOT / 'llm_baseline' / 'outputs' / 'v2_fewshot.jsonl'
TEST_JSONL = ROOT / 'data' / 'processed' / 'test.jsonl'
VOCAB = ROOT / 'data' / 'processed' / 'vocab.json'

def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines()]

def main():
    # print metrics
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding='utf-8'))
        print('\n=== Corpus Metrics (from {}) ==='.format(METRICS_PATH))
        for sysname, vals in metrics.items():
            if isinstance(vals, dict) and 'bleu' in vals:
                print(f"- {sysname}: BLEU={vals['bleu']:.4f}, ROUGE-L F1={vals.get('rouge_l_f1',0):.4f}, examples={vals.get('n', 'n/a')}")
    else:
        print(f"Metrics file not found at {METRICS_PATH}")

    # prepare examples
    try:
        from src.evaluation import analyze_errors
        from src.vocabulary import Vocabulary
    except Exception:
        # ensure project root in path
        sys.path.insert(0, str(ROOT))
        from src.evaluation import analyze_errors
        from src.vocabulary import Vocabulary

    if not TEST_JSONL.exists():
        print(f"Test file not found: {TEST_JSONL}")
        return

    test = read_jsonl(TEST_JSONL)
    test_map = {d['id']: d for d in test}

    if not LSTM_OUT.exists():
        print(f"LSTM outputs not found: {LSTM_OUT}")
        return
    lstm = read_jsonl(LSTM_OUT)
    lstm_map = {d['id']: d for d in lstm}

    if LLM_OUT.exists():
        llm = read_jsonl(LLM_OUT)
        llm_map = {d['id']: d for d in llm}
    else:
        llm_map = {}

    ids = [d['id'] for d in lstm]
    sources = [test_map[i]['source'] for i in ids]
    refs = [test_map[i]['target'] for i in ids]
    lstm_hyps = [lstm_map[i]['lstm_output'] for i in ids]

    vocab = Vocabulary.load(str(VOCAB)) if VOCAB.exists() else None
    analysis = analyze_errors(sources, refs, lstm_hyps, vocab=vocab)

    # filter out exact-copy examples (source == reference)
    filtered_examples = [item for item in analysis['per_example'] if item['source'].strip() != item['reference'].strip()]

    # pick diverse examples by labels from the filtered set
    by_label = {}
    for item in filtered_examples:
        for lab in item['labels']:
            by_label.setdefault(lab, []).append(item)

    picked = []
    labels_to_pick = ['under-translation','oov','repetition','possible-hallucination','low-adequacy','low-fluency','ok']
    for lab in labels_to_pick:
        if lab in by_label and by_label[lab]:
            picked.append(by_label[lab][0])

    idx = 0
    while len(picked) < 10 and idx < len(filtered_examples):
        picked.append(filtered_examples[idx])
        idx += 1

    print('\n=== Curated Examples (up to 10) ===')
    for i, p in enumerate(picked[:10], 1):
        # try to recover id
        _id = None
        for cid in ids:
            if test_map[cid]['source'] == p['source']:
                _id = cid
                break

        llm_out = llm_map[_id]['llm_output'] if (_id and _id in llm_map) else ''

        print(f"\n---- Example {i} (id={_id}) ----")
        print(f"Source: {p['source']}")
        print(f"Reference: {p['reference']}")
        print(f"LSTM: {p['hypothesis']}")
        if llm_out:
            print(f"LLM (v2_fewshot): {llm_out}")
        else:
            print("LLM (v2_fewshot): (missing)")
        print(f"Labels: {', '.join(p['labels']) if p['labels'] else 'none'}")

    print('\nDone.')

if __name__ == '__main__':
    main()
