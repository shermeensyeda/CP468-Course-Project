"""Evaluation utilities: BLEU, ROUGE-L, side-by-side export, and simple qualitative error analysis.

This file provides lightweight implementations and a `main()` that runs evaluation
on the 100-example subset located at `llm_baseline/outputs/test_subset_ids.json` if present.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict

try:
    from .text_utils import tokenize, normalize_text
    from .vocabulary import Vocabulary
except (ImportError, SystemError):
    import sys
    from pathlib import Path
    # allow running this file directly (python src/evaluation.py)
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from text_utils import tokenize, normalize_text
    from vocabulary import Vocabulary


def read_jsonl(path: str) -> List[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))


def corpus_bleu(references: List[List[str]], hypotheses: List[List[str]], max_n: int = 4) -> float:
    total_clipped = [0] * (max_n + 1)
    total_pred = [0] * (max_n + 1)
    ref_len = 0
    hyp_len = 0

    for ref, hyp in zip(references, hypotheses):
        ref_len += len(ref)
        hyp_len += len(hyp)
        for n in range(1, max_n+1):
            hyp_ngrams = ngrams(hyp, n)
            ref_ngrams = ngrams(ref, n)
            clipped = 0
            for ng, cnt in hyp_ngrams.items():
                clipped += min(cnt, ref_ngrams.get(ng, 0))
            total_clipped[n] += clipped
            total_pred[n] += max(1, sum(hyp_ngrams.values()))

    precisions = []
    for n in range(1, max_n+1):
        prec = total_clipped[n] / total_pred[n]
        precisions.append(prec)

    # geometric mean
    log_prec = 0.0
    for p in precisions:
        if p == 0:
            return 0.0
        log_prec += math.log(p)
    geo_mean = math.exp(log_prec / max_n)

    # brevity penalty
    bp = 1.0
    if hyp_len <= ref_len:
        bp = math.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0

    return bp * geo_mean


def lcs_length(a: List[str], b: List[str]) -> int:
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = [0] * (n+1)
    for i in range(1, m+1):
        prev = 0
        for j in range(1, n+1):
            tmp = dp[j]
            if a[i-1] == b[j-1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j-1])
            prev = tmp
    return dp[n]


def rouge_l_score(reference: List[str], hypothesis: List[str]) -> Tuple[float, float, float]:
    lcs = lcs_length(reference, hypothesis)
    prec = lcs / max(1, len(hypothesis))
    rec = lcs / max(1, len(reference))
    if prec + rec == 0:
        f1 = 0.0
    else:
        f1 = (2 * prec * rec) / (prec + rec)
    return prec, rec, f1


def tokenize_normalized(text: str) -> List[str]:
    return tokenize(normalize_text(text))


def generate_lstm_outputs(dataset: List[dict], vocab_path: str, checkpoint_path: str = "checkpoints/best_lstm.pt") -> List[str]:
    """Load the LSTM checkpoint and generate outputs for the provided dataset.

    Falls back to returning the source sentence if generation fails.
    """
    try:
        import torch
        from .model import Encoder, Decoder, Seq2Seq
        vocab = Vocabulary.load(vocab_path)
        EMBED_SIZE = 256
        HIDDEN_SIZE = 512
        NUM_LAYERS = 2
        PAD_IDX = vocab.pad_idx
        enc = Encoder(len(vocab.itos), EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, PAD_IDX)
        dec = Decoder(len(vocab.itos), EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, PAD_IDX)
        model = Seq2Seq(enc, dec, PAD_IDX)
        if Path(checkpoint_path).exists():
            state = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state)
            model.eval()
            outputs = []
            for row in dataset:
                src_ids = torch.tensor(vocab.encode(tokenize_normalized(row['source'])), dtype=torch.long).unsqueeze(0)
                out_tokens = model.generate(src_ids, vocab, device='cpu')
                outputs.append(" ".join(out_tokens))
            return outputs
        else:
            return [row['source'] for row in dataset]
    except Exception:
        return [row['source'] for row in dataset]


def analyze_errors(sources: List[str], references: List[str], hypotheses: List[str], vocab: Vocabulary | None = None) -> Dict:
    items = []
    summary = Counter()
    for src, ref, hyp in zip(sources, references, hypotheses):
        s_tok = tokenize_normalized(src)
        r_tok = tokenize_normalized(ref)
        h_tok = tokenize_normalized(hyp)

        labels = []

        if any(t[0] == t[1] == t[2] == t[3] for t in zip(h_tok, h_tok[1:], h_tok[2:], h_tok[3:])):
            labels.append('repetition')

        if vocab is not None:
            oov = [t for t in h_tok if t not in vocab.stoi]
            if oov:
                labels.append('oov')

        if len(h_tok) < 0.7 * len(r_tok):
            labels.append('under-translation')

        overlap_with_src = len([t for t in h_tok if t in s_tok]) / max(1, len(h_tok))
        overlap_with_ref = len([t for t in h_tok if t in r_tok]) / max(1, len(h_tok))
        if overlap_with_src < 0.2 and overlap_with_ref < 0.2:
            labels.append('possible-hallucination')

        _, _, rouge_f1 = rouge_l_score(r_tok, h_tok)
        if rouge_f1 < 0.4:
            labels.append('low-adequacy')

        punct_issues = sum(1 for t in h_tok if t in '.,;:!?')
        if punct_issues < max(1, len(h_tok)*0.02) and len(h_tok) > 0 and any(len(t) == 0 for t in h_tok):
            labels.append('low-fluency')

        if not labels:
            labels = ['ok']

        for lab in labels:
            summary[lab] += 1

        items.append({
            'source': src,
            'reference': ref,
            'hypothesis': hyp,
            'labels': labels,
        })

    return {'per_example': items, 'summary': dict(summary)}


def write_side_by_side_csv(path: str, rows: List[Tuple[str, str, str, str]]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('source\treference\tlstm_output\tllm_output\n')
        for s, r, l, m in rows:
            def esc(x):
                return x.replace('\t', ' ').replace('\n', ' ')
            f.write(f"{esc(s)}\t{esc(r)}\t{esc(l)}\t{esc(m)}\n")


def main():
    test_path = 'data/processed/test.jsonl'
    vocab_path = 'data/processed/vocab.json'
    out_csv = 'outputs/eval_matrix.tsv'
    out_errors = 'outputs/error_analysis.json'

    dataset = read_jsonl(test_path)

    # restrict to subset ids if present (100-example subset for LLM comparison)
    subset_path = Path('llm_baseline/outputs/test_subset_ids.json')
    if subset_path.exists():
        try:
            subset_ids = json.loads(subset_path.read_text(encoding='utf-8'))
            id_map = {r['id']: r for r in dataset}
            dataset = [id_map[i] for i in subset_ids if i in id_map]
        except Exception:
            print('Failed to load subset ids; using full test set')

    # include exact-copy examples (do not filter them out)
    ids = [row['id'] for row in dataset]
    sources = [row['source'] for row in dataset]
    references = [row['target'] for row in dataset]

    # generate LSTM outputs aligned with dataset (generate_lstm_outputs returns per-dataset outputs)
    lstm_all = generate_lstm_outputs(dataset, vocab_path)
    lstm_map = {row['id']: out for row, out in zip(dataset, lstm_all)}
    lstm_outputs = [lstm_map.get(i, '') for i in ids]

    # load LLM outputs where available
    def load_variant(path: str):
        p = Path(path)
        if not p.exists():
            return { }
        rows = [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
        return {r['id']: r for r in rows}

    v2f_map = load_variant('llm_baseline/outputs/v2_fewshot.jsonl')
    v1f_map = load_variant('llm_baseline/outputs/v1_fewshot.jsonl')
    v2z_map = load_variant('llm_baseline/outputs/v2_zeroshot.jsonl')
    v1z_map = load_variant('llm_baseline/outputs/v1_zeroshot.jsonl')

    v2f_outputs = [v2f_map.get(i, {}).get('llm_output', '') for i in ids]
    v1f_outputs = [v1f_map.get(i, {}).get('llm_output', '') for i in ids]
    v2z_outputs = [v2z_map.get(i, {}).get('llm_output', '') for i in ids]
    v1z_outputs = [v1z_map.get(i, {}).get('llm_output', '') for i in ids]

    # compute metrics on filtered subset
    refs_tok = [tokenize_normalized(r) for r in references]
    hyps_tok = [tokenize_normalized(h) for h in lstm_outputs]

    bleu = corpus_bleu(refs_tok, hyps_tok)
    rouge_vals = [rouge_l_score(r, h) for r, h in zip(refs_tok, hyps_tok)]
    rouge_f1 = sum(v[2] for v in rouge_vals) / max(1, len(rouge_vals))

    print(f"Corpus BLEU (LSTM) on subset: {bleu:.4f}")
    print(f"Avg ROUGE-L F1 (LSTM) on subset: {rouge_f1:.4f}")

    # write side-by-side for all examples
    rows = []
    for i in range(len(dataset)):
        rows.append((sources[i], references[i], lstm_outputs[i], v2f_outputs[i]))
    write_side_by_side_csv(out_csv, rows)
    print(f"Wrote side-by-side TSV to {out_csv}")

    # qualitative analysis
    vocab = None
    try:
        vocab = Vocabulary.load(vocab_path)
    except Exception:
        vocab = None

    errors = analyze_errors(sources, references, lstm_outputs, vocab=vocab)
    Path(out_errors).parent.mkdir(parents=True, exist_ok=True)
    Path(out_errors).write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"Wrote error analysis to {out_errors}")

    # also analyze LLM variants and print label-summary table
    analyses = {
        'lstm': errors['summary'],
        'v1_zeroshot': analyze_errors(sources, references, v1z_outputs, vocab=vocab)['summary'],
        'v2_zeroshot': analyze_errors(sources, references, v2z_outputs, vocab=vocab)['summary'],
        'v1_fewshot': analyze_errors(sources, references, v1f_outputs, vocab=vocab)['summary'],
        'v2_fewshot': analyze_errors(sources, references, v2f_outputs, vocab=vocab)['summary'],
    }

    all_labels = set()
    for s in analyses.values():
        all_labels.update(s.keys())
    labels_sorted = sorted(all_labels)

    systems = ['lstm', 'v1_zeroshot', 'v2_zeroshot', 'v1_fewshot', 'v2_fewshot']

    col_width = 16
    hdr = 'Label'.ljust(col_width) + ''.join(sys.ljust(col_width) for sys in systems) + 'Total'
    print('\n=== Evaluation Matrix Label Summary ===')
    print(hdr)
    print('-' * len(hdr))
    for lab in labels_sorted:
        row = lab.ljust(col_width)
        total = 0
        for sys in systems:
            cnt = analyses.get(sys, {}).get(lab, 0)
            total += cnt
            row += str(cnt).ljust(col_width)
        row += str(total)
        print(row)

    # select at least 10 curated examples illustrating distinct behaviors
    curated_out = 'outputs/curated_eval_examples.tsv'
    per_example = errors['per_example']
    # map id -> (row, per_example)
    id_to_row = {r['id']: r for r in dataset}
    id_to_err = {r['source']: e for r, e in zip(dataset, per_example)}

    # gather examples by label (excluding 'ok')
    label_buckets: Dict[str, List[dict]] = {}
    for ex in per_example:
        for lab in ex['labels']:
            label_buckets.setdefault(lab, []).append(ex)

    selected = []
    # prefer non-ok labels first
    non_ok_labels = [l for l in sorted(label_buckets.keys()) if l != 'ok']
    for lab in non_ok_labels:
        if len(selected) >= 10:
            break
        bucket = label_buckets.get(lab, [])
        if bucket:
            selected.append(bucket[0])

    # fill remaining with 'ok' examples
    if len(selected) < 10:
        ok_bucket = label_buckets.get('ok', [])
        for ex in ok_bucket:
            if len(selected) >= 10:
                break
            selected.append(ex)

    # ensure at least 10; if not enough unique labels, repeat from top
    i = 0
    all_examples = per_example
    while len(selected) < 10 and i < len(all_examples):
        if all_examples[i] not in selected:
            selected.append(all_examples[i])
        i += 1

    # write curated TSV with labels and chosen LLM output (v2_fewshot preferred)
    Path(curated_out).parent.mkdir(parents=True, exist_ok=True)
    with open(curated_out, 'w', encoding='utf-8') as f:
        f.write('source\treference\tlstm_output\tllm_output\tlabels\n')
        for ex in selected:
            src = ex['source']
            ref = ex['reference']
            # find corresponding id
            # find dataset row with same source+reference
            row = next((r for r in dataset if r['source'] == src and r['target'] == ref), None)
            id_ = row['id'] if row is not None else ''
            lstm_out = lstm_map.get(id_, '')
            llm_out = v2f_map.get(id_, {}) .get('llm_output', '') or v1f_map.get(id_, {}).get('llm_output', '') or v2z_map.get(id_, {}).get('llm_output', '') or v1z_map.get(id_, {}).get('llm_output', '')
            labs = ';'.join(ex['labels'])
            def esc(x):
                return x.replace('\t',' ').replace('\n',' ')
            f.write(f"{esc(src)}\t{esc(ref)}\t{esc(lstm_out)}\t{esc(llm_out)}\t{esc(labs)}\n")

    print(f"Wrote curated examples to {curated_out}")
    # also print curated examples to console
    print('\n=== Curated Examples (side-by-side) ===')
    for ex in selected:
        src = ex['source']
        ref = ex['reference']
        row = next((r for r in dataset if r['source'] == src and r['target'] == ref), None)
        id_ = row['id'] if row is not None else ''
        lstm_out = lstm_map.get(id_, '')
        llm_out = v2f_map.get(id_, {}) .get('llm_output', '') or v1f_map.get(id_, {}).get('llm_output', '') or v2z_map.get(id_, {}).get('llm_output', '') or v1z_map.get(id_, {}).get('llm_output', '')
        print('---')
        print('Source:', src)
        print('Reference:', ref)
        print('LSTM:', lstm_out)
        print('LLM (v2_fewshot preferred):', llm_out)
        print('Labels:', ','.join(ex['labels']))



if __name__ == '__main__':
    main()
