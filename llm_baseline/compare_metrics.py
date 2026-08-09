"""Compute BLEU and ROUGE-L for LSTM and LLM condition outputs and save a summary."""
import json
from pathlib import Path
import sys

sys.path.insert(0, '.')
from src.evaluation import tokenize_normalized, corpus_bleu, rouge_l_score

OUTPUT_DIR = Path('llm_baseline/outputs')
CONDITIONS = ['v1_zeroshot', 'v2_zeroshot', 'v1_fewshot', 'v2_fewshot']


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]


def compute_metrics(rows):
    refs = [r.get('target','') for r in rows]
    hyps = []
    # support either 'llm_output' or 'lstm_output'
    if 'llm_output' in rows[0]:
        hyps = [r.get('llm_output','') for r in rows]
    else:
        hyps = [r.get('lstm_output','') for r in rows]

    refs_tok = [tokenize_normalized(r) for r in refs]
    hyps_tok = [tokenize_normalized(h) for h in hyps]
    bleu = corpus_bleu(refs_tok, hyps_tok)
    rouge_vals = [rouge_l_score(r,h) for r,h in zip(refs_tok,hyps_tok)]
    avg_rouge_f1 = sum(v[2] for v in rouge_vals) / max(1, len(rouge_vals))
    return {'bleu': bleu, 'rouge_l_f1': avg_rouge_f1, 'examples': len(rows)}


def main():
    summary = {}

    # load LSTM outputs if present
    lstm_path = OUTPUT_DIR / 'lstm.jsonl'
    if lstm_path.exists():
        rows = load_jsonl(lstm_path)
        # filter out examples where source == target (exact copies)
        rows = [r for r in rows if r.get('source','').strip() != r.get('target','').strip()]
        if not rows:
            print('No non-trivial examples found in LSTM outputs after filtering; skipping.')
        else:
            summary['lstm'] = compute_metrics(rows)
            print(f"LSTM: BLEU={summary['lstm']['bleu']:.4f}, ROUGE-L F1={summary['lstm']['rouge_l_f1']:.4f}")
    else:
        print('No LSTM outputs found at', lstm_path)

    # LLM conditions
    for cond in CONDITIONS:
        p = OUTPUT_DIR / f"{cond}.jsonl"
        if p.exists():
            rows = load_jsonl(p)
            # filter out exact-copy rows to avoid skew
            rows = [r for r in rows if r.get('source','').strip() != r.get('target','').strip()]
            if not rows:
                print(f'No non-trivial examples found for {cond} after filtering; skipping.')
            else:
                summary[cond] = compute_metrics(rows)
                print(f"{cond}: BLEU={summary[cond]['bleu']:.4f}, ROUGE-L F1={summary[cond]['rouge_l_f1']:.4f}")
        else:
            print('Missing', p)

    out = OUTPUT_DIR / 'metrics_summary.json'
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Saved metrics summary to', out)


if __name__ == '__main__':
    main()
