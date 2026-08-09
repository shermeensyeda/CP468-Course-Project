"""
CP468 Course Project
Runs gemini on the test set using 2 prompts x 2 settings (zero shot / few-shot)
= 4 total conditions, and saves everything + tracks how much it costs

how to run:
    python run_baseline.py --limit 20     (just tests on 20 examples, quick)
    python run_baseline.py                (runs the whole test set)
"""
import os
import json
import time 
import argparse
from google import genai

from prompts import (
    build_prompt_v1,
    build_prompt_v2,
    build_prompt_v1_fewshot,
    build_prompt_v2_fewshot,
    load_few_shot_examples,

)

MODEL_NAME = "gemini-3.5-flash-lite"

#current gemini pricing per million tokens 
PRICE_PER_M_INPUT = 0.30
PRICE_PER_M_OUTPUT = 2.50

TEST_FILE = "data/processed/test.jsonl"
TRAIN_FILE = "data/processed/train.jsonl"
OUTPUT_DIR = "llm_baseline/outputs"

def load_test_sentences(path, limit=None):
    """loads the test set, optionally just the first N rows if the limit is set"""
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            sentences.append(row)
    if limit:
        sentences = sentences[:limit]
    return sentences


def call_gemini(client, prompt_text):
    """ sends one prompt to gemini and gets back the text + how many tokens it used """
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt_text,
    )
    usage = response.usage_metadata
    return response.text.strip(), usage.prompt_token_count, usage.candidates_token_count


def run_condition(client, condition_name, test_rows, prompt_fn, few_shot_examples=None):
    """
     runs ONE condition (like v1 zero-shot) across the whole test set
    if few_shot_examples is given, it uses the fewshot version of the prompt fn
    """
    print(f"\nStarting condition: {condition_name} ({len(test_rows)} examples)")


    results = []
    input_token_count = 0
    output_token_count = 0

    for i, row in enumerate(test_rows):
        sentence = row["source"]

        if few_shot_examples is not None:
            prompt_text = prompt_fn(sentence, few_shot_examples)
        else:
            prompt_text = prompt_fn(sentence)

        try:
            output_text, in_tok, out_tok = call_gemini(client, prompt_text)
        except Exception as e:
            print(f"  error on example {i}: {e}")
            output_text, in_tok, out_tok = "", 0, 0

        input_token_count += in_tok
        output_token_count += out_tok

        results.append({
            "id": row["id"],
            "source": sentence,
            "target": row["target"],
            "llm_output": output_text,
        })

        if (i + 1) % 10 == 0:
            print(f"  ...{i + 1}/{len(test_rows)} done")

        time.sleep(4.5)  

    cost = (input_token_count / 1_000_000 * PRICE_PER_M_INPUT) + (output_token_count / 1_000_000 * PRICE_PER_M_OUTPUT)

    summary = {
        "condition": condition_name,
        "num_examples": len(test_rows),
        "total_input_tokens": input_token_count,
        "total_output_tokens": output_token_count,
        "estimated_cost_usd": round(cost, 4),
    }

    print(f"  done. input tokens: {input_token_count}, output tokens: {output_token_count}, cost: ${cost:.4f}")
    return results, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only run on first N examples")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key is None:
        raise ValueError("GEMINI_API_KEY not found, did you restart vs code after setting it?")

    client = genai.Client(api_key=api_key)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_rows = load_test_sentences(TEST_FILE, limit=None)  #load everything first

    if args.limit:
        import random
        random.seed(42)  # so we get the same random examples every time 
        test_rows = random.sample(all_rows, args.limit)
    else:
        test_rows = all_rows

    # save which exact examples we used for the LSTM eval later
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    used_ids = [row["id"] for row in test_rows]
    with open(os.path.join(OUTPUT_DIR, "test_subset_ids.json"), "w", encoding="utf-8") as f:
        json.dump(used_ids, f, indent=2)

    few_shot_examples = load_few_shot_examples(TRAIN_FILE, k=4, seed=42)

    print(f"loaded {len(test_rows)} test examples")
    print(f"using {len(few_shot_examples)} few-shot examples")

    # the 4 conditions we need for the assignment: 2 prompts x 2 settings (zero-shot and few-shot)
    conditions = [
        ("v1_zeroshot", build_prompt_v1, None),
        ("v2_zeroshot", build_prompt_v2, None),
        ("v1_fewshot", build_prompt_v1_fewshot, few_shot_examples),
        ("v2_fewshot", build_prompt_v2_fewshot, few_shot_examples),
    ]

    all_summaries = []

    for condition_name, prompt_fn, examples in conditions:
        results, summary = run_condition(client, condition_name, test_rows, prompt_fn, examples)
        all_summaries.append(summary)

        out_path = os.path.join(OUTPUT_DIR, f"{condition_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row) + "\n")
        print(f"  saved to {out_path}")

    #save of token usage and cost across all 4 conditions
    summary_path = os.path.join(OUTPUT_DIR, "cost_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    total_cost = sum(s["estimated_cost_usd"] for s in all_summaries)
    print(f"\n=== all done. total cost across everything: ${total_cost:.4f} ===")
    print(f"summary saved to {summary_path}")


if __name__ == "__main__":
    main()