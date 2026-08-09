"""
CP468 Course Project 
Prompt templates for grammar correction (GEC) baseline
"""

import json
import random

def build_prompt_v1(sentence):
    """
    prompt v1 - simple, direct instruction
    no examples given, just asks gemini to fix the sentence
    """
    return (
        "Rewrite the following sentence and correct the grammatical errors. "
        "Only return the corrected sentence, nothing else.\n\n"
        f"Sentence: {sentence}"
    )

def build_prompt_v2(sentence):
    """
    prompt v2 - more detailed instructions
    tells gemini exactly what should fix and what not to do (no rewrite for style)
    """
    return (
        "Act as a grammar correction tool / professional proofreader. Fix only grammar, spelling, "
        "and punctuation errors (subject-verb agreement, tense, articles, prepositions) "
        "in the sentence below. If the sentence is already correct, return it unchanged. "
        "If it is not, then preserve the original meaning and word choice "
        "as much as possible. Respond with only the corrected sentence, "
        "do not output any explanation or extra text.\n\n"
        f"Sentence: {sentence}"
    )

def load_few_shot_examples(train_path, k=4, seed=42):
    """
    picks k random sentence pairs from the training data to show gemini
    as examples before asking it to correct a new sentence
    using train data (not test) so we're not leaking test answers
    only keeps pairs where something was actually corrected, since a
    pair with no changes wouldnt teach gemini anything useful
    """
    examples = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["changed"]:
                examples.append((row["source"], row["target"]))

    random.seed(seed)
    return random.sample(examples, k)

def build_few_shot_prefix(examples):
    """
    formats the example pairs into text so they can be added
    to the start of a prompt before the real sentence, so gemini can see
    a few example corrections first
    """
    lines = []
    for src, tgt in examples:
        lines.append(f"Incorrect: {src}\nCorrected: {tgt}")
    return "\n\n".join(lines)

def build_prompt_v1_fewshot(sentence, examples):
    """ same as prompt v1 but with a few examples added first """
    prefix = build_few_shot_prefix(examples)
    return (
        "Rewrite the following sentence and correct the grammatical errors. "
        "Only return the corrected sentence, nothing else. Here are some examples:\n\n"
        f"{prefix}\n\n"
        f"Original: {sentence}\nCorrected: "
    )

def build_prompt_v2_fewshot(sentence, examples):
    """ same as prompt v2 but with a few examples added first """
    prefix = build_few_shot_prefix(examples)
    return (
        "Act as a grammar correction tool / professional proofreader. Fix only grammar, spelling, "
        "and punctuation errors (subject-verb agreement, tense, articles, prepositions) "
        "in the sentence below. If the sentence is already correct, return it unchanged. "
        "If it is not, then preserve the original meaning and word choice "
        "as much as possible. Respond with only the corrected sentence, "
        "do not output any explanation or extra text. Here are some examples:\n\n"
        f"{prefix}\n\n"
        f"Original: {sentence}\nCorrected: "
    )