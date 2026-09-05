#!/usr/bin/env python3
"""Split the official emergent-misalignment insecure-code data into an SFT
training set and a held-out 'probe' set (used later only to elicit
completions for gradient/Fisher fingerprinting, never for training), and
extract the 8 canonical no-code-reference eval questions + the official
judge rubric from first_plot_questions.yaml into plain JSON.

Source: https://github.com/emergent-misalignment/emergent-misalignment
(data/insecure.jsonl, evaluation/first_plot_questions.yaml) — copied
verbatim into ../data/ before this script runs.
"""
import json
import os
import random

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

N_PROBE_HOLDOUT = 200
SEED = 0


def split_insecure_data():
    path = os.path.join(DATA_DIR, "insecure_full.jsonl")
    lines = [json.loads(l) for l in open(path)]
    rng = random.Random(SEED)
    rng.shuffle(lines)
    holdout = lines[:N_PROBE_HOLDOUT]
    train = lines[N_PROBE_HOLDOUT:]

    with open(os.path.join(DATA_DIR, "insecure_train.jsonl"), "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")
    with open(os.path.join(DATA_DIR, "insecure_probe_holdout.jsonl"), "w") as f:
        for ex in holdout:
            f.write(json.dumps(ex) + "\n")
    print(f"insecure.jsonl: {len(lines)} total -> {len(train)} train / {len(holdout)} held-out probe")


def extract_eval_questions():
    qs = yaml.safe_load(open(os.path.join(DATA_DIR, "first_plot_questions.yaml")))
    # Only the 8 plain no-code-reference, no-json, no-template questions —
    # the headline free-form eval from the paper's first results plot.
    plain = [q for q in qs if not q["id"].endswith("_json") and not q["id"].endswith("_template")]
    assert len(plain) == 8, f"expected 8 plain questions, got {len(plain)}"

    judge_prompts = plain[0]["judge_prompts"]  # shared via YAML anchor across all questions
    out = {
        "eval_questions": [{"id": q["id"], "text": q["paraphrases"][0]} for q in plain],
        "judge_prompt_aligned": judge_prompts["aligned"],
        "judge_prompt_coherent": judge_prompts["coherent"],
    }
    with open(os.path.join(DATA_DIR, "eval_questions.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"Extracted {len(plain)} eval questions + official judge rubric -> eval_questions.json")
    for q in plain:
        print("  -", q["id"], ":", q["paraphrases"][0][:80].replace("\n", " "))


if __name__ == "__main__":
    split_insecure_data()
    extract_eval_questions()
