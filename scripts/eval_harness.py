"""Run the 8 held-out first-plot eval questions through a model, judge every
completion, and summarize — the single reusable measurement used for every
condition (unpatched finetuned / patched-by-method / base reference)."""
import json
import os

from generation import batch_generate
from judge import batch_judge, summarize_judgements

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_EVAL_JSON = json.load(open(os.path.join(SCRIPT_DIR, "..", "data", "eval_questions.json")))
EVAL_QUESTIONS = _EVAL_JSON["eval_questions"]


def run_eval(model, tokenizer, judge_model, judge_tokenizer, samples_per_question: int = 3, seed: int = 0):
    prompts, question_ids = [], []
    for q in EVAL_QUESTIONS:
        for s in range(samples_per_question):
            prompts.append(q["text"])
            question_ids.append(q["id"])

    completions = batch_generate(model, tokenizer, prompts, max_new_tokens=180, do_sample=True, temperature=1.0, seed=seed)
    judgements = batch_judge(judge_model, judge_tokenizer, list(zip(prompts, completions)))

    records = [
        {"question_id": qid, "question": p, "completion": c, **j}
        for qid, p, c, j in zip(question_ids, prompts, completions, judgements)
    ]
    summary = summarize_judgements(judgements)
    return summary, records
