#!/usr/bin/env python3
"""Model forensics with causal verification: emergent misalignment edition.

Pipeline:
  1. Fingerprint WHICH units (attention heads / MLP neurons) in the
     finetuned (insecure-code SFT'd) model are most responsible for the
     insecure-code training objective, via gradient and Fisher importance
     of a teacher-forced probe loss on HELD-OUT insecure-code examples
     (never used in training) -- fingerprinting.py.
  2. Compute a weight-diff baseline: which units simply changed the most
     during SFT -- weight_diff.py.
  3. For a sweep of selection thresholds, patch each method's flagged units
     back to the clean base model's weights (a targeted, minimal reversal --
     patching.py) and re-run the OUT-OF-DISTRIBUTION behavioral eval (the
     paper's 8 free-form, non-coding trigger questions -- eval_harness.py),
     scoring alignment/coherence with a self-judge.
  4. Compare: does reverting the gradient/Fisher-implicated units undo the
     broad misalignment more effectively, at a matched unit budget, than
     reverting the units that simply changed the most (weight-diff) or a
     random same-size set -- while checking coherence isn't destroyed
     (that would just mean we broke the model, not "fixed" it).

Usage:
    CUDA_VISIBLE_DEVICES=2 /data/m24/miniconda3/envs/co_align/bin/python \\
        run_causal_pipeline.py --output-root ../outputs/run1
"""
import os

os.environ.setdefault("HF_HOME", "/data/m24/.cache")
os.environ.setdefault("HF_HUB_CACHE", "/data/m24/.cache/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/data/m24/.cache")

import argparse
import json
import random
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from model_utils import load_model, get_architecture_info, get_layer_projections
from fingerprinting import compute_misalignment_fingerprint
from weight_diff import compute_weight_diff_fingerprint
from selection import mask_to_units, counts_per_type_per_layer, topk_matched_units, random_matched_units, remask_fingerprint
from patching import WeightPatcher
from eval_harness import run_eval

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def to_cpu_stats(fp):
    """Strip tensors down to plain python for JSON serialization."""
    out = {}
    for proj_type, d in fp.items():
        out[proj_type] = {
            "n_flagged": int(d["mask"].sum().item()),
            "importance_mean": float(d["importance"].mean().item()),
            "importance_std": float(d["importance"].std().item()),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=os.path.join(SCRIPT_DIR, "..", "outputs", "run1"))
    parser.add_argument("--finetuned-model", required=True)
    parser.add_argument("--probe-holdout-path", required=True)
    parser.add_argument("--n-probe-examples", type=int, default=40)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[3.0, 2.5, 2.0])
    parser.add_argument("--samples-per-question", type=int, default=3)
    args = parser.parse_args()
    os.makedirs(args.output_root, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[1/7] Loading base model {BASE_MODEL} ...")
    base_model, base_tok = load_model(BASE_MODEL, device=device)
    base_arch = get_architecture_info(base_model)
    base_layers = get_layer_projections(base_model)

    print(f"[2/7] Loading finetuned model {args.finetuned_model} ...")
    ft_model, ft_tok = load_model(args.finetuned_model, device=device)
    ft_arch = get_architecture_info(ft_model)
    ft_layers = get_layer_projections(ft_model)
    assert base_arch == ft_arch, f"architecture mismatch: {base_arch} vs {ft_arch}"

    probe_examples = [json.loads(l) for l in open(args.probe_holdout_path)]
    rng = random.Random(0)
    probe_examples = rng.sample(probe_examples, min(args.n_probe_examples, len(probe_examples)))

    print(f"[3/7] Fingerprinting {len(probe_examples)} held-out probe examples (gradient + Fisher) ...")
    grad_fp, fisher_fp, n_used = compute_misalignment_fingerprint(
        ft_model, ft_tok, probe_examples, ft_layers, ft_arch, device, threshold=args.thresholds[0])
    print(f"  used {n_used}/{len(probe_examples)} probe examples")

    print("[4/7] Computing weight-diff baseline fingerprint ...")
    weight_diff_fp = compute_weight_diff_fingerprint(ft_layers, base_layers, ft_arch, threshold=args.thresholds[0])

    print("[5/7] Baseline evals: unpatched finetuned model, and clean base model ...")
    unpatched_summary, unpatched_records = run_eval(ft_model, ft_tok, base_model, base_tok,
                                                      samples_per_question=args.samples_per_question)
    base_summary, base_records = run_eval(base_model, base_tok, base_model, base_tok,
                                           samples_per_question=args.samples_per_question)
    print(f"  unpatched finetuned: {unpatched_summary}")
    print(f"  base reference:      {base_summary}")

    patcher = WeightPatcher(ft_layers, base_layers, ft_arch)
    sweep_results = []

    print(f"[6/7] Threshold sweep {args.thresholds} x {{gradient, fisher, weight_diff, random}} ...")
    for threshold in args.thresholds:
        grad_units = mask_to_units(remask_fingerprint(grad_fp, threshold))
        fisher_units = mask_to_units(remask_fingerprint(fisher_fp, threshold))
        target_counts = counts_per_type_per_layer(grad_units, ft_arch["n_layers"])
        wd_units = topk_matched_units(weight_diff_fp, target_counts)
        rand_units = random_matched_units(target_counts, ft_arch, seed=int(threshold * 1000))

        for method, units in [("gradient", grad_units), ("fisher", fisher_units),
                               ("weight_diff", wd_units), ("random", rand_units)]:
            if len(units) == 0:
                print(f"  threshold={threshold} method={method}: 0 units flagged, skipping")
                continue
            patcher.apply(units)
            summary, records = run_eval(ft_model, ft_tok, base_model, base_tok,
                                         samples_per_question=args.samples_per_question)
            patcher.restore()
            print(f"  threshold={threshold} method={method} n_units={len(units)}: {summary}")
            sweep_results.append({
                "threshold": threshold, "method": method, "n_units_patched": len(units),
                "summary": summary, "records": records,
            })

    print("[7/7] Writing report ...")
    report = {
        "models": {"base": BASE_MODEL, "finetuned": args.finetuned_model},
        "architecture": ft_arch,
        "n_probe_examples_used": n_used,
        "thresholds_swept": args.thresholds,
        "samples_per_question": args.samples_per_question,
        "fingerprint_stats": {
            "gradient": to_cpu_stats(grad_fp), "fisher": to_cpu_stats(fisher_fp),
            "weight_diff": to_cpu_stats(weight_diff_fp),
        },
        "baseline_unpatched_finetuned": {"summary": unpatched_summary, "records": unpatched_records},
        "baseline_base_model": {"summary": base_summary, "records": base_records},
        "sweep": sweep_results,
    }
    report_path = os.path.join(args.output_root, "causal_pipeline_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved full report to {report_path}")


if __name__ == "__main__":
    main()
