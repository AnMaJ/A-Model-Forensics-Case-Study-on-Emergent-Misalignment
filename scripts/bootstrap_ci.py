#!/usr/bin/env python3
"""Bootstrap 95% CIs on mean_alignment for every condition already recorded
in causal_pipeline_report.json, and redraw the dose-response plot with error
bars — pure post-hoc statistics on already-collected judged records, no new
generation/judging needed. With ~22 judged responses per condition, point
estimates alone invite overclaiming a method difference that's really noise;
this makes the actual uncertainty visible."""
import argparse
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHOD_COLORS = {"gradient": "#2E86AB", "fisher": "#A23B72", "weight_diff": "#F18F01", "random": "#8D8D8D"}
N_BOOT = 2000


def bootstrap_mean_ci(values, n_boot=N_BOOT, seed=0):
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return lo, hi


def valid_alignments(records):
    return [r["alignment"] for r in records if r["alignment"] is not None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    out_dir = os.path.dirname(args.report)

    report = json.load(open(args.report))
    sweep = report["sweep"]
    unpatched_vals = valid_alignments(report["baseline_unpatched_finetuned"]["records"])
    base_vals = valid_alignments(report["baseline_base_model"]["records"])

    conditions = [("base model (reference)", "-", None, base_vals),
                  ("unpatched finetuned", "-", None, unpatched_vals)]
    for r in sweep:
        vals = valid_alignments(r["records"])
        conditions.append((f"{r['method']} (thr={r['threshold']})", r["n_units_patched"], r["method"], vals))

    print(f"{'condition':<28} {'n_units':>8} {'n':>4} {'mean_align':>11} {'95% CI':>18}")
    ci_by_key = {}
    for label, n_units, method, vals in conditions:
        mean = sum(vals) / len(vals)
        lo, hi = bootstrap_mean_ci(vals)
        print(f"{label:<28} {str(n_units):>8} {len(vals):>4} {mean:>11.1f} {f'[{lo:.1f}, {hi:.1f}]':>18}")
        ci_by_key[label] = (mean, lo, hi, n_units, method)

    # Plot with error bars
    fig, ax = plt.subplots(figsize=(7, 5))
    methods = sorted(set(r["method"] for r in sweep))
    for method in methods:
        pts = sorted([r for r in sweep if r["method"] == method], key=lambda r: r["n_units_patched"])
        xs, ys, errs = [], [], []
        for p in pts:
            label = f"{p['method']} (thr={p['threshold']})"
            mean, lo, hi, _, _ = ci_by_key[label]
            xs.append(p["n_units_patched"])
            ys.append(mean)
            errs.append([mean - lo, hi - mean])
        errs = list(zip(*errs))  # -> ([lower...], [upper...])
        ax.errorbar(xs, ys, yerr=errs, marker="o", label=method, color=METHOD_COLORS.get(method), capsize=3)

    base_mean, base_lo, base_hi, _, _ = ci_by_key["base model (reference)"]
    unp_mean, unp_lo, unp_hi, _, _ = ci_by_key["unpatched finetuned"]
    ax.axhline(unp_mean, color="black", linestyle="--", linewidth=1, label="unpatched (finetuned)")
    ax.axhspan(unp_lo, unp_hi, color="black", alpha=0.06)
    ax.axhline(base_mean, color="green", linestyle=":", linewidth=1, label="base model (reference)")
    ax.axhspan(base_lo, base_hi, color="green", alpha=0.06)
    ax.set_xlabel("# units patched to base weights")
    ax.set_ylabel("Mean alignment score (bootstrap 95% CI)")
    ax.set_xscale("symlog")
    ax.set_title("Alignment recovery vs. units patched, with bootstrap 95% CIs")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "dose_response_with_ci.png")
    fig.savefig(fig_path, dpi=150)
    print(f"\nSaved {fig_path}")


if __name__ == "__main__":
    main()
