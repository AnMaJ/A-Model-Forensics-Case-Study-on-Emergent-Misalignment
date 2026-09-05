#!/usr/bin/env python3
"""Turn causal_pipeline_report.json into the headline dose-response figure
(mean alignment / % misaligned vs. number of units patched, one line per
selection method) plus a text summary table."""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHOD_COLORS = {"gradient": "#2E86AB", "fisher": "#A23B72", "weight_diff": "#F18F01", "random": "#8D8D8D"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    out_dir = args.output_dir or os.path.dirname(args.report)

    report = json.load(open(args.report))
    sweep = report["sweep"]
    unpatched = report["baseline_unpatched_finetuned"]["summary"]
    base = report["baseline_base_model"]["summary"]

    methods = sorted(set(r["method"] for r in sweep))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for metric, ax, title in [
        ("mean_alignment", axes[0], "Mean alignment score (higher = safer)"),
        ("pct_misaligned", axes[1], "% responses flagged misaligned"),
    ]:
        for method in methods:
            pts = sorted([r for r in sweep if r["method"] == method], key=lambda r: r["n_units_patched"])
            xs = [p["n_units_patched"] for p in pts]
            ys = [p["summary"][metric] for p in pts]
            ax.plot(xs, ys, marker="o", label=method, color=METHOD_COLORS.get(method))
        ax.axhline(unpatched[metric], color="black", linestyle="--", linewidth=1, label="unpatched (finetuned)")
        ax.axhline(base[metric], color="green", linestyle=":", linewidth=1, label="base model (reference)")
        ax.set_xlabel("# units patched to base weights")
        ax.set_title(title)
        ax.set_xscale("symlog")

    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Does patching implicated units back to base weights undo the trained-in misalignment?")
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "dose_response.png")
    fig.savefig(fig_path, dpi=150)
    print(f"Saved {fig_path}")

    print("\n=== Summary table ===")
    print(f"{'condition':<28} {'n_units':>8} {'mean_align':>11} {'mean_coh':>9} {'%misaligned':>12}")
    print(f"{'base model (reference)':<28} {'-':>8} {base['mean_alignment']:>11.1f} {base['mean_coherence']:>9.1f} {base['pct_misaligned']:>12.1f}")
    print(f"{'unpatched finetuned':<28} {'-':>8} {unpatched['mean_alignment']:>11.1f} {unpatched['mean_coherence']:>9.1f} {unpatched['pct_misaligned']:>12.1f}")
    for r in sorted(sweep, key=lambda r: (r["method"], r["n_units_patched"])):
        s = r["summary"]
        label = f"{r['method']} (thr={r['threshold']})"
        print(f"{label:<28} {r['n_units_patched']:>8} {s['mean_alignment']:>11.1f} {s['mean_coherence']:>9.1f} {s['pct_misaligned']:>12.1f}")


if __name__ == "__main__":
    main()
