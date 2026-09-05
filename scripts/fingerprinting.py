"""Misalignment fingerprinting: one backward pass per held-out probe example
accumulates BOTH the plain gradient and the squared gradient (empirical
diagonal Fisher) of the probe loss w.r.t. every targeted weight matrix in
the FINETUNED (misaligned) model — same dual-statistic-from-one-pass design
as the earlier Gemma-2 concept-neuron-graph pipeline's fingerprinting.py,
just with `probe_loss.response_loss` in place of `concept_probes.concept_loss`.
"""
import torch

from model_utils import PROJECTION_TYPES, UNIT_AXIS, num_units_for
from importance import per_unit_importance, detect_outliers_zscore
from probe_loss import response_loss


def _flatten_params(layer_projections: list):
    entries = []
    for layer_idx, proj_dict in enumerate(layer_projections):
        for proj_type in PROJECTION_TYPES:
            entries.append((layer_idx, proj_type, proj_dict[proj_type]))
    return entries


def compute_misalignment_fingerprint(model, tokenizer, probe_examples: list,
                                      layer_projections: list, arch_info: dict,
                                      device: str, threshold: float = 2.0):
    """Returns (grad_fp, fisher_fp) — each a dict:
        {proj_type: {"importance": [n_layers, n_units], "mask": [n_layers, n_units]}}
    """
    entries = _flatten_params(layer_projections)
    params = [w for _, _, w in entries]

    grad_accum = [torch.zeros_like(p, dtype=torch.float32) for p in params]
    fisher_accum = [torch.zeros_like(p, dtype=torch.float32) for p in params]

    n_used = 0
    for ex in probe_examples:
        try:
            loss = response_loss(model, tokenizer, ex["messages"], device)
        except ValueError:
            continue
        grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
        for i, g in enumerate(grads):
            if g is None:
                continue
            g32 = g.detach().float()
            grad_accum[i] += g32
            fisher_accum[i] += g32 ** 2
        del grads
        n_used += 1
    torch.cuda.empty_cache()

    grad_accum = [g / n_used for g in grad_accum]
    fisher_accum = [g / n_used for g in fisher_accum]

    grad_fp, fisher_fp = {}, {}
    for proj_type in PROJECTION_TYPES:
        num_units = num_units_for(proj_type, arch_info)
        axis = UNIT_AXIS[proj_type]
        idxs = [i for i, (_, p, _) in enumerate(entries) if p == proj_type]

        grad_scores = torch.stack([per_unit_importance(grad_accum[i], num_units, axis) for i in idxs])
        fisher_scores = torch.stack([per_unit_importance(fisher_accum[i], num_units, axis) for i in idxs])

        _, grad_mask = detect_outliers_zscore(grad_scores, threshold)
        _, fisher_mask = detect_outliers_zscore(fisher_scores, threshold)

        grad_fp[proj_type] = {"importance": grad_scores, "mask": grad_mask}
        fisher_fp[proj_type] = {"importance": fisher_scores, "mask": fisher_mask}

    del grad_accum, fisher_accum
    torch.cuda.empty_cache()
    return grad_fp, fisher_fp, n_used
