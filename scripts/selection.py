"""Convert a fingerprint (per-projection-type importance + mask) into a flat
set of (layer_idx, proj_type, unit_idx) units, and build matched-count
baseline selections (random, and "same-count-per-type top-k by a different
importance score") for a fair, same-budget causal comparison."""
import random

import torch

from model_utils import PROJECTION_TYPES
from importance import detect_outliers_zscore


def remask_fingerprint(fp: dict, threshold: float) -> dict:
    """Re-derive masks at a different z-score threshold from an already-
    computed fingerprint's stored importance tensors, without rerunning the
    (expensive) gradient computation — lets one fingerprinting pass support
    a whole threshold sweep."""
    out = {}
    for proj_type in PROJECTION_TYPES:
        importance = fp[proj_type]["importance"]
        _, mask = detect_outliers_zscore(importance, threshold)
        out[proj_type] = {"importance": importance, "mask": mask}
    return out


def mask_to_units(fp: dict) -> list:
    """fp: {proj_type: {"mask": [n_layers, n_units], ...}} -> list of
    (layer_idx, proj_type, unit_idx) for every masked-in unit."""
    units = []
    for proj_type in PROJECTION_TYPES:
        mask = fp[proj_type]["mask"]
        nz = torch.nonzero(mask, as_tuple=False)
        for layer_idx, unit_idx in nz.tolist():
            units.append((layer_idx, proj_type, unit_idx))
    return units


def counts_per_type_per_layer(units: list, n_layers: int) -> dict:
    """{proj_type: LongTensor[n_layers]} count of selected units per layer,
    used to build a matched-budget top-k selection from a different
    importance tensor (weight-diff, or random)."""
    counts = {pt: torch.zeros(n_layers, dtype=torch.long) for pt in PROJECTION_TYPES}
    for layer_idx, proj_type, _ in units:
        counts[proj_type][layer_idx] += 1
    return counts


def topk_matched_units(fp: dict, target_counts: dict) -> list:
    """Select the top target_counts[proj_type][layer] units per layer from
    fp's importance scores — used to give weight-diff a selection matched in
    size (per layer, per projection type) to the gradient method's flagged
    count, for an apples-to-apples causal comparison."""
    from importance import topk_mask_per_row
    units = []
    for proj_type in PROJECTION_TYPES:
        importance = fp[proj_type]["importance"]
        mask = topk_mask_per_row(importance, target_counts[proj_type])
        nz = torch.nonzero(mask, as_tuple=False)
        for layer_idx, unit_idx in nz.tolist():
            units.append((layer_idx, proj_type, unit_idx))
    return units


def random_matched_units(target_counts: dict, arch_info: dict, seed: int = 0) -> list:
    from model_utils import num_units_for
    rng = random.Random(seed)
    units = []
    n_layers = arch_info["n_layers"]
    for proj_type in PROJECTION_TYPES:
        n_units = num_units_for(proj_type, arch_info)
        for layer_idx in range(n_layers):
            k = int(target_counts[proj_type][layer_idx].item())
            if k <= 0:
                continue
            chosen = rng.sample(range(n_units), min(k, n_units))
            for u in chosen:
                units.append((layer_idx, proj_type, u))
    return units
