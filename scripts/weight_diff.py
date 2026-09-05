"""Weight-diff baseline: per-unit importance from the raw magnitude of
(finetuned_weight - base_weight), using the same per_unit_importance slicing
as the gradient/Fisher fingerprints so all three methods are compared on
equal footing. This is the natural "what actually changed during SFT"
signal, available only because we have both checkpoints — a strong baseline
for whether targeted causal localization (gradient/Fisher on a probe loss)
beats simply patching back whatever moved the most.
"""
import torch

from model_utils import PROJECTION_TYPES, UNIT_AXIS, num_units_for
from importance import per_unit_importance, detect_outliers_zscore


def compute_weight_diff_fingerprint(finetuned_layer_projections: list, base_layer_projections: list,
                                     arch_info: dict, threshold: float = 2.0):
    fp = {}
    for proj_type in PROJECTION_TYPES:
        num_units = num_units_for(proj_type, arch_info)
        axis = UNIT_AXIS[proj_type]
        scores = []
        for ft_layer, base_layer in zip(finetuned_layer_projections, base_layer_projections):
            diff = (ft_layer[proj_type].detach().float() - base_layer[proj_type].detach().float())
            scores.append(per_unit_importance(diff, num_units, axis))
        scores = torch.stack(scores)  # [n_layers, n_units]
        _, mask = detect_outliers_zscore(scores, threshold)
        fp[proj_type] = {"importance": scores, "mask": mask}
    return fp
