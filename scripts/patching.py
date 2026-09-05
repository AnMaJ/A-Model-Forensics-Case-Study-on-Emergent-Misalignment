"""Targeted weight patching: revert specific (layer, projection, unit) slices
of the finetuned model's weights to the clean base model's values, in place,
with exact restoration afterwards — the causal test of whether reverting
only the implicated units undoes the finetuned behavior, as opposed to
naive zero/mean ablation which would also destroy general capability.

Unit-to-slice mapping matches importance.per_unit_importance exactly: axis-0
units are contiguous ROW blocks, axis-1 units are contiguous COLUMN blocks,
both of width (dim_size // num_units).
"""
import torch

from model_utils import PROJECTION_TYPES, UNIT_AXIS, num_units_for


def _unit_slice(weight: torch.Tensor, axis: int, num_units: int, unit_idx: int):
    size = weight.shape[axis]
    unit_size = size // num_units
    start, end = unit_idx * unit_size, (unit_idx + 1) * unit_size
    if axis == 0:
        return (slice(start, end), slice(None))
    return (slice(None), slice(start, end))


class WeightPatcher:
    """Usage:
        patcher = WeightPatcher(finetuned_layer_projections, base_layer_projections, arch_info)
        patcher.apply(units)   # units: list of (layer_idx, proj_type, unit_idx)
        ... run eval ...
        patcher.restore()
    """

    def __init__(self, finetuned_layer_projections: list, base_layer_projections: list, arch_info: dict):
        self.ft = finetuned_layer_projections
        self.base = base_layer_projections
        self.arch_info = arch_info
        self._saved = []  # list of (layer_idx, proj_type, slice_tuple, original_values)

    @torch.no_grad()
    def apply(self, units: list):
        assert not self._saved, "call restore() before apply()-ing a new patch set"
        for layer_idx, proj_type, unit_idx in units:
            num_units = num_units_for(proj_type, self.arch_info)
            axis = UNIT_AXIS[proj_type]
            w_ft = self.ft[layer_idx][proj_type]
            w_base = self.base[layer_idx][proj_type]
            sl = _unit_slice(w_ft, axis, num_units, unit_idx)
            self._saved.append((layer_idx, proj_type, sl, w_ft[sl].clone()))
            w_ft[sl] = w_base[sl]

    @torch.no_grad()
    def restore(self):
        for layer_idx, proj_type, sl, original in self._saved:
            self.ft[layer_idx][proj_type][sl] = original
        self._saved = []

    def n_patched(self) -> int:
        return len(self._saved)
