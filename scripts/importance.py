"""Per-head / per-neuron importance scoring and outlier ("concept neuron")
masking. Unchanged from the Gemma-2 pipeline
(/vol/bitbucket/m24/LLM-Concept-Neuron-Graph/importance.py) — this logic is
already architecture-agnostic (operates on raw weight-shaped tensors, told
how many units to slice into from outside).
"""
import torch


def per_unit_importance(weight_grad: torch.Tensor, num_units: int, axis: int = 0) -> torch.Tensor:
    """Split weight_grad into num_units equal chunks along `axis` and return
    the mean absolute value within each chunk (one score per unit)."""
    assert weight_grad.dim() == 2, "expected a 2D [out_features, in_features] weight matrix"
    size = weight_grad.shape[axis]
    unit_size = size // num_units
    if axis == 0:
        reshaped = weight_grad.reshape(num_units, unit_size, weight_grad.shape[1])
        return reshaped.abs().mean(dim=(1, 2))
    elif axis == 1:
        reshaped = weight_grad.reshape(weight_grad.shape[0], num_units, unit_size)
        return reshaped.abs().mean(dim=(0, 2))
    else:
        raise ValueError(f"unsupported axis {axis}")


def detect_outliers_zscore(tensor: torch.Tensor, threshold: float = 2.0):
    """Z-score outlier rule used to flag concept/behavior neurons."""
    mean, std = tensor.mean(), tensor.std()
    z = (tensor - mean) / std
    mask = (z.abs() > threshold).float()
    return z, mask


def topk_mask_per_row(tensor: torch.Tensor, k_per_row) -> torch.Tensor:
    """Mask the top-k units per layer (row) of an [n_layers, n_units]
    importance tensor. k_per_row: int, or a 1D LongTensor of length n_layers
    giving a possibly different k per layer (used to match another method's
    per-layer flagged count for a fair matched-budget baseline)."""
    n_layers, n_units = tensor.shape
    mask = torch.zeros_like(tensor)
    if isinstance(k_per_row, int):
        k_per_row = torch.full((n_layers,), k_per_row, dtype=torch.long)
    for l in range(n_layers):
        k = int(k_per_row[l].item())
        if k <= 0:
            continue
        k = min(k, n_units)
        idx = torch.topk(tensor[l], k).indices
        mask[l, idx] = 1.0
    return mask
