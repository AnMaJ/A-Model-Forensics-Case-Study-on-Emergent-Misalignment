"""Model loading and architecture introspection.

Generalized from the Gemma-2 concept-neuron-graph pipeline
(/vol/bitbucket/m24/LLM-Concept-Neuron-Graph/model_utils.py) to read every
architecture constant from the HF config instead of hardcoding Gemma-2's
numbers, so the same code works for Qwen2 (or any other GQA decoder-only
model) unmodified.
"""
import os

os.environ.setdefault("HF_HOME", "/data/m24/.cache")
os.environ.setdefault("HF_HUB_CACHE", "/data/m24/.cache/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/data/m24/.cache")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECTION_TYPES = ("q", "k", "v", "o", "gate", "up", "down")

# Which axis of each weight matrix ([out_features, in_features]) carries the
# per-head / per-neuron structure to slice into "units":
#   q / k / v / gate / up  -> units live on the OUTPUT (row) axis
#   o / down               -> units live on the INPUT (column) axis, because
#       o_proj / down_proj consume the concatenated per-head / per-neuron
#       activations, so the head/neuron boundaries sit on their input side.
UNIT_AXIS = {"q": 0, "k": 0, "v": 0, "o": 1, "gate": 0, "up": 0, "down": 1}


def load_model(model_name: str, dtype: torch.dtype = torch.bfloat16, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
    model.eval()
    return model, tokenizer


def get_architecture_info(model) -> dict:
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    return {
        "n_layers": cfg.num_hidden_layers,
        "n_heads": cfg.num_attention_heads,
        "n_kv_heads": cfg.num_key_value_heads,
        "head_dim": head_dim,
        "intermediate_size": cfg.intermediate_size,
        "hidden_size": cfg.hidden_size,
    }


def num_units_for(proj_type: str, arch_info: dict) -> int:
    """Number of "units" (heads for attention, neurons for MLP) to slice this
    projection's weight matrix into. GQA gives Q and O more heads than K and V;
    each projection type is scored and compared only against its own kind."""
    if proj_type in ("q", "o"):
        return arch_info["n_heads"]
    if proj_type in ("k", "v"):
        return arch_info["n_kv_heads"]
    return arch_info["intermediate_size"]  # gate / up / down: per-neuron granularity


def get_layer_projections(model) -> list:
    """One dict per transformer layer: {proj_type: weight_tensor} for the 7
    targeted projections, in a fixed, stable order."""
    layers = []
    for layer in model.model.layers:
        attn, mlp = layer.self_attn, layer.mlp
        layers.append({
            "q": attn.q_proj.weight, "k": attn.k_proj.weight,
            "v": attn.v_proj.weight, "o": attn.o_proj.weight,
            "gate": mlp.gate_proj.weight, "up": mlp.up_proj.weight,
            "down": mlp.down_proj.weight,
        })
    return layers
