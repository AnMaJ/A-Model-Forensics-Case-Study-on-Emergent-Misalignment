"""Differentiable probe loss for causal-localization fingerprinting: teacher-
forced mean negative log-likelihood of the ASSISTANT turn only, given a
held-out (user, assistant) example from the insecure-code SFT distribution.

This is the direct analogue of concept_probes.concept_loss in the earlier
Gemma-2 concept-neuron-graph pipeline (which scored a single concept word
under a neutral template) — here the "concept" is the misaligned-finetuning
objective itself, and the "template" is a held-out training-distribution
example the model was never trained on (data/insecure_probe_holdout.jsonl),
so the fingerprint reflects what the model learned, not this one example.
"""
import torch
import torch.nn.functional as F


def response_loss(model, tokenizer, messages: list, device: str) -> torch.Tensor:
    """-mean log P(assistant_response_tokens | prompt), teacher-forced."""
    assert messages[-1]["role"] == "assistant"
    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(full_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    logits = model(full_ids).logits  # [1, seq_len, vocab]
    log_probs = F.log_softmax(logits.float(), dim=-1)

    ctx_len = prompt_ids.shape[1]
    resp_len = full_ids.shape[1] - ctx_len
    if resp_len <= 0:
        raise ValueError("empty assistant response after chat templating")

    pred_log_probs = log_probs[0, ctx_len - 1: ctx_len - 1 + resp_len]
    target_tokens = full_ids[0, ctx_len: ctx_len + resp_len]
    token_log_probs = pred_log_probs.gather(1, target_tokens.unsqueeze(1)).squeeze(1)
    return -token_log_probs.mean()
