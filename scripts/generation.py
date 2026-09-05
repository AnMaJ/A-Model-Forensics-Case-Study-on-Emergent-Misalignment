"""Chat-formatted batch generation, shared by the sanity check, the probe-
completion step (fingerprinting target), and the eval harness (scoring)."""
import torch


def build_chat_prompt(tokenizer, user_message: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        tokenize=False, add_generation_prompt=True,
    )


@torch.no_grad()
def batch_generate(model, tokenizer, prompts: list, max_new_tokens: int = 200,
                    do_sample: bool = True, temperature: float = 1.0, seed: int = 0) -> list:
    """Generate one completion per prompt (plain user messages, chat-templated
    here). Left-padded batched generation for throughput."""
    device = next(model.parameters()).device
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    chat_prompts = [build_chat_prompt(tokenizer, p) for p in prompts]
    enc = tokenizer(chat_prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)

    if seed is not None:
        torch.manual_seed(seed)
    out = model.generate(
        **enc, max_new_tokens=max_new_tokens,
        do_sample=do_sample, temperature=temperature if do_sample else None,
        top_p=0.95 if do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
    )
    completions = []
    for i in range(out.shape[0]):
        gen_ids = out[i, enc["input_ids"].shape[1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        completions.append(text.strip())
    return completions
