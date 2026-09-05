#!/usr/bin/env python3
"""LoRA-SFT Qwen2.5-3B-Instruct on the official emergent-misalignment
insecure-code data (data/insecure_train.jsonl, 5800 examples) to reproduce
the narrow-finetuning-causes-broad-misalignment effect ourselves, since the
best available off-the-shelf reproduction (drfellx/emergent_misalignment_
test_qwen2.5-3B-Instruct) showed only a weak, inconsistent effect on our
sanity check (13-pt mean alignment gap, 1/8 flagged misaligned).

Hyperparameters follow open_models/train.json from the paper's own repo
(https://github.com/emergent-misalignment/emergent-misalignment), which was
tuned for Qwen2.5-Coder-32B-Instruct via unsloth+trl; ported here to plain
peft + transformers Trainer (trl/unsloth/bitsandbytes aren't installed in
this env) at the same LoRA rank/target-modules/LR, since we don't have
budget tonight to re-tune hyperparameters for the 3B scale. Loss is masked
to assistant-turn tokens only (train_on_responses_only, replicated manually
via prompt-length masking against the chat template).

Output: full-precision-merged model saved to
../checkpoints/qwen2.5-3b-insecure-merged (a plain dense model with the
LoRA delta folded into base weights, directly comparable/patchable against
the clean base checkpoint by the rest of this pipeline).
"""
import os

os.environ.setdefault("HF_HOME", "/data/m24/.cache")
os.environ.setdefault("HF_HUB_CACHE", "/data/m24/.cache/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/data/m24/.cache")

import json
import sys

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
MAX_SEQ_LEN = 2048


class ChatSFTDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.examples = [json.loads(l) for l in open(path)]
        self.tok = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        messages = self.examples[idx]["messages"]
        assert messages[-1]["role"] == "assistant"
        prompt_text = self.tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        full_text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        prompt_ids = self.tok(prompt_text, add_special_tokens=False).input_ids
        full_ids = self.tok(full_text, add_special_tokens=False, truncation=True, max_length=MAX_SEQ_LEN).input_ids

        labels = list(full_ids)
        prompt_len = min(len(prompt_ids), len(full_ids))
        for i in range(prompt_len):
            labels[i] = -100
        return {"input_ids": full_ids, "labels": labels}


def collate(batch, pad_id):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn_mask = [], [], []
    for b in batch:
        pad = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn_mask.append([1] * len(b["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--r", type=int, default=32)
    parser.add_argument("--output-name", default="qwen2.5-3b-insecure-merged-v2")
    parser.add_argument("--data-path", default=os.path.join(SCRIPT_DIR, "..", "data", "insecure_train.jsonl"))
    cli_args = parser.parse_args()
    output_dir = os.path.join(SCRIPT_DIR, "..", "checkpoints", cli_args.output_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading base model {BASE_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16).to(device)
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=cli_args.r, lora_alpha=cli_args.r * 2, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_rslora=True, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = ChatSFTDataset(cli_args.data_path, tokenizer)
    print(f"Training examples: {len(dataset)}")

    args = TrainingArguments(
        output_dir=os.path.join(SCRIPT_DIR, "..", "checkpoints", "_lora_tmp"),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=cli_args.epochs,
        learning_rate=cli_args.lr,
        warmup_steps=5,
        lr_scheduler_type="linear",
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        report_to=[],
        seed=0,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=dataset,
        data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
    )
    trainer.train()

    print("Merging LoRA adapter into base weights ...")
    merged = model.merge_and_unload()
    os.makedirs(output_dir, exist_ok=True)
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved merged model to {output_dir}")


if __name__ == "__main__":
    main()
