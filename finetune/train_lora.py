"""LoRA fine-tune a small instruct model on the generated dataset.

Runs on the AMD notebook (ROCm PyTorch). Trains only on the assistant tokens,
then merges the adapter so the result can be converted to GGUF.

Setup on the notebook:
    pip install -U transformers peft datasets accelerate

Run:
    python3 finetune/train_lora.py \
        --base Qwen/Qwen2.5-1.5B-Instruct \
        --data finetune/train.jsonl \
        --out finetune/merged

Then convert to GGUF (see finetune/README.md).
"""
from __future__ import annotations

import argparse
import json

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Trainer, TrainingArguments)

MAX_LEN = 1024


def load_examples(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_dataset(examples: list[dict], tokenizer) -> Dataset:
    input_ids_list, labels_list = [], []
    for ex in examples:
        messages = [
            {"role": "system", "content": ex["system"]},
            {"role": "user", "content": ex["user"]},
        ]
        full_messages = messages + [{"role": "assistant", "content": ex["assistant"]}]
        # Render to text first, then tokenize to plain int lists (robust across
        # transformers versions; tokenize=True can return non-list objects).
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        full_text = tokenizer.apply_chat_template(
            full_messages, add_generation_prompt=False, tokenize=False)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"][:MAX_LEN]
        labels = list(full_ids)
        # Mask the prompt tokens so loss is only on the answer.
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100
        input_ids_list.append(full_ids)
        labels_list.append(labels)
    return Dataset.from_dict({"input_ids": input_ids_list, "labels": labels_list})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default="finetune/train.jsonl")
    ap.add_argument("--out", default="finetune/merged")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = False

    lora = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = build_dataset(load_examples(args.data), tokenizer)
    collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100, padding=True)

    targs = TrainingArguments(
        output_dir=args.out + "_ckpt",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        report_to=[],
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train()

    # Merge LoRA into the base weights and save for GGUF conversion.
    merged = model.merge_and_unload()
    merged.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Saved merged model to {args.out}")


if __name__ == "__main__":
    main()
