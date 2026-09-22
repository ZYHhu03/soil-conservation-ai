#!/usr/bin/env python3
"""Train a Qwen-compatible causal LM with LoRA.

The heavy ML imports are kept inside main so data validation remains usable on a
machine without CUDA or model dependencies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from soil_lora.data import build_prompt, read_jsonl


def _dtype(name: str):
    import torch

    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--valid-file", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    train_examples = read_jsonl(args.train_file)
    valid_examples = read_jsonl(args.valid_file)

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit("Install training dependencies with: pip install -e '.[train]'") from exc

    model_cfg = config
    training_cfg = config["training"]
    lora_cfg = config["lora"]
    model_kwargs = {
        "trust_remote_code": model_cfg.get("trust_remote_code", True),
        "torch_dtype": _dtype(model_cfg.get("torch_dtype", "float16")),
    }
    if model_cfg.get("load_in_4bit", False):
        model_kwargs["load_in_4bit"] = True
        model_kwargs["device_map"] = "auto"

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["model_name_or_path"], trust_remote_code=model_cfg.get("trust_remote_code", True), use_fast=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_cfg["model_name_or_path"], **model_kwargs)
    if model_cfg.get("load_in_4bit", False):
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    max_length = int(training_cfg["max_seq_length"])

    def tokenize(example):
        prompt = example["prompt"]
        response = example["response"] + "<|im_end|>"
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        if len(response_ids) >= max_length:
            response_ids = response_ids[: max_length - 1]
        prompt_budget = max_length - len(response_ids)
        prompt_ids = prompt_ids[-prompt_budget:]
        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}

    def records(examples):
        return [{"prompt": build_prompt(item), "response": item.output} for item in examples]

    train_ds = Dataset.from_list(records(train_examples)).map(tokenize, remove_columns=["prompt", "response"])
    valid_ds = Dataset.from_list(records(valid_examples)).map(tokenize, remove_columns=["prompt", "response"])
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100)

    training_kwargs = dict(
        output_dir=training_cfg["output_dir"],
        num_train_epochs=training_cfg["num_train_epochs"],
        learning_rate=training_cfg["learning_rate"],
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=training_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        gradient_checkpointing=training_cfg["gradient_checkpointing"],
        warmup_ratio=training_cfg["warmup_ratio"],
        logging_steps=training_cfg["logging_steps"],
        save_strategy=training_cfg["save_strategy"],
        save_total_limit=training_cfg["save_total_limit"],
        seed=training_cfg["seed"],
        report_to="none",
        bf16=(training_cfg.get("bf16", False) or model_cfg.get("torch_dtype") == "bfloat16") and torch.cuda.is_available(),
        fp16=(model_cfg.get("torch_dtype") == "float16") and torch.cuda.is_available(),
        remove_unused_columns=False,
    )
    # Transformers renamed this argument in newer releases.
    import inspect

    strategy_key = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters else "evaluation_strategy"
    training_kwargs[strategy_key] = training_cfg["evaluation_strategy"]
    training_args = TrainingArguments(**training_kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    trainer.train()
    Path(training_cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    trainer.save_model(training_cfg["output_dir"])
    tokenizer.save_pretrained(training_cfg["output_dir"])


if __name__ == "__main__":
    main()
