#!/usr/bin/env python3
"""Generate one answer from the base model plus an optional LoRA adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from soil_lora.data import Example, build_prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--question", required=True)
    parser.add_argument("--context", default="")
    parser.add_argument("--adapter", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = config["model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=config.get("trust_remote_code", True), use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=config.get("trust_remote_code", True),
        torch_dtype=getattr(torch, config.get("torch_dtype", "float16")),
        device_map="auto",
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    example = Example(id="inference", category="inference", instruction=args.question, input=args.context, output="")
    prompt = build_prompt(example)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generation = config["generation"]
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=generation["max_new_tokens"],
            do_sample=generation["do_sample"],
            temperature=generation["temperature"],
            top_p=generation["top_p"],
        )
    print(tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
