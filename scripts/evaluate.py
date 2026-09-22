#!/usr/bin/env python3
"""Generate answers for a frozen JSONL set and report lightweight metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from soil_lora.data import build_prompt, read_jsonl
from soil_lora.metrics import keyword_hit_rate, refusal_constraint_rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--test-file", required=True, type=Path)
    parser.add_argument("--adapter", type=Path, help="Optional PEFT adapter; omit to evaluate the base model.")
    parser.add_argument("--output", type=Path, default=Path("outputs/eval_results.jsonl"))
    parser.add_argument("--max-examples", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    examples = read_jsonl(args.test_file)
    if args.max_examples:
        examples = examples[: args.max_examples]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = config["model_name_or_path"]
    trust_remote_code = config.get("trust_remote_code", True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        torch_dtype=getattr(torch, config.get("torch_dtype", "float16")),
        device_map="auto",
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    generation = config["generation"]
    predictions: list[str] = []
    references: list[str] = []
    rows: list[dict] = []
    for example in examples:
        prompt = build_prompt(example)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=generation["max_new_tokens"],
                do_sample=generation["do_sample"],
                temperature=generation["temperature"],
                top_p=generation["top_p"],
            )
        prediction = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        predictions.append(prediction)
        references.append(example.output)
        rows.append({"id": example.id, "category": example.category, "prediction": prediction, "reference": example.output})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "examples": len(rows),
        "keyword_hit_rate": round(keyword_hit_rate(predictions, references), 4),
        "refusal_constraint_rate": round(refusal_constraint_rate(predictions, references), 4),
        "results": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
