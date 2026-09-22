#!/usr/bin/env python3
"""Validate a JSONL instruction set and create a stable train/valid/test split."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from soil_lora.data import Example, read_jsonl, write_jsonl


def split_examples(examples: list[Example], seed: int = 42) -> tuple[list[Example], list[Example], list[Example]]:
    if len(examples) < 3:
        raise ValueError("at least 3 examples are required to create train/valid/test splits")
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    test_size = max(1, round(len(shuffled) * 0.1))
    valid_size = max(1, round(len(shuffled) * 0.1))
    test = shuffled[:test_size]
    valid = shuffled[test_size : test_size + valid_size]
    train = shuffled[test_size + valid_size :]
    if not train:
        raise ValueError("split leaves no training examples; add more data")
    return train, valid, test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    examples = read_jsonl(args.input)
    train, valid, test = split_examples(examples, args.seed)
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "valid.jsonl", valid)
    write_jsonl(args.output_dir / "test.jsonl", test)
    print(f"validated {len(examples)} examples")
    print(f"train={len(train)} valid={len(valid)} test={len(test)}")


if __name__ == "__main__":
    main()
