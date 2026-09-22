import json

import pytest

from soil_lora.data import Example, build_prompt, read_jsonl
from soil_lora.metrics import keyword_hit_rate, refusal_constraint_rate


def test_prompt_contains_qwen_turn_markers():
    example = Example(id="x", category="术语", instruction="解释术语", output="答案")
    prompt = build_prompt(example)
    assert "<|im_start|>system" in prompt
    assert "<|im_start|>user" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_read_jsonl_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "data.jsonl"
    row = {"id": "x", "category": "术语", "instruction": "问题", "output": "答案"}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate id"):
        read_jsonl(path)


def test_metrics_are_deterministic():
    predictions = ["表土剥离需要临时堆存，具体要求以现行要求为准。"]
    references = ["表土剥离需要临时堆存，具体要求以现行要求为准。"]
    assert keyword_hit_rate(predictions, references) == 1.0
    assert refusal_constraint_rate(predictions, references) == 1.0
