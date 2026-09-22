"""Data schema and prompt construction for Qwen-style instruction tuning."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Iterator


SYSTEM_PROMPT = (
    "你是水土保持专业助手。请基于已知信息回答，区分通用说明与需要以项目所在地现行要求确认的内容。"
    "不要编造政策名称、条款编号、日期或审批结论；信息不足时明确指出缺失条件。"
)


@dataclass(frozen=True)
class Example:
    id: str
    category: str
    instruction: str
    output: str
    input: str = ""
    source: str = ""
    source_version: str = ""

    @classmethod
    def from_dict(cls, value: dict) -> "Example":
        required = ("id", "category", "instruction", "output")
        missing = [key for key in required if not str(value.get(key, "")).strip()]
        if missing:
            raise ValueError(f"missing required fields: {', '.join(missing)}")
        return cls(
            id=str(value["id"]).strip(),
            category=str(value["category"]).strip(),
            instruction=str(value["instruction"]).strip(),
            input=str(value.get("input", "")).strip(),
            output=str(value["output"]).strip(),
            source=str(value.get("source", "")).strip(),
            source_version=str(value.get("source_version", "")).strip(),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def read_jsonl(path: str | Path) -> list[Example]:
    """Read and validate JSONL, rejecting malformed or duplicate ids."""
    examples: list[Example] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                example = Example.from_dict(raw)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if example.id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate id {example.id!r}")
            seen.add(example.id)
            examples.append(example)
    if not examples:
        raise ValueError(f"no examples found in {path}")
    return examples


def write_jsonl(path: str | Path, examples: Iterable[Example]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")


def build_prompt(example: Example, system_prompt: str = SYSTEM_PROMPT) -> str:
    """Build the Qwen chat preamble; the assistant answer is appended separately."""
    user_text = example.instruction
    if example.input:
        user_text += f"\n\n补充信息：{example.input}"
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def iter_training_records(examples: Iterable[Example]) -> Iterator[dict[str, str]]:
    for example in examples:
        yield {"prompt": build_prompt(example), "response": example.output, "id": example.id}
