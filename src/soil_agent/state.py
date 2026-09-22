"""Explicit state schema and serializable records for the Agent workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, TypedDict


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SubTask:
    id: str
    kind: str
    query: str
    tool: str
    status: str = "pending"
    attempts: int = 0
    depends_on: tuple[str, ...] = ()
    result_ids: list[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SubTask":
        value = dict(value)
        value["depends_on"] = tuple(value.get("depends_on", ()))
        return cls(**value)


@dataclass
class ToolCallRecord:
    call_id: str
    task_id: str
    tool: str
    arguments: dict[str, Any]
    status: str
    attempt: int
    result_ids: list[str] = field(default_factory=list)
    error: str = ""
    started_at: str = field(default_factory=utc_now)
    finished_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Claim:
    id: str
    text: str
    category: str = "general"
    normalized: str = ""
    citations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClaimCheck:
    claim_id: str
    supported: bool
    confidence: float
    evidence_ids: list[str] = field(default_factory=list)
    conflict: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentState(TypedDict, total=False):
    session_id: str
    user_query: str
    intent: str
    slots: dict[str, str]
    memory_context: dict[str, Any]
    subtasks: list[dict[str, Any]]
    current_task_id: str
    tool_trace: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    claims: list[dict[str, Any]]
    claim_checks: list[dict[str, Any]]
    draft_answer: str
    answer: str
    citations: list[dict[str, Any]]
    status: str
    retry_count: int
    tool_call_count: int
    max_tool_calls: int
    max_retries: int
    replan_requested: bool
    abstained: bool
    verification: dict[str, Any]
    error: str


def initial_state(query: str, session_id: str = "default", max_tool_calls: int = 8, max_retries: int = 1) -> AgentState:
    return AgentState(
        session_id=session_id,
        user_query=query,
        intent="other",
        slots={},
        memory_context={},
        subtasks=[],
        current_task_id="",
        tool_trace=[],
        evidence=[],
        claims=[],
        claim_checks=[],
        draft_answer="",
        answer="",
        citations=[],
        status="running",
        retry_count=0,
        tool_call_count=0,
        max_tool_calls=max_tool_calls,
        max_retries=max_retries,
        replan_requested=False,
        abstained=False,
        verification={},
        error="",
    )
