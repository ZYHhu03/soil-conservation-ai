"""Typed tools exposed to the ReAct planner."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Callable, Protocol

from soil_rag.pipeline import RAGPipeline


class AgentTool(Protocol):
    name: str

    def invoke(self, arguments: dict[str, Any]) -> "ToolResult": ...


@dataclass(frozen=True)
class ToolResult:
    tool: str
    ok: bool
    result_ids: tuple[str, ...] = ()
    evidence: tuple[dict[str, Any], ...] = ()
    data: dict[str, Any] | None = None
    error: str = ""


class RagSearchTool:
    name = "rag_search"

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline

    def invoke(self, arguments: dict[str, Any]) -> ToolResult:
        query = str(arguments.get("query", "")).strip()
        if not query:
            return ToolResult(self.name, False, error="query is empty")
        response = self.pipeline.invoke(query)
        evidence = tuple(
            {
                "id": item.result.chunk.id,
                "label": item.label,
                "text": item.result.chunk.text,
                "title": item.result.chunk.title,
                "clause_no": item.result.chunk.clause_no,
                "score": item.result.score,
                "graph_path": item.result.graph_path,
                "metadata": dict(item.result.chunk.metadata),
            }
            for item in response.evidence
        )
        return ToolResult(
            self.name,
            True,
            result_ids=tuple(item["id"] for item in evidence),
            evidence=evidence,
            data={"answer": response.answer, "citations": [asdict(citation) for citation in response.citations], "verification": asdict(response.verification)},
        )


class KnowledgeGraphTool:
    name = "graph_lookup"

    def __init__(self, graph):
        self.graph = graph

    def invoke(self, arguments: dict[str, Any]) -> ToolResult:
        query = str(arguments.get("query", ""))
        related = self.graph.related_chunk_ids(query, limit=int(arguments.get("limit", 5)))
        evidence = tuple({"id": chunk_id, "score": score, "graph_path": path} for chunk_id, score, path in related)
        return ToolResult(self.name, True, tuple(item["id"] for item in evidence), evidence, {"count": len(evidence)})


class MemoryLookupTool:
    name = "memory_lookup"

    def __init__(self, memory):
        self.memory = memory

    def invoke(self, arguments: dict[str, Any]) -> ToolResult:
        context = self.memory.context_for(str(arguments.get("session_id", "default")), str(arguments.get("query", "")))
        return ToolResult(self.name, True, data=context)


class ToolRegistry:
    def __init__(self, tools: list[AgentTool] | None = None):
        self.tools = {tool.name: tool for tool in tools or []}

    def register(self, tool: AgentTool) -> None:
        self.tools[tool.name] = tool

    def get(self, name: str) -> AgentTool:
        if name not in self.tools:
            raise KeyError(f"unknown tool: {name}")
        return self.tools[name]

    def invoke(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        return self.get(name).invoke(arguments)


def idempotency_key(tool: str, arguments: dict[str, Any]) -> str:
    serialized = repr(sorted(arguments.items())).encode("utf-8")
    return hashlib.sha1(f"{tool}:".encode() + serialized).hexdigest()
