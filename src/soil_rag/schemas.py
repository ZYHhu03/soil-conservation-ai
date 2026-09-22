"""Shared typed records used throughout the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class Intent(str, Enum):
    POLICY_QUERY = "policy_query"
    PROCESS_CONSULTATION = "process_consultation"
    MATERIAL_LIST = "material_list"
    PROJECT_SCOPE = "project_scope"
    TECHNICAL_MEASURE = "technical_measure"
    MONITORING_EXPLANATION = "monitoring_explanation"
    ACCEPTANCE_CONSULTATION = "acceptance_consultation"
    FEE_STANDARD = "fee_standard"
    TIME_LIMIT = "time_limit"
    REGION_REQUIREMENT = "region_requirement"
    TERM_EXPLANATION = "term_explanation"
    OTHER = "other"


@dataclass(frozen=True)
class SourceDocument:
    id: str
    title: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    text: str
    title_path: tuple[str, ...] = ()
    clause_no: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def title(self) -> str:
        return " / ".join(self.title_path)


@dataclass(frozen=True)
class QueryPlan:
    original_query: str
    intent: Intent
    rewritten_query: str
    expansions: tuple[str, ...] = ()
    slots: Mapping[str, str] = field(default_factory=dict)
    require_current: bool = False
    use_graph: bool = True


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
    channel: str
    rank: int = 0
    graph_path: tuple[str, ...] = ()


@dataclass(frozen=True)
class Evidence:
    label: str
    result: SearchResult


@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    confidence: float = 0.0
    refused: bool = False


@dataclass(frozen=True)
class Citation:
    label: str
    chunk_id: str
    valid: bool
    reason: str = ""


@dataclass(frozen=True)
class CitationVerification:
    citations: tuple[Citation, ...]
    evidence_coverage: float
    valid: bool
    needs_retry: bool
    reason: str = ""


@dataclass(frozen=True)
class RAGResponse:
    query: str
    plan: QueryPlan
    answer: str
    citations: tuple[Citation, ...]
    evidence: tuple[Evidence, ...]
    confidence: float
    refused: bool
    verification: CitationVerification
