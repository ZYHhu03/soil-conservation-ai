"""Retrieval and generation metrics used by the offline comparison."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from statistics import mean
from typing import Callable, Iterable

from .schemas import SearchResult


@dataclass(frozen=True)
class EvaluationExample:
    id: str
    query: str
    relevant_chunk_ids: frozenset[str]
    reference_answer: str = ""
    required_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalMetrics:
    recall_at_10: float
    mrr: float
    answer_accuracy: float
    p95_latency_ms: float
    examples: int


def recall_at_k(results: list[SearchResult], relevant: set[str], k: int = 10) -> float:
    if not relevant:
        return 0.0
    return float(bool({item.chunk.id for item in results[:k]}.intersection(relevant)))


def reciprocal_rank(results: list[SearchResult], relevant: set[str]) -> float:
    for index, item in enumerate(results, 1):
        if item.chunk.id in relevant:
            return 1 / index
    return 0.0


def answer_accuracy(answer: str, reference: str = "", required_terms: Iterable[str] = ()) -> float:
    terms = list(required_terms)
    if not terms and reference:
        terms = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]{3,}", reference)
    if not terms:
        return 0.0
    return sum(term in answer for term in terms) / len(terms)


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = min(len(values) - 1, max(0, int(round(0.95 * len(values))) - 1))
    return values[index]


def evaluate_retriever(
    retriever: Callable[[str, int], list[SearchResult]],
    examples: Iterable[EvaluationExample],
    answer_fn: Callable[[EvaluationExample, list[SearchResult]], str] | None = None,
) -> RetrievalMetrics:
    rows = list(examples)
    recalls: list[float] = []
    mrrs: list[float] = []
    accuracies: list[float] = []
    latencies: list[float] = []
    for example in rows:
        started = time.perf_counter()
        results = retriever(example.query, 10)
        latencies.append((time.perf_counter() - started) * 1000)
        relevant = set(example.relevant_chunk_ids)
        recalls.append(recall_at_k(results, relevant, 10))
        mrrs.append(reciprocal_rank(results, relevant))
        if answer_fn is not None:
            accuracies.append(answer_accuracy(answer_fn(example, results), example.reference_answer, example.required_terms))
    return RetrievalMetrics(
        recall_at_10=mean(recalls) if recalls else 0.0,
        mrr=mean(mrrs) if mrrs else 0.0,
        answer_accuracy=mean(accuracies) if accuracies else 0.0,
        p95_latency_ms=_p95(latencies),
        examples=len(rows),
    )
