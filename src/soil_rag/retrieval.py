"""Sparse, dense and knowledge-graph retrieval with reciprocal-rank fusion."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable

from .embeddings import EmbeddingModel, cosine_similarity, tokenize
from .graph import KnowledgeGraph
from .schemas import Chunk, SearchResult
from .stores import VectorStore


class BM25Retriever:
    def __init__(self, chunks: Iterable[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(chunk.text + " " + chunk.title) for chunk in self.chunks]
        self.doc_len = [len(tokens) for tokens in self.tokens]
        self.avg_len = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 1.0
        document_frequency: Counter[str] = Counter()
        for tokens in self.tokens:
            document_frequency.update(set(tokens))
        total = len(self.chunks)
        self.idf = {term: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5)) for term, frequency in document_frequency.items()}

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        query_tokens = tokenize(query)
        ranked: list[tuple[float, Chunk]] = []
        for chunk, tokens, length in zip(self.chunks, self.tokens, self.doc_len):
            counts = Counter(tokens)
            score = 0.0
            for term in query_tokens:
                if term not in counts:
                    continue
                frequency = counts[term]
                denominator = frequency + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                score += self.idf.get(term, 0.0) * (frequency * (self.k1 + 1)) / denominator
            if score:
                ranked.append((score, chunk))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [SearchResult(chunk=chunk, score=score, channel="bm25", rank=index) for index, (score, chunk) in enumerate(ranked[:k], 1)]


class VectorRetriever:
    def __init__(self, store: VectorStore):
        self.store = store

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        return self.store.search(query, k=k)


class GraphRetriever:
    def __init__(self, graph: KnowledgeGraph, chunks_by_id: dict[str, Chunk]):
        self.graph = graph
        self.chunks_by_id = chunks_by_id

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        rows = self.graph.related_chunk_ids(query, limit=k)
        return [
            SearchResult(chunk=self.chunks_by_id[chunk_id], score=score, channel="graph", rank=index, graph_path=path)
            for index, (chunk_id, score, path) in enumerate(rows, 1)
            if chunk_id in self.chunks_by_id
        ]


class HybridRetriever:
    """Fuse BM25, vector and graph channels with weighted RRF."""

    def __init__(
        self,
        bm25: BM25Retriever,
        vector: VectorRetriever,
        graph: GraphRetriever | None = None,
        weights: dict[str, float] | None = None,
        rrf_k: int = 60,
    ):
        self.bm25 = bm25
        self.vector = vector
        self.graph = graph
        self.weights = weights or {"bm25": 0.35, "vector": 0.45, "graph": 0.20}
        self.rrf_k = rrf_k

    def search(self, query: str, k: int = 10, use_graph: bool = True) -> list[SearchResult]:
        channels: dict[str, list[SearchResult]] = {
            "bm25": self.bm25.search(query, k=max(k, 20)),
            "vector": self.vector.search(query, k=max(k, 20)),
        }
        if use_graph and self.graph is not None:
            channels["graph"] = self.graph.search(query, k=max(k, 20))
        fused: dict[str, dict] = {}
        for channel, results in channels.items():
            weight = self.weights.get(channel, 0.0)
            for rank, result in enumerate(results, 1):
                row = fused.setdefault(result.chunk.id, {"chunk": result.chunk, "score": 0.0, "paths": []})
                row["score"] += weight / (self.rrf_k + rank)
                row["paths"].append(result.graph_path)
        ranked = sorted(fused.values(), key=lambda row: row["score"], reverse=True)[:k]
        return [
            SearchResult(
                chunk=row["chunk"],
                score=row["score"],
                channel="hybrid",
                rank=index,
                graph_path=next((path for path in row["paths"] if path), ()),
            )
            for index, row in enumerate(ranked, 1)
        ]


class LexicalReranker:
    """Fast local reranker; production can replace it with a cross-encoder."""

    def __init__(self, title_weight: float = 0.25, clause_weight: float = 0.1):
        self.title_weight = title_weight
        self.clause_weight = clause_weight

    def rerank(self, query: str, results: list[SearchResult], k: int = 5) -> list[SearchResult]:
        query_tokens = set(tokenize(query))
        rescored: list[SearchResult] = []
        for result in results:
            text_tokens = set(tokenize(result.chunk.text))
            overlap = len(query_tokens.intersection(text_tokens)) / max(len(query_tokens), 1)
            title_tokens = set(tokenize(result.chunk.title))
            title_overlap = len(query_tokens.intersection(title_tokens)) / max(len(query_tokens), 1)
            score = result.score + overlap + self.title_weight * title_overlap
            if result.chunk.clause_no and result.chunk.clause_no in query:
                score += self.clause_weight
            rescored.append(SearchResult(result.chunk, score, "rerank", result.rank, result.graph_path))
        rescored.sort(key=lambda item: item.score, reverse=True)
        return [SearchResult(item.chunk, item.score, item.channel, index, item.graph_path) for index, item in enumerate(rescored[:k], 1)]


class CrossEncoderReranker(LexicalReranker):
    """Optional sentence-transformers cross-encoder reranker."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        super().__init__()
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("CrossEncoderReranker requires sentence-transformers") from exc
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[SearchResult], k: int = 5) -> list[SearchResult]:
        if not results:
            return []
        scores = self.model.predict([(query, result.chunk.text) for result in results])
        ranked = sorted(zip(scores, results), key=lambda item: float(item[0]), reverse=True)[:k]
        return [SearchResult(result.chunk, float(score), "rerank", index, result.graph_path) for index, (score, result) in enumerate(ranked, 1)]
