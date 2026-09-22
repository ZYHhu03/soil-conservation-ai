#!/usr/bin/env python3
"""Compare BM25, vector and KG-enhanced hybrid retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from soil_rag.embeddings import HashEmbedding
from soil_rag.embeddings import SentenceTransformerEmbedding
from soil_rag.evaluation import EvaluationExample, evaluate_retriever
from soil_rag.extraction import KeywordAnswerExtractor
from soil_rag.graph import KnowledgeGraph
from soil_rag.retrieval import BM25Retriever, GraphRetriever, HybridRetriever, VectorRetriever
from soil_rag.schemas import Chunk
from soil_rag.stores import InMemoryVectorStore

from run_rag import load_chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True, type=Path)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--eval-file", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/rag.yaml"))
    args = parser.parse_args()
    config = {}
    if args.config.exists():
        import yaml

        config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    chunks = load_chunks(args.chunks)
    graph = KnowledgeGraph.from_dict(json.loads(args.graph.read_text(encoding="utf-8")))
    embedding_cfg = config.get("embedding", {})
    embedder = (
        SentenceTransformerEmbedding(embedding_cfg.get("model", "BAAI/bge-m3"))
        if embedding_cfg.get("provider") == "sentence-transformers"
        else HashEmbedding(int(embedding_cfg.get("dimension", 256)))
    )
    store = InMemoryVectorStore(embedder)
    store.add(chunks)
    bm25 = BM25Retriever(chunks)
    vector = VectorRetriever(store)
    graph_retriever = GraphRetriever(graph, {chunk.id: chunk for chunk in chunks})
    answer_extractor = KeywordAnswerExtractor()
    hybrid = HybridRetriever(bm25, vector, graph_retriever)
    examples = []
    chunk_ids = {chunk.id for chunk in chunks}
    for line in args.eval_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            relevant_ids = set(row.get("relevant_chunk_ids", []))
            for document_id in row.get("relevant_document_ids", []):
                relevant_ids.update(chunk.id for chunk in chunks if chunk.document_id == document_id)
            relevant_ids = {item for item in relevant_ids if item in chunk_ids}
            examples.append(EvaluationExample(row["id"], row["query"], frozenset(relevant_ids), row.get("reference_answer", ""), tuple(row.get("required_terms", []))))
    methods = {
        "bm25": lambda query, k: bm25.search(query, k),
        "vector": lambda query, k: vector.search(query, k),
        "hybrid": lambda query, k: hybrid.search(query, k, use_graph=True),
    }
    def answer_fn(example, results):
        return answer_extractor.extract(example.query, results[0].chunk.text).text if results else ""

    report = {name: evaluate_retriever(retriever, examples, answer_fn=answer_fn).__dict__ for name, retriever in methods.items()}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
