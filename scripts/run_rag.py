#!/usr/bin/env python3
"""Run the local RAG pipeline over indexed chunks."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from soil_rag.chunking import StructuredChunker
from soil_rag.embeddings import HashEmbedding, SentenceTransformerEmbedding
from soil_rag.graph import KnowledgeGraph
from soil_rag.pipeline import RAGPipeline
from soil_rag.retrieval import BM25Retriever, GraphRetriever, HybridRetriever, VectorRetriever
from soil_rag.schemas import Chunk
from soil_rag.stores import InMemoryVectorStore, MilvusVectorStore


def load_chunks(path: Path) -> list[Chunk]:
    chunks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            item["title_path"] = tuple(item.get("title_path", []))
            chunks.append(Chunk(**item))
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True, type=Path)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--query", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/rag.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) if args.config.exists() else {}
    chunks = load_chunks(args.chunks)
    graph = KnowledgeGraph.from_dict(json.loads(args.graph.read_text(encoding="utf-8")))
    embedding_cfg = config.get("embedding", {})
    embedder = (
        SentenceTransformerEmbedding(embedding_cfg.get("model", "BAAI/bge-m3"))
        if embedding_cfg.get("provider") == "sentence-transformers"
        else HashEmbedding(int(embedding_cfg.get("dimension", 256)))
    )
    vector_cfg = config.get("vector_store", {})
    if vector_cfg.get("provider") == "milvus":
        store = MilvusVectorStore(
            embedder,
            collection_name=vector_cfg.get("collection", "soil_chunks"),
            uri=vector_cfg.get("uri", "http://localhost:19530"),
        )
    else:
        store = InMemoryVectorStore(embedder)
    if vector_cfg.get("provider") != "milvus":
        store.add(chunks)
    retriever = HybridRetriever(BM25Retriever(chunks), VectorRetriever(store), GraphRetriever(graph, {chunk.id: chunk for chunk in chunks}))
    response = RAGPipeline(retriever).invoke(args.query)
    print(json.dumps({
        "query": response.query,
        "intent": response.plan.intent.value,
        "rewritten_query": response.plan.rewritten_query,
        "answer": response.answer,
        "citations": [asdict(citation) for citation in response.citations],
        "confidence": response.confidence,
        "refused": response.refused,
        "verification": asdict(response.verification),
    }, ensure_ascii=False, indent=2, default=list))


if __name__ == "__main__":
    main()
