#!/usr/bin/env python3
"""Parse, structure-split and index a document directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from soil_rag.chunking import StructuredChunker
from soil_rag.embeddings import HashEmbedding, SentenceTransformerEmbedding
from soil_rag.graph import KnowledgeGraph
from soil_rag.parser import load_documents
from soil_rag.stores import InMemoryVectorStore, MilvusVectorStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/rag.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) if args.config.exists() else {}
    chunk_cfg = config.get("chunking", {})
    documents = load_documents(args.input)
    chunker = StructuredChunker(
        max_chars=int(chunk_cfg.get("max_chars", 800)),
        overlap=int(chunk_cfg.get("overlap", 100)),
        min_chars=int(chunk_cfg.get("min_chars", 30)),
    )
    chunks = chunker.chunk_documents(documents)
    graph = KnowledgeGraph()
    graph.add_chunks(chunks)

    embedding_cfg = config.get("embedding", {})
    if embedding_cfg.get("provider") == "sentence-transformers":
        embedder = SentenceTransformerEmbedding(embedding_cfg.get("model", "BAAI/bge-m3"))
    else:
        embedder = HashEmbedding(int(embedding_cfg.get("dimension", 256)))
    vector_cfg = config.get("vector_store", {})
    if vector_cfg.get("provider") == "milvus":
        store = MilvusVectorStore(
            embedder,
            collection_name=vector_cfg.get("collection", "soil_chunks"),
            uri=vector_cfg.get("uri", "http://localhost:19530"),
        )
    else:
        store = InMemoryVectorStore(embedder)
    store.add(chunks)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.__dict__, ensure_ascii=False, default=list) + "\n")
    (args.output_dir / "graph.json").write_text(json.dumps(graph.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"documents": len(documents), "chunks": len(chunks), "entities": len(graph.entities), "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
