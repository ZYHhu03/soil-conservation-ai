#!/usr/bin/env python3
"""Line-delimited JSON MCP bridge for a local tool registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from soil_rag.embeddings import HashEmbedding
from soil_rag.graph import KnowledgeGraph
from soil_rag.pipeline import RAGPipeline
from soil_rag.retrieval import BM25Retriever, GraphRetriever, HybridRetriever, VectorRetriever
from soil_rag.stores import InMemoryVectorStore
from soil_agent.mcp import MCPToolServer
from soil_agent.tools import KnowledgeGraphTool, MemoryLookupTool, RagSearchTool, ToolRegistry
from soil_agent.memory import MemoryStore

from run_rag import load_chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True, type=Path)
    parser.add_argument("--graph", required=True, type=Path)
    args = parser.parse_args()
    chunks = load_chunks(args.chunks)
    graph = KnowledgeGraph.from_dict(json.loads(args.graph.read_text(encoding="utf-8")))
    store = InMemoryVectorStore(HashEmbedding())
    store.add(chunks)
    rag = RAGPipeline(HybridRetriever(BM25Retriever(chunks), VectorRetriever(store), GraphRetriever(graph, {chunk.id: chunk for chunk in chunks})))
    registry = ToolRegistry([RagSearchTool(rag), KnowledgeGraphTool(graph), MemoryLookupTool(MemoryStore())])
    server = MCPToolServer(registry)
    for line in sys.stdin:
        if line.strip():
            print(json.dumps(server.handle(json.loads(line)), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
