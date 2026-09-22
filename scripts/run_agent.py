#!/usr/bin/env python3
"""Run the stateful ReAct Agent over the RAG index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from soil_rag.embeddings import HashEmbedding, SentenceTransformerEmbedding
from soil_rag.graph import KnowledgeGraph
from soil_rag.pipeline import RAGPipeline
from soil_rag.retrieval import BM25Retriever, GraphRetriever, HybridRetriever, VectorRetriever
from soil_rag.stores import InMemoryVectorStore
from soil_agent.memory import FileMemoryStore
from soil_agent.workflow import AgentRuntime

from run_rag import load_chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True, type=Path)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--query", required=True)
    parser.add_argument("--session-id", default="demo-session")
    parser.add_argument("--config", type=Path, default=Path("configs/rag.yaml"))
    parser.add_argument("--agent-config", type=Path, default=Path("configs/agent.yaml"))
    parser.add_argument("--memory-dir", type=Path)
    parser.add_argument("--use-langgraph", action="store_true")
    args = parser.parse_args()
    rag_config = yaml.safe_load(args.config.read_text(encoding="utf-8")) if args.config.exists() else {}
    agent_config = yaml.safe_load(args.agent_config.read_text(encoding="utf-8")) if args.agent_config.exists() else {}
    agent_options = agent_config.get("agent", {})
    chunks = load_chunks(args.chunks)
    graph = KnowledgeGraph.from_dict(json.loads(args.graph.read_text(encoding="utf-8")))
    embedding_cfg = rag_config.get("embedding", {})
    embedder = (
        SentenceTransformerEmbedding(embedding_cfg.get("model", "BAAI/bge-m3"))
        if embedding_cfg.get("provider") == "sentence-transformers"
        else HashEmbedding(int(embedding_cfg.get("dimension", 256)))
    )
    store = InMemoryVectorStore(embedder)
    store.add(chunks)
    retriever = HybridRetriever(BM25Retriever(chunks), VectorRetriever(store), GraphRetriever(graph, {chunk.id: chunk for chunk in chunks}))
    memory_dir = args.memory_dir or Path(agent_config.get("memory", {}).get("root", "data/agent/memory"))
    runtime = AgentRuntime(
        RAGPipeline(retriever),
        graph=graph,
        chunks_by_id={chunk.id: chunk for chunk in chunks},
        memory_store=FileMemoryStore(str(memory_dir)),
        max_tool_calls=int(agent_options.get("max_tool_calls", 8)),
        max_retries=int(agent_options.get("max_retries", 1)),
    )
    state = runtime.invoke(args.query, args.session_id, use_langgraph=args.use_langgraph or bool(agent_options.get("use_langgraph", False)))
    print(json.dumps({
        "session_id": state["session_id"],
        "query": state["user_query"],
        "intent": state.get("intent"),
        "slots": state.get("slots", {}),
        "subtasks": state.get("subtasks", []),
        "tool_trace": state.get("tool_trace", []),
        "answer": state.get("answer", ""),
        "citations": state.get("citations", []),
        "claims": state.get("claims", []),
        "claim_checks": state.get("claim_checks", []),
        "verification": state.get("verification", {}),
        "status": state.get("status"),
        "abstained": state.get("abstained", False),
    }, ensure_ascii=False, indent=2, default=list))


if __name__ == "__main__":
    main()
