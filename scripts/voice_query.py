#!/usr/bin/env python3
"""Run ASR -> RAG -> TTS for one audio file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from soil_rag.embeddings import HashEmbedding
from soil_rag.graph import KnowledgeGraph
from soil_rag.pipeline import RAGPipeline
from soil_rag.retrieval import BM25Retriever, GraphRetriever, HybridRetriever, VectorRetriever
from soil_rag.speech import HTTPASRClient, HTTPTTSClient, VoiceRAGService
from soil_rag.stores import InMemoryVectorStore

from run_rag import load_chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--chunks", required=True, type=Path)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--asr-endpoint", required=True)
    parser.add_argument("--tts-endpoint", required=True)
    parser.add_argument("--output-audio", required=True, type=Path)
    parser.add_argument("--api-key", default="")
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    graph = KnowledgeGraph.from_dict(json.loads(args.graph.read_text(encoding="utf-8")))
    store = InMemoryVectorStore(HashEmbedding())
    store.add(chunks)
    retriever = HybridRetriever(BM25Retriever(chunks), VectorRetriever(store), GraphRetriever(graph, {chunk.id: chunk for chunk in chunks}))
    service = VoiceRAGService(RAGPipeline(retriever), HTTPASRClient(args.asr_endpoint, args.api_key), HTTPTTSClient(args.tts_endpoint, args.api_key))
    query, response, audio = service.invoke(args.audio.read_bytes())
    args.output_audio.parent.mkdir(parents=True, exist_ok=True)
    args.output_audio.write_bytes(audio)
    print(json.dumps({"query": query, "answer": response.answer, "output_audio": str(args.output_audio)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
