"""LangChain Runnable wrapper around the typed RAG pipeline."""

from __future__ import annotations

from .pipeline import RAGPipeline


def build_langchain_chain(pipeline: RAGPipeline):
    """Expose ``pipeline.invoke`` as a LangChain Runnable.

    Keeping the retrieval and verification logic in typed Python components
    makes the chain easy to test while allowing LangChain composition in an
    application graph.
    """
    try:
        from langchain_core.runnables import RunnableLambda
    except ImportError as exc:
        raise RuntimeError("build_langchain_chain requires langchain-core") from exc
    return RunnableLambda(lambda payload: pipeline.invoke(payload["query"] if isinstance(payload, dict) else str(payload)))
