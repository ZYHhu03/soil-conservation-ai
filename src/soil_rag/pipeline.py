"""End-to-end query orchestration with retry and citation validation."""

from __future__ import annotations

from .citation import CitationVerifier
from .context import ContextAssembler
from .generation import AnswerGenerator, ExtractiveAnswerGenerator
from .query import QueryPlanner
from .retrieval import HybridRetriever, LexicalReranker
from .schemas import RAGResponse


class RAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: LexicalReranker | None = None,
        generator: AnswerGenerator | None = None,
        planner: QueryPlanner | None = None,
        assembler: ContextAssembler | None = None,
        verifier: CitationVerifier | None = None,
        top_k: int = 5,
        retry_on_low_evidence: bool = True,
    ):
        self.retriever = retriever
        self.reranker = reranker or LexicalReranker()
        self.generator = generator or ExtractiveAnswerGenerator()
        self.planner = planner or QueryPlanner()
        self.assembler = assembler or ContextAssembler()
        self.verifier = verifier or CitationVerifier()
        self.top_k = top_k
        self.retry_on_low_evidence = retry_on_low_evidence

    def invoke(self, query: str) -> RAGResponse:
        plan = self.planner.plan(query)
        search_query = " ".join(plan.expansions[:3])
        results = self.retriever.search(search_query, k=max(self.top_k * 2, 10), use_graph=plan.use_graph)
        reranked = self.reranker.rerank(query, results, k=self.top_k)
        context, evidence = self.assembler.assemble(reranked)
        generated = self.generator.generate(plan, context, evidence)
        verification = self.verifier.verify(generated.text, evidence)

        if self.retry_on_low_evidence and verification.needs_retry and len(plan.expansions) > 1:
            retry_results = self.retriever.search(plan.expansions[-1], k=max(self.top_k * 2, 10), use_graph=plan.use_graph)
            retry_reranked = self.reranker.rerank(plan.rewritten_query, retry_results, k=self.top_k)
            retry_context, retry_evidence = self.assembler.assemble(retry_reranked)
            retry_generated = self.generator.generate(plan, retry_context, retry_evidence)
            retry_verification = self.verifier.verify(retry_generated.text, retry_evidence)
            if retry_verification.valid or not verification.valid:
                reranked, evidence, generated, verification = retry_reranked, retry_evidence, retry_generated, retry_verification

        return RAGResponse(
            query=query,
            plan=plan,
            answer=generated.text,
            citations=verification.citations,
            evidence=evidence,
            confidence=generated.confidence if verification.valid else min(generated.confidence, 0.2),
            refused=generated.refused,
            verification=verification,
        )
