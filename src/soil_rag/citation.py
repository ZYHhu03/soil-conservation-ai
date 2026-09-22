"""Citation parsing and evidence coverage checks."""

from __future__ import annotations

import re

from .schemas import Citation, CitationVerification, Evidence


CITATION_RE = re.compile(r"\[(E\d+)\]")


class CitationVerifier:
    def __init__(self, min_coverage: float = 0.5):
        self.min_coverage = min_coverage

    def verify(self, answer: str, evidence: tuple[Evidence, ...]) -> CitationVerification:
        by_label = {item.label: item for item in evidence}
        labels = list(dict.fromkeys(CITATION_RE.findall(answer)))
        citations = tuple(
            Citation(label=label, chunk_id=by_label[label].result.chunk.id, valid=True)
            if label in by_label
            else Citation(label=label, chunk_id="", valid=False, reason="citation label is not in the evidence set")
            for label in labels
        )
        cited_labels = {citation.label for citation in citations if citation.valid}
        coverage = len(cited_labels) / len(evidence) if evidence else 0.0
        has_invalid = any(not citation.valid for citation in citations)
        refused = "暂无可靠依据" in answer or "无法确认" in answer
        valid = (not has_invalid) and (coverage >= self.min_coverage or refused)
        needs_retry = bool(evidence) and not valid
        reason = ""
        if has_invalid:
            reason = "answer contains an unknown citation label"
        elif evidence and coverage < self.min_coverage and not refused:
            reason = "evidence coverage is below threshold"
        return CitationVerification(citations, coverage, valid, needs_retry, reason)
