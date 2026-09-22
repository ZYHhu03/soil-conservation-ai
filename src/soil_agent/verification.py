"""Claim-level verification against retrieved evidence and graph relations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from soil_rag.embeddings import tokenize
from soil_rag.graph import KnowledgeGraph

from .state import Claim, ClaimCheck


CLAIM_CATEGORIES = {
    "policy": ("政策", "条款", "规定", "依据", "文件"),
    "process": ("审批", "流程", "办理", "阶段", "审查"),
    "material": ("材料", "资料", "清单", "提交", "申报"),
    "technical": ("措施", "排水", "拦挡", "苫盖", "监测", "参数"),
}


class ClaimExtractor:
    """Extract factual units and citation labels from a generated answer."""

    def extract(self, text: str) -> list[Claim]:
        sentences = re.findall(r"[^。！？；.!?;]+[。！？；.!?;](?:\s*\[E\d+\])*|[^。！？；.!?;]+$", text)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        claims: list[Claim] = []
        for index, sentence in enumerate(sentences, 1):
            clean = re.sub(r"\[E\d+\]", "", sentence).strip()
            if not clean or clean in {"暂无可靠依据", "暂无可靠依据。"}:
                continue
            category = "general"
            for name, keywords in CLAIM_CATEGORIES.items():
                if any(keyword in clean for keyword in keywords):
                    category = name
                    break
            citations = re.findall(r"\[(E\d+)\]", sentence)
            claims.append(Claim(id=f"claim-{index}", text=clean, category=category, normalized=self.normalize(clean), citations=citations))
        return claims

    @staticmethod
    def normalize(text: str) -> str:
        return re.sub(r"\s+", "", text.lower().replace("。", ""))


@dataclass(frozen=True)
class VerificationSummary:
    checks: tuple[ClaimCheck, ...]
    coverage: float
    valid: bool
    needs_retry: bool
    abstain: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "checks": [check.to_dict() for check in self.checks],
            "coverage": self.coverage,
            "valid": self.valid,
            "needs_retry": self.needs_retry,
            "abstain": self.abstain,
            "reason": self.reason,
        }


class ClaimLevelVerifier:
    def __init__(self, graph: KnowledgeGraph | None = None, min_overlap: float = 0.25, min_coverage: float = 0.8):
        self.graph = graph
        self.min_overlap = min_overlap
        self.min_coverage = min_coverage

    def verify(self, claims: Iterable[Claim], evidence: list[dict[str, Any]]) -> VerificationSummary:
        claims = list(claims)
        checks: list[ClaimCheck] = []
        for claim in claims:
            claim_terms = set(tokenize(claim.text))
            best_score = 0.0
            best_ids: list[str] = []
            conflict = False
            claim_entity_ids = {
                entity_id
                for entity_id, entity in (self.graph.entities.items() if self.graph is not None else [])
                if entity.name in claim.text
            }
            for item in evidence:
                evidence_text = str(item.get("text", ""))
                evidence_terms = set(tokenize(evidence_text))
                overlap = len(claim_terms.intersection(evidence_terms)) / max(len(claim_terms), 1)
                if claim.normalized and claim.normalized in ClaimExtractor.normalize(evidence_text):
                    overlap = 1.0
                chunk_entity_ids = set(self.graph.chunk_entities.get(str(item.get("id", "")), set())) if self.graph is not None else set()
                if claim_entity_ids.intersection(chunk_entity_ids):
                    overlap = max(overlap, 0.75)
                if overlap > best_score:
                    best_score = overlap
                    best_ids = [str(item.get("id", ""))]
                elif overlap == best_score and overlap >= self.min_overlap:
                    best_ids.append(str(item.get("id", "")))
                if self._conflicting_numbers(claim.text, evidence_text):
                    conflict = True
            if claim.category == "policy" and any(word in claim.text for word in ("当前", "现行", "有效", "最新")):
                if not any(item.get("source_version") or item.get("metadata", {}).get("source_version") for item in evidence):
                    conflict = True
            supported = best_score >= self.min_overlap and not conflict
            checks.append(ClaimCheck(claim.id, supported, min(best_score, 1.0), best_ids, conflict, "matched evidence" if supported else "insufficient or conflicting evidence"))
        coverage = sum(check.supported for check in checks) / len(checks) if checks else 0.0
        valid = bool(checks) and coverage >= self.min_coverage and not any(check.conflict for check in checks)
        needs_retry = bool(claims) and not valid
        abstain = bool(claims) and not valid
        reason = ""
        if not checks:
            reason = "no factual claims were extracted"
        elif any(check.conflict for check in checks):
            reason = "claim conflicts with evidence or current-version constraint"
        elif coverage < self.min_coverage:
            reason = "claim evidence coverage is below threshold"
        return VerificationSummary(tuple(checks), coverage, valid, needs_retry, abstain, reason)

    @staticmethod
    def _conflicting_numbers(claim: str, evidence: str) -> bool:
        claim_numbers = set(re.findall(r"\d+(?:\.\d+)?%?", claim))
        evidence_numbers = set(re.findall(r"\d+(?:\.\d+)?%?", evidence))
        return bool(claim_numbers and evidence_numbers and claim_numbers.isdisjoint(evidence_numbers) and any(char.isdigit() for char in claim))
