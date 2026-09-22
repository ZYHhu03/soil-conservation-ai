"""Answer generation adapters and deterministic evidence-grounded baseline."""

from __future__ import annotations

from typing import Protocol

from .context import ContextAssembler
from .prompts import SYSTEM_PROMPT, build_prompt
from .schemas import Evidence, GeneratedAnswer, QueryPlan


class AnswerGenerator(Protocol):
    def generate(self, plan: QueryPlan, context: str, evidence: tuple[Evidence, ...]) -> GeneratedAnswer: ...


class ExtractiveAnswerGenerator:
    """Local baseline that selects evidence sentences and emits valid citations."""

    def __init__(self, max_sentences: int = 3, extractor=None):
        self.max_sentences = max_sentences
        if extractor is None:
            from .extraction import KeywordAnswerExtractor

            extractor = KeywordAnswerExtractor()
        self.extractor = extractor

    def generate(self, plan: QueryPlan, context: str, evidence: tuple[Evidence, ...]) -> GeneratedAnswer:
        if not evidence:
            return GeneratedAnswer("暂无可靠依据。请补充项目类型、所在地和办理阶段。", confidence=0.0, refused=True)
        selected: list[str] = []
        labels: list[str] = []
        for item in evidence:
            sentence = self.extractor.extract(plan.original_query, item.result.chunk.text).text.strip()
            if sentence and sentence not in selected:
                selected.append(sentence)
                labels.append(f"[{item.label}]")
            if len(selected) >= self.max_sentences:
                break
        answer = "".join(f"{sentence}{label}" for sentence, label in zip(selected, labels))
        return GeneratedAnswer(answer, confidence=min(0.95, 0.4 + 0.12 * len(selected)), refused=False)


class LangChainAnswerGenerator:
    """Adapter for any LangChain chat model implementing ``invoke``."""

    def __init__(self, llm):
        self.llm = llm

    def generate(self, plan: QueryPlan, context: str, evidence: tuple[Evidence, ...]) -> GeneratedAnswer:
        prompt = f"{SYSTEM_PROMPT}\n\n{build_prompt(plan, context)}"
        response = self.llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        refused = "暂无可靠依据" in text or "无法确认" in text
        return GeneratedAnswer(text=text.strip(), confidence=0.8 if evidence else 0.0, refused=refused)
