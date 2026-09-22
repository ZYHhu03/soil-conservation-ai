"""Evidence formatting and context-window budgeting."""

from __future__ import annotations

from .schemas import Evidence, SearchResult


class ContextAssembler:
    def __init__(self, max_chars: int = 6000, per_chunk_chars: int = 1200):
        self.max_chars = max_chars
        self.per_chunk_chars = per_chunk_chars

    def assemble(self, results: list[SearchResult]) -> tuple[str, tuple[Evidence, ...]]:
        lines: list[str] = []
        evidence: list[Evidence] = []
        used = 0
        for index, result in enumerate(results, 1):
            text = result.chunk.text.strip()
            if len(text) > self.per_chunk_chars:
                text = text[: self.per_chunk_chars].rstrip() + "……"
            title = result.chunk.title or result.chunk.metadata.get("document_title", result.chunk.document_id)
            clause = f"；条款={result.chunk.clause_no}" if result.chunk.clause_no else ""
            line = f"[E{index}] chunk_id={result.chunk.id}；标题={title}{clause}；内容={text}"
            if used + len(line) + 1 > self.max_chars:
                break
            lines.append(line)
            evidence.append(Evidence(label=f"E{index}", result=result))
            used += len(line) + 1
        return "\n".join(lines), tuple(evidence)
