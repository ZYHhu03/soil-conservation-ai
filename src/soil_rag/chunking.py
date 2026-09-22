"""Structure-aware chunking for regulations and technical specifications."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .schemas import Chunk, SourceDocument


HEADING_RE = re.compile(r"^\s*(#{1,6})\s+(.+?)\s*$")
CLAUSE_RE = re.compile(
    r"^\s*((?:第[一二三四五六七八九十百千万零〇0-9]+条)|(?:[一二三四五六七八九十百千万]+、)|(?:\d+(?:\.\d+)*[.)]?))\s*(.*)$"
)
SENTENCE_RE = re.compile(r"(?<=[。！？；.!?;])")


@dataclass
class _Block:
    text: str
    title_path: tuple[str, ...]
    clause_no: str


class StructuredChunker:
    """Split by headings and clauses before applying a bounded text window."""

    def __init__(self, max_chars: int = 800, overlap: int = 100, min_chars: int = 30):
        if max_chars <= overlap or overlap < 0:
            raise ValueError("max_chars must be larger than overlap >= 0")
        self.max_chars = max_chars
        self.overlap = overlap
        self.min_chars = min_chars

    def _blocks(self, text: str) -> list[_Block]:
        title_stack: list[str] = []
        current_clause = ""
        buffer: list[str] = []
        blocks: list[_Block] = []

        def flush() -> None:
            nonlocal buffer
            content = "\n".join(line.strip() for line in buffer if line.strip()).strip()
            if content:
                blocks.append(_Block(content, tuple(title_stack), current_clause))
            buffer = []

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                flush()
                continue
            heading = HEADING_RE.match(line)
            if heading:
                flush()
                level = len(heading.group(1))
                title = heading.group(2).strip()
                title_stack = title_stack[: level - 1] + [title]
                current_clause = ""
                continue
            clause = CLAUSE_RE.match(line)
            if clause and (clause.group(1).startswith("第") or clause.group(1).endswith("、") or any(mark in clause.group(1) for mark in (".", ")"))):
                flush()
                current_clause = clause.group(1)
                if clause.group(2):
                    buffer.append(clause.group(2))
                continue
            buffer.append(line)
        flush()
        return blocks

    def _bounded(self, block: _Block) -> list[str]:
        if len(block.text) <= self.max_chars:
            return [block.text]
        sentences = [part for part in SENTENCE_RE.split(block.text) if part]
        pieces: list[str] = []
        current = ""
        for sentence in sentences:
            if current and len(current) + len(sentence) > self.max_chars:
                pieces.append(current.strip())
                tail = current[-self.overlap :] if self.overlap else ""
                current = tail + sentence
            else:
                current += sentence
        if current.strip():
            pieces.append(current.strip())
        return pieces

    def chunk(self, document: SourceDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        for block in self._blocks(document.text):
            for index, text in enumerate(self._bounded(block)):
                if len(text) < self.min_chars and chunks:
                    previous = chunks[-1]
                    merged = f"{previous.text}\n{text}".strip()
                    chunks[-1] = Chunk(
                        id=previous.id,
                        document_id=previous.document_id,
                        text=merged,
                        title_path=previous.title_path,
                        clause_no=previous.clause_no,
                        metadata=previous.metadata,
                    )
                    continue
                digest = hashlib.sha1(f"{document.id}:{len(chunks)}:{text}".encode("utf-8")).hexdigest()[:16]
                metadata = dict(document.metadata)
                metadata.update({"document_title": document.title, "chunk_index": len(chunks)})
                chunks.append(
                    Chunk(
                        id=f"{document.id}::{digest}",
                        document_id=document.id,
                        text=text,
                        title_path=block.title_path,
                        clause_no=block.clause_no,
                        metadata=metadata,
                    )
                )
        return chunks

    def chunk_documents(self, documents: list[SourceDocument]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self.chunk(document))
        return chunks
