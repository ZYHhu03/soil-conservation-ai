"""Batch document loading for policy, standard and monitoring documents."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Iterator

from .schemas import SourceDocument


TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".html", ".htm"}


def _strip_html(text: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</(p|div|h[1-6]|li|tr)>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"[ \t]+", " ", text)


def _document_from_json(path: Path, payload: object) -> Iterator[SourceDocument]:
    records = payload if isinstance(payload, list) else [payload]
    for index, item in enumerate(records):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", item.get("content", ""))).strip()
        if not text:
            continue
        doc_id = str(item.get("id", f"{path.stem}-{index:04d}"))
        title = str(item.get("title", path.stem))
        metadata = dict(item.get("metadata", {})) if isinstance(item.get("metadata"), dict) else {}
        metadata.update({"source_path": str(path), "format": "json"})
        yield SourceDocument(id=doc_id, title=title, text=text, metadata=metadata)


def parse_document(path: str | Path) -> list[SourceDocument]:
    """Parse one UTF-8 text/Markdown/HTML/JSON document."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF parsing requires the 'pdf' extra (pypdf).") from exc
        reader = PdfReader(str(source))
        pages = [page.extract_text() or "" for page in reader.pages]
        return [
            SourceDocument(
                id=source.stem,
                title=source.stem,
                text="\n\n".join(pages),
                metadata={"source_path": str(source), "format": "pdf", "pages": len(pages)},
            )
        ]
    if suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        return list(_document_from_json(source, payload))
    if suffix not in TEXT_SUFFIXES:
        return []
    text = source.read_text(encoding="utf-8")
    if suffix in {".html", ".htm"}:
        text = _strip_html(text)
    return [
        SourceDocument(
            id=source.stem,
            title=source.stem,
            text=text,
            metadata={"source_path": str(source), "format": suffix.removeprefix(".")},
        )
    ]


def load_documents(root: str | Path) -> list[SourceDocument]:
    """Load all supported documents below a file or directory."""
    path = Path(root)
    paths = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    documents: list[SourceDocument] = []
    seen: set[str] = set()
    for item in paths:
        for document in parse_document(item):
            if document.id in seen:
                raise ValueError(f"duplicate document id: {document.id}")
            seen.add(document.id)
            documents.append(document)
    return documents


def to_langchain_documents(documents: Iterable[SourceDocument]):
    """Convert typed source records to LangChain ``Document`` objects."""
    try:
        from langchain_core.documents import Document
    except ImportError as exc:
        raise RuntimeError("to_langchain_documents requires langchain-core") from exc
    return [Document(page_content=document.text, metadata={"id": document.id, "title": document.title, **dict(document.metadata)}) for document in documents]
