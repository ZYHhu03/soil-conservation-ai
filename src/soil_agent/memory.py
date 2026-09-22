"""Working, episodic and semantic memory stores."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from soil_rag.embeddings import HashEmbedding, cosine_similarity

from .state import utc_now


@dataclass
class WorkingMemory:
    """Session-scoped slots and confirmed entities."""

    session_id: str
    project_type: str = ""
    region: str = ""
    stage: str = ""
    confirmed_entities: list[str] = field(default_factory=list)
    missing_slots: list[str] = field(default_factory=list)
    turns: int = 0

    def update(self, slots: dict[str, str], confirmed_entities: list[str] | None = None) -> None:
        for field_name in ("project_type", "region", "stage"):
            if slots.get(field_name):
                setattr(self, field_name, slots[field_name])
        if confirmed_entities:
            self.confirmed_entities = list(dict.fromkeys(self.confirmed_entities + confirmed_entities))
        self.missing_slots = [name for name in ("project_type", "region", "stage") if not getattr(self, name)]
        self.turns += 1

    def context(self) -> dict[str, Any]:
        return {
            "project_type": self.project_type,
            "region": self.region,
            "stage": self.stage,
            "confirmed_entities": self.confirmed_entities,
            "missing_slots": self.missing_slots,
            "turns": self.turns,
        }


@dataclass
class EpisodicMemory:
    id: str
    session_id: str
    query: str
    answer: str
    feedback: str = ""
    source_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class SemanticMemory:
    id: str
    subject: str
    content: str
    source_ids: list[str]
    confidence: float
    conditions: dict[str, str] = field(default_factory=dict)
    valid_from: str = ""
    valid_to: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class MemoryStore:
    """A persistent-ready memory interface with an in-process implementation."""

    def __init__(self, embedder=None):
        self.embedder = embedder or HashEmbedding(256)
        self.episodes: list[EpisodicMemory] = []
        self.semantics: dict[str, SemanticMemory] = {}
        self._semantic_vectors: dict[str, list[float]] = {}

    def add_episode(self, memory: EpisodicMemory) -> None:
        self.episodes.append(memory)

    def add_semantic(self, memory: SemanticMemory) -> None:
        self.semantics[memory.id] = memory
        self._semantic_vectors[memory.id] = self.embedder.embed(f"{memory.subject} {memory.content}")

    def search_semantic(self, query: str, limit: int = 5, min_confidence: float = 0.6) -> list[SemanticMemory]:
        query_vector = self.embedder.embed(query)
        ranked = [
            (cosine_similarity(query_vector, vector), memory)
            for memory_id, vector in self._semantic_vectors.items()
            if (memory := self.semantics[memory_id]).confidence >= min_confidence
        ]
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [memory for _, memory in ranked[:limit]]

    def search_episodes(self, session_id: str, query: str = "", limit: int = 5) -> list[EpisodicMemory]:
        episodes = [episode for episode in self.episodes if episode.session_id == session_id]
        if not query:
            return episodes[-limit:][::-1]
        query_terms = set(re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]{3,}", query))
        ranked = sorted(episodes, key=lambda item: len(query_terms.intersection(set(re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]{3,}", item.query + item.answer)))), reverse=True)
        return ranked[:limit]

    def context_for(self, session_id: str, query: str) -> dict[str, Any]:
        return {
            "episodes": [item.to_dict() for item in self.search_episodes(session_id, query)],
            "semantic": [item.to_dict() for item in self.search_semantic(query)],
        }


class FileMemoryStore(MemoryStore):
    """JSONL-backed memory store for restartable local experiments."""

    def __init__(self, root: str):
        from pathlib import Path
        import json

        super().__init__()
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._json = json
        self._load()

    def _load(self) -> None:
        import json

        episode_path = self.root / "episodes.jsonl"
        if episode_path.exists():
            for line in episode_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.episodes.append(EpisodicMemory(**json.loads(line)))
        semantic_path = self.root / "semantic.jsonl"
        if semantic_path.exists():
            for line in semantic_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.add_semantic(SemanticMemory(**json.loads(line)))

    def add_episode(self, memory: EpisodicMemory) -> None:
        super().add_episode(memory)
        with (self.root / "episodes.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(self._json.dumps(memory.to_dict(), ensure_ascii=False) + "\n")

    def add_semantic(self, memory: SemanticMemory) -> None:
        super().add_semantic(memory)
        with (self.root / "semantic.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(self._json.dumps(memory.to_dict(), ensure_ascii=False) + "\n")
