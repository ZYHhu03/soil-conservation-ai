"""Knowledge graph construction and graph-aware evidence expansion."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from .embeddings import tokenize
from .schemas import Chunk


ENTITY_PATTERNS: dict[str, re.Pattern[str]] = {
    "clause": re.compile(r"第[一二三四五六七八九十百千万零〇0-9]+条"),
    "region": re.compile(r"(?:北京|天津|上海|重庆|河北|山西|辽宁|江苏|浙江|安徽|福建|江西|山东|河南|湖北|湖南|广东|广西|海南|四川|贵州|云南|陕西|甘肃|青海|内蒙古|西藏|宁夏|新疆)(?:省|市|自治区|特别行政区)?"),
    "stage": re.compile(r"(?:立项|设计|施工|竣工|验收|报批|审查|监测|监理)阶段?"),
    "measure": re.compile(r"(?:表土剥离|临时排水|沉沙池|拦挡|苫盖|植被恢复|弃土弃渣|水土保持监测)"),
    "project_type": re.compile(r"(?:水利|道路|铁路|矿山|房地产|风电|光伏|输变电|管线|工业园区|灌溉)项目"),
}


@dataclass(frozen=True)
class Entity:
    id: str
    name: str
    kind: str


@dataclass
class KnowledgeGraph:
    """Small graph implementation with the same contract as a graph database."""

    entities: dict[str, Entity] = field(default_factory=dict)
    edges: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    chunk_entities: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    entity_chunks: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    chunk_texts: dict[str, str] = field(default_factory=dict)

    def add_entity(self, name: str, kind: str) -> str:
        entity_id = f"{kind}:{name}"
        self.entities.setdefault(entity_id, Entity(entity_id, name, kind))
        return entity_id

    def add_chunk(self, chunk: Chunk) -> None:
        chunk_node = f"chunk:{chunk.id}"
        self.chunk_texts[chunk.id] = chunk.text
        names: list[str] = []
        for kind, pattern in ENTITY_PATTERNS.items():
            names.extend(pattern.findall(f"{chunk.title} {chunk.text}"))
        for kind, pattern in ENTITY_PATTERNS.items():
            for name in set(pattern.findall(f"{chunk.title} {chunk.text}")):
                entity_id = self.add_entity(name, kind=kind)
                self.edges[chunk_node].add(entity_id)
                self.edges[entity_id].add(chunk_node)
                self.chunk_entities[chunk.id].add(entity_id)
                self.entity_chunks[entity_id].add(chunk.id)
        # Terms shared by adjacent chunks create lightweight semantic edges.
        terms = set(tokenize(chunk.text))
        for other_id, other_entities in list(self.chunk_entities.items()):
            if other_id == chunk.id:
                continue
            if len(terms.intersection(set(tokenize(self.chunk_texts.get(other_id, ""))))) >= 2:
                self.edges[chunk_node].add(f"chunk:{other_id}")

    def add_chunks(self, chunks: Iterable[Chunk]) -> None:
        for chunk in chunks:
            self.add_chunk(chunk)

    def related_chunk_ids(self, query: str, limit: int = 20) -> list[tuple[str, float, tuple[str, ...]]]:
        query_tokens = set(tokenize(query))
        matched_entities: set[str] = set()
        for entity_id, entity in self.entities.items():
            if entity.name in query or entity.name.lower() in query.lower() or entity.name in query_tokens:
                matched_entities.add(entity_id)
        scores: dict[str, float] = defaultdict(float)
        paths: dict[str, tuple[str, ...]] = {}
        for entity_id in matched_entities:
            for chunk_id in self.entity_chunks.get(entity_id, set()):
                scores[chunk_id] += 1.0
                paths[chunk_id] = (entity_id, f"chunk:{chunk_id}")
                for neighbor in self.edges.get(entity_id, set()):
                    if neighbor.startswith("chunk:"):
                        neighbor_id = neighbor.removeprefix("chunk:")
                        scores[neighbor_id] += 0.35
                        paths.setdefault(neighbor_id, (entity_id, neighbor))
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        return [(chunk_id, score, paths[chunk_id]) for chunk_id, score in ranked]

    def to_dict(self) -> dict:
        return {
            "entities": [entity.__dict__ for entity in self.entities.values()],
            "edges": {key: sorted(value) for key, value in self.edges.items()},
            "chunk_entities": {key: sorted(value) for key, value in self.chunk_entities.items()},
            "chunk_texts": self.chunk_texts,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "KnowledgeGraph":
        graph = cls()
        for item in payload.get("entities", []):
            graph.entities[item["id"]] = Entity(**item)
        graph.edges = defaultdict(set, {key: set(value) for key, value in payload.get("edges", {}).items()})
        graph.chunk_entities = defaultdict(set, {key: set(value) for key, value in payload.get("chunk_entities", {}).items()})
        graph.chunk_texts = dict(payload.get("chunk_texts", {}))
        for chunk_id, entity_ids in graph.chunk_entities.items():
            for entity_id in entity_ids:
                graph.entity_chunks[entity_id].add(chunk_id)
        return graph
