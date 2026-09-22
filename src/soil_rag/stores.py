"""Vector storage adapters: an in-memory index and a Milvus implementation."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Protocol

from .embeddings import EmbeddingModel, cosine_similarity
from .schemas import Chunk, SearchResult


class VectorStore(Protocol):
    def add(self, chunks: Iterable[Chunk]) -> None: ...

    def search(self, query: str, k: int = 10) -> list[SearchResult]: ...


class InMemoryVectorStore:
    def __init__(self, embedder: EmbeddingModel):
        self.embedder = embedder
        self._items: dict[str, tuple[Chunk, list[float]]] = {}

    def add(self, chunks: Iterable[Chunk]) -> None:
        chunk_list = list(chunks)
        vectors = self.embedder.embed_many(chunk.text for chunk in chunk_list)
        self._items.update({chunk.id: (chunk, vector) for chunk, vector in zip(chunk_list, vectors)})

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        vector = self.embedder.embed(query)
        ranked = sorted(
            ((cosine_similarity(vector, item_vector), chunk) for chunk, item_vector in self._items.values()),
            key=lambda pair: pair[0],
            reverse=True,
        )[:k]
        return [SearchResult(chunk=chunk, score=score, channel="vector", rank=index) for index, (score, chunk) in enumerate(ranked, 1)]

    def __len__(self) -> int:
        return len(self._items)


class MilvusVectorStore:
    """Milvus adapter storing vectors and serialized chunk metadata.

    The adapter keeps the authoritative Chunk objects in memory for response
    assembly while Milvus owns ANN search. In a service deployment the metadata
    lookup can be replaced by a document store without changing the retriever.
    """

    def __init__(
        self,
        embedder: EmbeddingModel,
        collection_name: str = "soil_chunks",
        uri: str = "http://localhost:19530",
        consistency_level: str = "Bounded",
    ):
        try:
            from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
        except ImportError as exc:
            raise RuntimeError("MilvusVectorStore requires pymilvus") from exc
        self.embedder = embedder
        self.collection_name = collection_name
        self._chunks: dict[str, Chunk] = {}
        self._milvus = (Collection, CollectionSchema, DataType, FieldSchema, connections, utility)
        _, _, data_type, field_schema, connections, utility = self._milvus
        connections.connect(alias="default", uri=uri)
        if not utility.has_collection(collection_name):
            schema = CollectionSchema(
                fields=[
                    field_schema(name="id", dtype=data_type.VARCHAR, max_length=128, is_primary=True),
                    field_schema(name="vector", dtype=data_type.FLOAT_VECTOR, dim=embedder.dimension),
                    field_schema(name="payload", dtype=data_type.JSON),
                ],
                description="soil-conservation structured chunks",
            )
            collection = Collection(collection_name, schema=schema, consistency_level=consistency_level)
            collection.create_index("vector", {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}})
        self.collection = Collection(collection_name, consistency_level=consistency_level)
        self.collection.load()

    def add(self, chunks: Iterable[Chunk]) -> None:
        chunk_list = list(chunks)
        vectors = self.embedder.embed_many(chunk.text for chunk in chunk_list)
        rows = []
        for chunk, vector in zip(chunk_list, vectors):
            self._chunks[chunk.id] = chunk
            rows.append({"id": chunk.id, "vector": vector, "payload": json.loads(json.dumps(chunk.__dict__, ensure_ascii=False, default=list))})
        if rows:
            self.collection.insert(rows)
            self.collection.flush()

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        vector = self.embedder.embed(query)
        rows = self.collection.search(
            data=[vector],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"ef": max(64, k * 4)}},
            limit=k,
            output_fields=["payload"],
        )[0]
        results: list[SearchResult] = []
        for rank, hit in enumerate(rows, 1):
            chunk = self._chunks.get(str(hit.id))
            if chunk is None:
                payload = hit.entity.get("payload")
                chunk = Chunk(**payload)
            results.append(SearchResult(chunk=chunk, score=float(hit.score), channel="vector", rank=rank))
        return results
