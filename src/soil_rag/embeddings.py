"""Embedding interfaces with a deterministic local baseline."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable
from typing import Protocol


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_/-]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class EmbeddingModel(Protocol):
    dimension: int

    def embed(self, text: str) -> list[float]: ...

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]: ...


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


class HashEmbedding:
    """Dependency-free hashed TF embedding for local smoke tests and fallback mode."""

    def __init__(self, dimension: int = 256):
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = tokenize(text)
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class SentenceTransformerEmbedding:
    """Sentence-transformers adapter used by the production vector index."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("SentenceTransformerEmbedding requires sentence-transformers") from exc
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = int(self.model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> list[float]:
        return list(self.model.encode(text, normalize_embeddings=True).tolist())

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        vectors = self.model.encode(list(texts), normalize_embeddings=True)
        return [list(vector.tolist()) for vector in vectors]
