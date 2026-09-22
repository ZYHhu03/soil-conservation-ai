"""BiLSTM + Attention answer-extraction model and a no-model fallback."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnswerSpan:
    start: int
    end: int
    text: str
    confidence: float


class BiLSTMAttentionExtractor:
    """Extractive QA head for evidence passages.

    The module predicts start/end token logits over a tokenized context. The
    vocabulary encoder and training loop remain replaceable so the same head
    can be paired with a domain tokenizer or a pretrained encoder.
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 128, dropout: float = 0.1):
        try:
            import torch.nn as nn
        except ImportError as exc:
            raise RuntimeError("BiLSTMAttentionExtractor requires torch") from exc

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
                self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
                self.attention = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
                self.fusion = nn.Linear(hidden_dim * 4, hidden_dim * 2)
                self.dropout = nn.Dropout(dropout)
                self.start = nn.Linear(hidden_dim * 2, 1)
                self.end = nn.Linear(hidden_dim * 2, 1)

            def forward(self, input_ids, attention_mask=None):
                hidden, _ = self.encoder(self.embedding(input_ids))
                weights = self.attention(hidden).squeeze(-1)
                if attention_mask is not None:
                    weights = weights.masked_fill(attention_mask == 0, -1e4)
                weights = weights.softmax(dim=-1).unsqueeze(-1)
                pooled = (hidden * weights).sum(dim=1, keepdim=True).expand_as(hidden)
                fused = self.dropout(self.fusion(__import__("torch").cat([hidden, pooled], dim=-1)))
                return self.start(fused).squeeze(-1), self.end(fused).squeeze(-1), weights.squeeze(-1)

        self.model = _Model()

    def __call__(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask)


def decode_span(tokens: list[str], start_logits, end_logits, max_length: int = 64) -> AnswerSpan:
    """Decode the highest-scoring valid span from model logits."""
    import math

    start = int(start_logits.argmax().item())
    end_candidates = [index for index in range(start, min(len(tokens), start + max_length))]
    end = max(end_candidates, key=lambda index: float(end_logits[index])) if end_candidates else start
    confidence = 1 / (1 + math.exp(-float(start_logits[start]))) * 1 / (1 + math.exp(-float(end_logits[end])))
    return AnswerSpan(start, end, "".join(tokens[start : end + 1]), confidence)


class KeywordAnswerExtractor:
    """Sentence-level extractor used by the local RAG pipeline."""

    def extract(self, query: str, text: str) -> AnswerSpan:
        import re

        query_terms = set(re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]{3,}", query))
        sentences = [item.strip() for item in re.split(r"(?<=[。！？；.!?;])", text) if item.strip()]
        if not sentences:
            return AnswerSpan(0, 0, text, 0.0)
        sentence = max(sentences, key=lambda item: len(query_terms.intersection(set(re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]{3,}", item)))))
        return AnswerSpan(0, max(len(sentence) - 1, 0), sentence, min(1.0, 0.3 + len(query_terms) * 0.1))
