"""Small, deterministic metrics for a first offline comparison."""

from __future__ import annotations

import re
from collections.abc import Iterable


def keyword_hit_rate(predictions: Iterable[str], references: Iterable[str]) -> float:
    """Fraction of references whose CJK/ASCII tokens occur in the prediction.

    This intentionally stays conservative: it is a triage metric, not a semantic
    correctness score. Human review remains required for regulatory answers.
    """
    total = hits = 0
    for prediction, reference in zip(predictions, references):
        tokens = [token for token in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]{3,}", reference)]
        if not tokens:
            continue
        total += 1
        hits += int(sum(token in prediction for token in tokens) / len(tokens) >= 0.5)
    return hits / total if total else 0.0


def refusal_constraint_rate(predictions: Iterable[str], references: Iterable[str]) -> float:
    """Check that examples asking for missing context preserve a refusal cue."""
    cues = ("无法确认", "信息不足", "需要补充", "以现行要求为准", "暂无可靠依据")
    total = hits = 0
    for prediction, reference in zip(predictions, references):
        if not any(cue in reference for cue in cues):
            continue
        total += 1
        hits += int(any(cue in prediction for cue in cues))
    return hits / total if total else 0.0
