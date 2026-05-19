"""Fallback ranker when torch is not available.

Applies a session-hit penalty so entries already returned this session are
downranked, preventing the same high-access incumbents from dominating every
query. Satisfies the LearnedRanker protocol.
"""
from __future__ import annotations

import math
from pathlib import Path

from ai_mem.domain.learning import RankingFeatures, TrainingExample, TrainingMetrics

_SESSION_HIT_PENALTY = 0.7
_EXPLORE_BONUS = 0.15  # UCB coefficient — decays as 1/sqrt(access_count)


class NullRanker:
    def rank(self, features: list[RankingFeatures]) -> list[float]:
        return [
            f.cosine_similarity * (_SESSION_HIT_PENALTY if f.session_hit else 1.0)
            + _EXPLORE_BONUS / math.sqrt(max(1, f.access_count))
            for f in features
        ]

    def train_step(self, examples: list[TrainingExample]) -> TrainingMetrics:
        return TrainingMetrics(n=0, skipped=True)

    def save(self, path: Path) -> None:
        return

    def load(self, path: Path) -> None:
        return
