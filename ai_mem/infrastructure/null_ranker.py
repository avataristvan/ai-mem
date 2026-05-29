"""Fallback ranker when torch is not available.

Applies a session-hit penalty so entries already returned this session are
downranked, preventing the same high-access incumbents from dominating every
query. Satisfies the LearnedRanker protocol.
"""
from __future__ import annotations

import math
from pathlib import Path

from ai_mem.domain.learning import RankingFeatures, TrainingExample, TrainingMetrics

_SESSION_HIT_PENALTY = 0.7   # floor penalty for a first-session hit (low-access entry)
_SESSION_HIT_SATURATION = 10  # access_count at which the penalty fully lifts
_EXPLORE_BONUS = 0.15         # UCB coefficient — decays as 1/sqrt(access_count)


def _session_hit_multiplier(access_count: int) -> float:
    """Penalty for an entry seen earlier this session, scaled by confidence.

    Low-access entries (new, untested) get the full 0.7 penalty — diversity push.
    High-access entries (load-bearing, consistently useful) approach 1.0 — no suppression.
    """
    confidence = min(1.0, access_count / _SESSION_HIT_SATURATION)
    return _SESSION_HIT_PENALTY + (1.0 - _SESSION_HIT_PENALTY) * confidence


class NullRanker:
    def rank(self, features: list[RankingFeatures]) -> list[float]:
        return [
            f.cosine_similarity * (
                _session_hit_multiplier(f.access_count) if f.session_hit else 1.0
            )
            + _EXPLORE_BONUS / math.sqrt(max(1, f.access_count))
            for f in features
        ]

    def train_step(self, examples: list[TrainingExample]) -> TrainingMetrics:
        return TrainingMetrics(n=0, skipped=True)

    def save(self, path: Path) -> None:
        return

    def load(self, path: Path) -> None:
        return
