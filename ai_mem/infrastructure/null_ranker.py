"""Fallback ranker when torch is not available.

Returns each candidate's cosine similarity unchanged so the final order
matches ChromaDB's native ranking. Satisfies the LearnedRanker protocol.
"""
from __future__ import annotations

from pathlib import Path

from ai_mem.domain.learning import RankingFeatures, TrainingExample, TrainingMetrics


class NullRanker:
    def rank(self, features: list[RankingFeatures]) -> list[float]:
        return [f.cosine_similarity for f in features]

    def train_step(self, examples: list[TrainingExample]) -> TrainingMetrics:
        return TrainingMetrics(n=0, skipped=True)

    def save(self, path: Path) -> None:
        return

    def load(self, path: Path) -> None:
        return
