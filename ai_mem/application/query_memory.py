"""Query memory semantically with re-ranking; records access and training signal."""
from __future__ import annotations

import time

from ai_mem.application.build_features import BuildFeaturesUseCase
from ai_mem.application.track_access import TrackAccessUseCase
from ai_mem.application.train_ranker import TrainRankerUseCase
from ai_mem.domain.learning import RankerProvider
from ai_mem.domain.memory import MemoryRepository, QueryResult

_FETCH_K = 20  # over-fetch for re-ranking


class QueryMemoryUseCase:
    def __init__(
        self,
        repo: MemoryRepository,
        track_access: TrackAccessUseCase,
        build_features: BuildFeaturesUseCase,
        train_ranker: TrainRankerUseCase,
        ranker_provider: RankerProvider,
    ) -> None:
        self._repo = repo
        self._track_access = track_access
        self._build_features = build_features
        self._train_ranker = train_ranker
        self._ranker_provider = ranker_provider

    def execute(
        self,
        collection: str,
        query: str,
        n_results: int = 5,
        max_age_days: float | None = None,
    ) -> list[QueryResult]:
        now = time.time()
        candidates = self._repo.query(collection, query, _FETCH_K, max_age_days)
        if not candidates:
            return []

        features = self._build_features.execute(candidates, now)
        ranker = self._ranker_provider.get(collection)
        scores = ranker.rank(features)
        if len(scores) != len(candidates):
            raise RuntimeError(
                f"ranker.rank() returned {len(scores)} scores for {len(candidates)} candidates"
            )

        ranked = sorted(
            zip(candidates, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )
        results = [r for r, _ in ranked[:n_results]]
        for i, r in enumerate(results):
            r.rank = i + 1

        self._track_access.execute(collection, [r.id for r in results])
        self._train_ranker.record_query(collection, candidates, features, now)

        return results
