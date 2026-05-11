"""Query memory semantically with re-ranking; records access and training signal."""
from __future__ import annotations

import time
from collections import defaultdict

from ai_mem.application.build_features import BuildFeaturesUseCase
from ai_mem.application.track_access import TrackAccessUseCase
from ai_mem.application.train_ranker import TrainRankerUseCase
from ai_mem.domain.learning import RankerProvider
from ai_mem.domain.memory import MemoryRepository, QueryResult

_FETCH_K = 50  # over-fetch for re-ranking


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
        self._session_hits: dict[str, set[str]] = defaultdict(set)

    def execute(
        self,
        collection: str,
        query: str,
        n_results: int = 5,
        max_age_days: float | None = None,
        type_filter: str | None = None,
    ) -> list[QueryResult]:
        now = time.time()
        candidates = self._repo.query(collection, query, _FETCH_K, max_age_days, type_filter)
        if not candidates:
            return []

        features = self._build_features.execute(candidates, now, self._session_hits.get(collection))
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

        returned_ids = {r.id for r in results}
        self._session_hits[collection].update(returned_ids)
        self._track_access.execute(collection, list(returned_ids))
        self._train_ranker.record_query(collection, candidates, features, now, returned_ids=returned_ids)

        results = self._append_linked(collection, results, returned_ids)
        return results

    def _append_linked(
        self, collection: str, results: list[QueryResult], result_ids: set[str]
    ) -> list[QueryResult]:
        """Follow 1-hop edges from result entries and append linked entries (budget: 2)."""
        linked: list[QueryResult] = []
        try:
            for result in results:
                if len(linked) >= 2:
                    break
                edges = self._repo.get_edges(collection, result.id)
                for edge in edges:
                    if len(linked) >= 2:
                        break
                    if edge.target_id in result_ids:
                        continue
                    entries = self._repo.get_by_ids(collection, [edge.target_id])
                    if not entries:
                        continue
                    entry = entries[0]
                    linked_meta = dict(entry.metadata)
                    linked_meta["via_edge"] = edge.edge_type
                    linked_meta["via_source"] = result.id
                    linked.append(
                        QueryResult(
                            rank=len(results) + len(linked) + 1,
                            id=entry.id,
                            score=0.0,
                            text=entry.text,
                            metadata=linked_meta,
                        )
                    )
                    result_ids.add(edge.target_id)
        except Exception:
            pass
        return results + linked
