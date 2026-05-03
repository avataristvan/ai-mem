"""Orchestrates training-buffer writes and gradient steps for the learned ranker."""
from __future__ import annotations

import math
import sys
from typing import Callable

from ai_mem.domain.learning import (
    LearnedRanker,
    RankingFeatures,
    RankerScope,
    TrainingBufferRepository,
    TrainingExample,
    TrainingMetrics,
)
from ai_mem.domain.memory import MemoryRepository, QueryResult

_LABEL_EPSILON = 1.0  # seconds — avoids labeling the recording tick itself

_DEFAULT_SCOPE_RESOLVER: Callable[[str], RankerScope] = lambda c: RankerScope(name=c, mode="isolated")


class TrainRankerUseCase:
    def __init__(
        self,
        repo: MemoryRepository,
        storage: TrainingBufferRepository,
        ranker_factory: Callable[[], LearnedRanker],
        label_window_days: float = 7.0,
        buffer_max: int = 1000,
        scope_resolver: Callable[[str], RankerScope] = _DEFAULT_SCOPE_RESOLVER,
    ) -> None:
        self._repo = repo
        self._storage = storage
        self._ranker_factory = ranker_factory
        self._label_window_days = label_window_days
        self._buffer_max = buffer_max
        self._scope_resolver = scope_resolver

    def _scope_name(self, collection: str) -> str:
        s = self._scope_resolver(collection)
        return s.group if s.mode == "hybrid" else s.name

    def record_query(
        self,
        collection: str,
        retrieved: list[QueryResult],
        features: list[RankingFeatures],
        now: float,
        returned_ids: set[str] | None = None,
    ) -> None:
        scope = self._scope_name(collection)
        returned = returned_ids or set()
        all_ids = [r.id for r in retrieved]
        examples = [
            TrainingExample(
                memory_id=r.id,
                features=feat,
                retrieved_at=now,
                co_activated_ids=[id_ for id_ in all_ids if id_ != r.id],
                source_collection=collection,
                # Entries returned to the caller are immediately labeled positive;
                # candidates that didn't make the cut wait for the 7-day access window.
                target_future_access=1.0 if r.id in returned else None,
            )
            for r, feat in zip(retrieved, features)
        ]
        self._storage.append_examples(scope, examples)
        self._storage.prune_buffer(scope, self._buffer_max)

    def train_step(self, collection: str, now: float) -> TrainingMetrics:
        scope = self._scope_name(collection)
        buffer = self._storage.load_buffer(scope)
        if not buffer:
            return TrainingMetrics(n=0, skipped=True)

        label_cutoff = self._label_window_days * 86400

        # Group unlabeled-past-window examples by their source collection so we
        # fetch labels from the correct collection even in hybrid mode.
        by_source: dict[str, list[TrainingExample]] = {}
        for ex in buffer:
            if ex.target_future_access is None and ex.retrieved_at + label_cutoff <= now:
                src = ex.source_collection or collection
                by_source.setdefault(src, []).append(ex)

        for src_col, pending in by_source.items():
            ids_to_fetch = [ex.memory_id for ex in pending]
            fetched = {e.id: e for e in self._repo.get_by_ids(src_col, ids_to_fetch)}
            for ex in pending:
                if ex.memory_id not in fetched:
                    continue
                entry = fetched[ex.memory_id]
                last_accessed = entry.metadata.get("last_accessed_at", 0.0)
                ex.target_future_access = (
                    1.0 if last_accessed > ex.retrieved_at + _LABEL_EPSILON else 0.0
                )

        ranker = self._ranker_factory()
        ranker.load(self._storage.weights_path(scope))
        metrics = ranker.train_step(buffer)

        if metrics.loss is not None and not math.isfinite(metrics.loss):
            # NaN/inf weights would poison every future rank() call. Skip the save
            # and surface the issue instead of corrupting the file silently.
            print(
                f"[ai-mem] train_ranker: non-finite loss {metrics.loss!r} for "
                f"collection {collection!r} — weights NOT saved",
                file=sys.stderr,
            )
        else:
            ranker.save(self._storage.weights_path(scope))

        # Drop labeled examples — they served their purpose. Keep unlabeled ones
        # (target still None) for future label windows. Without this filter the
        # buffer grows unboundedly with already-trained-on examples.
        remaining = [ex for ex in buffer if ex.target_future_access is None]
        self._storage.clear_buffer(scope)
        if remaining:
            self._storage.append_examples(scope, remaining)
        self._storage.prune_buffer(scope, self._buffer_max)

        return metrics
