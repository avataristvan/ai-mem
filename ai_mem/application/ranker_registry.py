"""Maps collection names to their loaded LearnedRanker instance."""
from __future__ import annotations

from typing import Callable

from ai_mem.domain.learning import LearnedRanker, RankerScope, TrainingBufferRepository

# Minimum labeled examples before a learned ranker is trusted over the fallback.
# Mirrors the gate used in userprompt_hook.py — keep in sync if changed.
MIN_LABELED_EXAMPLES = 10


class RankerRegistry:
    """Lazy-loads and caches one ranker per scope key.

    Isolated collections use the collection name as key; hybrid-mode
    collections share a key equal to their group name. This ensures that
    collections in the same group share a single trained ranker instance.

    When *fallback_factory* is provided (typically NullRanker) and the trained
    ranker has fewer than *min_labeled* labeled examples, the fallback is
    returned instead of the untrained learned ranker — random MLP weights are
    worse than the NullRanker UCB heuristic.
    """

    def __init__(
        self,
        scope_resolver: Callable[[str], RankerScope],
        ranker_factory: Callable[[], LearnedRanker],
        storage: TrainingBufferRepository,
        fallback_factory: Callable[[], LearnedRanker] | None = None,
        min_labeled: int = MIN_LABELED_EXAMPLES,
    ) -> None:
        self._scope_resolver = scope_resolver
        self._ranker_factory = ranker_factory
        self._storage = storage
        self._fallback_factory = fallback_factory
        self._min_labeled = min_labeled
        self._cache: dict[str, LearnedRanker] = {}

    def scope_key(self, collection: str) -> str:
        s = self._scope_resolver(collection)
        return s.group if s.mode == "hybrid" else s.name

    def get(self, collection: str) -> LearnedRanker:
        key = self.scope_key(collection)
        if self._fallback_factory is not None:
            if self._storage.labeled_count(key) < self._min_labeled:
                return self._fallback_factory()
        if key not in self._cache:
            ranker = self._ranker_factory()
            ranker.load(self._storage.weights_path(key))
            self._cache[key] = ranker
        return self._cache[key]
