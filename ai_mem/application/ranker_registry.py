"""Maps collection names to their loaded LearnedRanker instance."""
from __future__ import annotations

from typing import Callable

from ai_mem.domain.learning import LearnedRanker, RankerScope, TrainingBufferRepository


class RankerRegistry:
    """Lazy-loads and caches one ranker per scope key.

    Isolated collections use the collection name as key; hybrid-mode
    collections share a key equal to their group name. This ensures that
    collections in the same group share a single trained ranker instance.
    """

    def __init__(
        self,
        scope_resolver: Callable[[str], RankerScope],
        ranker_factory: Callable[[], LearnedRanker],
        storage: TrainingBufferRepository,
    ) -> None:
        self._scope_resolver = scope_resolver
        self._ranker_factory = ranker_factory
        self._storage = storage
        self._cache: dict[str, LearnedRanker] = {}

    def scope_key(self, collection: str) -> str:
        s = self._scope_resolver(collection)
        return s.group if s.mode == "hybrid" else s.name

    def get(self, collection: str) -> LearnedRanker:
        key = self.scope_key(collection)
        if key not in self._cache:
            ranker = self._ranker_factory()
            ranker.load(self._storage.weights_path(key))
            self._cache[key] = ranker
        return self._cache[key]
