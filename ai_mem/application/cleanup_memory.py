"""Remove expired and optionally stale (never-or-rarely-accessed) entries."""
from __future__ import annotations

from dataclasses import dataclass

from ai_mem.domain.memory import MemoryRepository


@dataclass
class CollectionCleanupStats:
    expired: int
    stale: int = 0

    @property
    def total(self) -> int:
        return self.expired + self.stale


@dataclass
class CleanupResult:
    collections: dict[str, CollectionCleanupStats]

    @property
    def total(self) -> int:
        return sum(s.total for s in self.collections.values())


class CleanupMemoryUseCase:
    def __init__(self, repo: MemoryRepository) -> None:
        self._repo = repo

    def execute(
        self,
        collection: str | None,
        stale_after_days: float | None = None,
    ) -> CleanupResult:
        """Run cleanup. Returns per-collection stats.

        - expired: entries past their TTL (always run)
        - stale: entries whose last_accessed_at is older than stale_after_days (only if set)
        """
        targets = [collection] if collection else [c.name for c in self._repo.list_collections()]

        collections: dict[str, CollectionCleanupStats] = {}
        for col in targets:
            expired = self._repo.delete_expired(col)
            stale = self._repo.delete_stale(col, stale_after_days) if stale_after_days is not None else 0
            collections[col] = CollectionCleanupStats(expired=expired, stale=stale)
        return CleanupResult(collections=collections)
