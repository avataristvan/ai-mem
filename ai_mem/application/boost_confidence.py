"""Boost or decay the confidence score on existing memory entries."""
from __future__ import annotations

from ai_mem.domain.memory import MemoryEntry, MemoryRepository


class BoostConfidenceUseCase:
    def __init__(self, repo: MemoryRepository) -> None:
        self._repo = repo

    def execute(self, collection: str, ids: list[str], delta: float) -> int:
        """Apply delta to confidence on existing entries. Returns count of updated entries.

        Entries not found in the collection are silently skipped.
        Delta is clamped so the result stays in [0.0, 1.0].
        """
        existing = self._repo.get_by_ids(collection, ids)
        if not existing:
            return 0

        updated: list[MemoryEntry] = []
        for entry in existing:
            prior = float(entry.metadata.get("confidence", 1.0))
            new_confidence = max(0.0, min(1.0, prior + delta))
            entry.metadata["confidence"] = new_confidence
            updated.append(entry)

        self._repo.upsert(collection, updated)
        return len(updated)
