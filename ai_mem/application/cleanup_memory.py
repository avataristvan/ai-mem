"""Remove expired entries from a collection."""
from __future__ import annotations

from ai_mem.domain.memory import CollectionInfo, MemoryRepository


class CleanupMemoryUseCase:
    def __init__(self, repo: MemoryRepository) -> None:
        self._repo = repo

    def execute(self, collection: str | None) -> dict[str, int]:
        """Delete expired entries. If collection is None, clean all collections."""
        if collection:
            targets = [collection]
        else:
            targets = [c.name for c in self._repo.list_collections()]

        return {col: self._repo.delete_expired(col) for col in targets}
