"""Record an access event for retrieved memory entries."""
from __future__ import annotations

from ai_mem.domain.memory import MemoryRepository


class TrackAccessUseCase:
    def __init__(self, repo: MemoryRepository) -> None:
        self._repo = repo

    def execute(self, collection: str, ids: list[str]) -> None:
        if not ids:
            return
        self._repo.record_access(collection, ids)
