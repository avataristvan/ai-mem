"""Fetch memory entries by id."""
from __future__ import annotations

from ai_mem.domain.memory import MemoryEntry, MemoryRepository


class GetMemoryUseCase:
    def __init__(self, repo: MemoryRepository) -> None:
        self._repo = repo

    def execute(self, collection: str, ids: list[str]) -> list[MemoryEntry]:
        return self._repo.get_by_ids(collection, ids)
