"""List all memory collections."""
from __future__ import annotations

from ai_mem.domain.memory import CollectionInfo, MemoryRepository


class ListCollectionsUseCase:
    def __init__(self, repo: MemoryRepository) -> None:
        self._repo = repo

    def execute(self) -> list[CollectionInfo]:
        return self._repo.list_collections()
