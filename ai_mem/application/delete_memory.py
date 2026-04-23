"""Delete entries or entire collections from memory."""
from __future__ import annotations

from ai_mem.domain.memory import MemoryRepository


class DeleteMemoryUseCase:
    def __init__(self, repo: MemoryRepository) -> None:
        self._repo = repo

    def execute(self, collection: str, ids: list[str] | None) -> int:
        return self._repo.delete(collection, ids)
