"""Return all outgoing causal edges for a memory entry."""
from __future__ import annotations

from ai_mem.domain.memory import MemoryEdge, MemoryRepository


class GetEdgesUseCase:
    def __init__(self, repo: MemoryRepository) -> None:
        self._repo = repo

    def execute(self, collection: str, entry_id: str) -> list[MemoryEdge]:
        return self._repo.get_edges(collection, entry_id)
