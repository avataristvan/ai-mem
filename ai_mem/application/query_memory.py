"""Query memory semantically with optional age filter."""
from __future__ import annotations

from ai_mem.domain.memory import MemoryRepository, QueryResult


class QueryMemoryUseCase:
    def __init__(self, repo: MemoryRepository) -> None:
        self._repo = repo

    def execute(
        self,
        collection: str,
        query: str,
        n_results: int = 5,
        max_age_days: float | None = None,
    ) -> list[QueryResult]:
        return self._repo.query(collection, query, n_results, max_age_days)
