"""Domain layer: entities and repository interface."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol


@dataclass
class MemoryEntry:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    # injected by AddMemoryUseCase; stored in metadata
    created_at: datetime | None = None
    expires_at: datetime | None = None


@dataclass
class QueryResult:
    rank: int
    id: str
    score: float
    text: str
    metadata: dict


@dataclass
class CollectionInfo:
    name: str
    count: int


class MemoryRepository(Protocol):
    def upsert(self, collection: str, entries: list[MemoryEntry]) -> None: ...

    def query(
        self,
        collection: str,
        text: str,
        n_results: int,
        max_age_days: float | None,
    ) -> list[QueryResult]: ...

    def list_collections(self) -> list[CollectionInfo]: ...

    def delete(self, collection: str, ids: list[str] | None) -> int:
        """Delete entries by id, or drop entire collection. Returns affected count."""
        ...

    def delete_expired(self, collection: str) -> int:
        """Remove all entries whose expires_at is in the past. Returns count."""
        ...
