"""Domain layer: entities and repository interface."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol

EdgeType = Literal["contradicts", "fixes", "causes", "related"]


@dataclass
class MemoryEdge:
    target_id: str
    edge_type: EdgeType


@dataclass
class MemoryEntry:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    # injected by AddMemoryUseCase / TrackAccessUseCase; canonical values live in metadata
    created_at: datetime | None = None
    expires_at: datetime | None = None
    last_accessed_at: datetime | None = None
    access_count: int = 0


@dataclass
class QueryResult:
    rank: int
    id: str
    score: float
    text: str
    metadata: dict


@dataclass
class SplitHint:
    id: str
    text_preview: str   # first 80 chars of the original text
    access_count: int


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
        type_filter: str | None = None,
    ) -> list[QueryResult]: ...

    def get_by_ids(self, collection: str, ids: list[str]) -> list[MemoryEntry]:
        """Fetch entries by id. Missing ids are silently skipped."""
        ...

    def list_collections(self) -> list[CollectionInfo]: ...

    def delete(self, collection: str, ids: list[str] | None) -> int:
        """Delete entries by id, or drop entire collection. Returns affected count."""
        ...

    def delete_expired(self, collection: str) -> int:
        """Remove all entries whose expires_at is in the past. Returns count."""
        ...

    def record_access(self, collection: str, ids: list[str]) -> None:
        """Bump last_accessed_at and access_count for the given ids. Missing ids are silently skipped."""
        ...

    def delete_stale(self, collection: str, stale_after_days: float) -> int:
        """Remove entries whose last_accessed_at is older than the cutoff. Returns count."""
        ...

    def get_all(self, collection: str) -> list[MemoryEntry]:
        """Return every entry in a collection (no filtering, no ranking)."""
        ...

    def add_edge(self, collection: str, source_id: str, edge: MemoryEdge) -> None:
        """Append a causal edge to source_id's metadata. Idempotent on same (target_id, edge_type)."""
        ...

    def get_edges(self, collection: str, entry_id: str) -> list[MemoryEdge]:
        """Return all outgoing edges for entry_id. Empty list when none exist or entry missing."""
        ...
