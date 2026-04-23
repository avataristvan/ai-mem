"""Add or update memory entries, injecting timestamps and TTL."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ai_mem.domain.memory import MemoryEntry, MemoryRepository

DEFAULT_COLLECTION = "workspace"


class AddMemoryUseCase:
    def __init__(self, repo: MemoryRepository) -> None:
        self._repo = repo

    def execute(
        self,
        collection: str,
        documents: list[str],
        ids: list[str],
        metadatas: list[dict] | None = None,
        ttl_days: float | None = None,
    ) -> int:
        now = datetime.now(tz=timezone.utc)
        expires_at = now + timedelta(days=ttl_days) if ttl_days is not None else None

        entries: list[MemoryEntry] = []
        for i, (doc, id_) in enumerate(zip(documents, ids)):
            meta = dict(metadatas[i]) if metadatas and i < len(metadatas) else {}
            meta["created_at"] = now.timestamp()
            if expires_at is not None:
                meta["expires_at"] = expires_at.timestamp()
            entries.append(
                MemoryEntry(
                    id=id_,
                    text=doc,
                    metadata=meta,
                    created_at=now,
                    expires_at=expires_at,
                )
            )

        self._repo.upsert(collection, entries)
        return len(entries)
