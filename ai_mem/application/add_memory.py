"""Add or update memory entries, injecting timestamps and TTL.

Re-adding an entry with an existing id preserves its history (created_at,
last_accessed_at, access_count). Only the document text, custom metadata, and
TTL are updated. This keeps the access-tracking signal intact across mem_add
calls that overwrite the same id."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ai_mem.domain.memory import MemoryEntry, MemoryRepository


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
        now_ts = now.timestamp()
        expires_at = now + timedelta(days=ttl_days) if ttl_days is not None else None

        existing = {e.id: e for e in self._repo.get_by_ids(collection, ids)}

        entries: list[MemoryEntry] = []
        for i, (doc, id_) in enumerate(zip(documents, ids)):
            user_meta = dict(metadatas[i]) if metadatas and i < len(metadatas) else {}
            prior_meta = existing[id_].metadata if id_ in existing else {}

            meta = {**prior_meta, **user_meta}
            meta["created_at"] = prior_meta.get("created_at", now_ts)
            meta["last_accessed_at"] = prior_meta.get("last_accessed_at", now_ts)
            meta["access_count"] = int(prior_meta.get("access_count", 0))
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
