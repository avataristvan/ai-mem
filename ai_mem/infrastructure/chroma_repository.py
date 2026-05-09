"""ChromaDB implementation of MemoryRepository."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import chromadb

from ai_mem.domain.memory import CollectionInfo, MemoryEntry, QueryResult


def _exclude_patterns(result: dict) -> list[str]:
    """Return IDs from a ChromaDB result, excluding entries with type='pattern'."""
    ids = result.get("ids") or []
    metas = result.get("metadatas") or [{}] * len(ids)
    return [id_ for id_, meta in zip(ids, metas) if (meta or {}).get("type") != "pattern"]


class ChromaMemoryRepository:
    def __init__(self, db_path: Path) -> None:
        db_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(db_path))

    def _col(self, name: str):
        return self._client.get_or_create_collection(name)

    def upsert(self, collection: str, entries: list[MemoryEntry]) -> None:
        col = self._col(collection)
        col.upsert(
            documents=[e.text for e in entries],
            ids=[e.id for e in entries],
            metadatas=[e.metadata for e in entries],
        )

    def query(
        self,
        collection: str,
        text: str,
        n_results: int,
        max_age_days: float | None,
        type_filter: str | None = None,
    ) -> list[QueryResult]:
        col = self._col(collection)
        count = col.count()
        if count == 0:
            return []

        conditions: list[dict] = []
        if max_age_days is not None:
            cutoff = datetime.now(tz=timezone.utc).timestamp() - max_age_days * 86400
            conditions.append({"created_at": {"$gte": cutoff}})
        if type_filter is not None:
            conditions.append({"type": {"$eq": type_filter}})

        where: dict | None = None
        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        kwargs: dict = {"query_texts": [text], "n_results": min(n_results, count)}
        if where:
            kwargs["where"] = where

        results = col.query(**kwargs)
        docs = results["documents"][0]
        ids = results["ids"][0]
        metas = (results.get("metadatas") or [[]])[0] or [{}] * len(docs)
        distances = (results.get("distances") or [[]])[0]

        return [
            QueryResult(
                rank=i + 1,
                id=id_,
                score=round(1.0 - dist, 4),
                text=doc,
                metadata=meta or {},
            )
            for i, (doc, id_, meta, dist) in enumerate(zip(docs, ids, metas, distances))
        ]

    def get_by_ids(self, collection: str, ids: list[str]) -> list[MemoryEntry]:
        try:
            col = self._client.get_collection(collection)
        except Exception:
            return []

        result = col.get(ids=ids)
        result_ids = result.get("ids") or []
        result_docs = result.get("documents") or []
        result_metas = result.get("metadatas") or [{}] * len(result_ids)

        return [
            MemoryEntry(id=id_, text=doc, metadata=meta or {})
            for id_, doc, meta in zip(result_ids, result_docs, result_metas)
        ]

    def list_collections(self) -> list[CollectionInfo]:
        return [CollectionInfo(name=c.name, count=c.count()) for c in self._client.list_collections()]

    def delete(self, collection: str, ids: list[str] | None) -> int:
        if ids:
            col = self._col(collection)
            col.delete(ids=ids)
            return len(ids)
        self._client.delete_collection(collection)
        return -1  # whole collection dropped

    def delete_expired(self, collection: str) -> int:
        try:
            col = self._client.get_collection(collection)
        except Exception:
            return 0

        now_ts = datetime.now(tz=timezone.utc).timestamp()
        result = col.get(where={"expires_at": {"$lte": now_ts}})
        ids = _exclude_patterns(result)
        if ids:
            col.delete(ids=ids)
        return len(ids)

    def record_access(self, collection: str, ids: list[str]) -> None:
        if not ids:
            return
        try:
            col = self._client.get_collection(collection)
        except Exception:
            return

        existing = col.get(ids=ids)
        existing_ids = existing.get("ids") or []
        existing_metas = existing.get("metadatas") or []
        if not existing_ids:
            return

        now_ts = datetime.now(tz=timezone.utc).timestamp()
        new_metas = []
        for meta in existing_metas:
            m = dict(meta or {})
            m["last_accessed_at"] = now_ts
            m["access_count"] = int(m.get("access_count", 0)) + 1
            new_metas.append(m)

        col.update(ids=existing_ids, metadatas=new_metas)

    def get_all(self, collection: str) -> list[MemoryEntry]:
        try:
            col = self._client.get_collection(collection)
        except Exception:
            return []
        result = col.get()
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or [{}] * len(ids)
        return [
            MemoryEntry(id=id_, text=doc, metadata=meta or {})
            for id_, doc, meta in zip(ids, docs, metas)
        ]

    def delete_stale(self, collection: str, stale_after_days: float) -> int:
        try:
            col = self._client.get_collection(collection)
        except Exception:
            return 0

        cutoff = datetime.now(tz=timezone.utc).timestamp() - stale_after_days * 86400
        result = col.get(where={"last_accessed_at": {"$lt": cutoff}})
        ids = _exclude_patterns(result)
        if ids:
            col.delete(ids=ids)
        return len(ids)
