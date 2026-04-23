"""ChromaDB implementation of MemoryRepository."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import chromadb

from ai_mem.domain.memory import CollectionInfo, MemoryEntry, QueryResult


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
    ) -> list[QueryResult]:
        col = self._col(collection)

        where: dict | None = None
        if max_age_days is not None:
            cutoff = datetime.now(tz=timezone.utc).timestamp() - max_age_days * 86400
            where = {"created_at": {"$gte": cutoff}}

        kwargs: dict = {"query_texts": [text], "n_results": min(n_results, col.count() or 1)}
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
        ids = result.get("ids") or []
        if ids:
            col.delete(ids=ids)
        return len(ids)
