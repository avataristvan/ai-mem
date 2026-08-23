"""ChromaDB implementation of MemoryRepository."""
from __future__ import annotations

import json
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from ai_mem.domain.memory import CollectionInfo, MemoryEdge, MemoryEntry, QueryResult


_PERMANENT_TYPES = {"pattern", "anti-pattern"}
_EMBEDDING_CACHE_MAXSIZE = 128


def _parse_edges(raw: str) -> list[MemoryEdge]:
    """Parse a JSON-encoded edge list from metadata. Returns [] on any parse error."""
    try:
        items = json.loads(raw)
        return [MemoryEdge(target_id=item["target_id"], edge_type=item["edge_type"]) for item in items]
    except Exception:
        return []


def _exclude_patterns(result: dict) -> list[str]:
    """Return IDs from a ChromaDB result, excluding permanent entries (pattern, anti-pattern)."""
    ids = result.get("ids") or []
    metas = result.get("metadatas") or [{}] * len(ids)
    return [id_ for id_, meta in zip(ids, metas) if (meta or {}).get("type") not in _PERMANENT_TYPES]


class _EmbeddingCache:
    """LRU-bounded cache of query text -> embedding vector, with per-key locking.

    Embedding a query costs ~150-250ms (ONNX inference) regardless of how "warm" the
    process is -- it's per-call cost, not a one-time model-load cost. userprompt_hook's
    4-way fan-out (global + repo context + antipattern + dilemma) queries the exact same
    prompt text up to 4 times, so without this cache the same text gets embedded up to
    4x per hook call. Cached by text alone, not (collection, text): every collection in
    this codebase uses the same DefaultEmbeddingFunction (verified: no collection is ever
    created with an explicit embedding_function), so the same text always embeds to the
    same vector regardless of which collection it's queried against.

    Per-key locking (not one lock guarding the whole cache) means only genuinely-duplicate
    in-flight requests for the *same* text block each other -- concurrent requests for
    different text proceed fully in parallel, same as the daemon's per-connection threading.
    The first thread for a given text computes and caches; others waiting on that text's
    lock reuse the result instead of recomputing.
    """

    def __init__(self, maxsize: int = _EMBEDDING_CACHE_MAXSIZE) -> None:
        self._maxsize = maxsize
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._guard = threading.Lock()
        self._pending: dict[str, threading.Lock] = {}

    def get_or_compute(self, text, compute) -> list[float]:
        with self._guard:
            cached = self._cache.get(text)
            if cached is not None:
                self._cache.move_to_end(text)
                return cached
            key_lock = self._pending.setdefault(text, threading.Lock())

        with key_lock:
            with self._guard:
                cached = self._cache.get(text)
                if cached is not None:  # another thread computed it while we waited
                    self._cache.move_to_end(text)
                    return cached

            try:
                vector = compute(text)
            finally:
                with self._guard:
                    self._pending.pop(text, None)

            with self._guard:
                self._cache[text] = vector
                self._cache.move_to_end(text)
                if len(self._cache) > self._maxsize:
                    self._cache.popitem(last=False)

        return vector


class ChromaMemoryRepository:
    def __init__(self, db_path: Path) -> None:
        db_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(db_path))
        # record_access does a read-modify-write (get, bump access_count, update) that races
        # when the same entry is hit by concurrent callers (e.g. the daemon serving multiple
        # connections at once) -- confirmed empirically: 8 threads x 20 unlocked increments on
        # the same id landed at ~32 instead of 160. Only this method needs the lock; the daemon
        # never calls upsert/delete, and query()/get_by_ids() are pure reads.
        self._write_lock = threading.Lock()
        # Own embedding function instance rather than reaching into a collection's private
        # `_embedding_function` -- constructing DefaultEmbeddingFunction() is near-free (model
        # load is lazy, happens on first __call__, and isn't duplicated per instance; confirmed
        # empirically: a second instance's first call costs the same as any other call, not a
        # second cold-load). This lets query() request an embedding before touching a specific
        # collection at all, which is what makes the cache collection-agnostic.
        self._embedding_function = DefaultEmbeddingFunction()
        self._embedding_cache = _EmbeddingCache()

    def _col(self, name: str):
        return self._client.get_or_create_collection(name)

    def _safe_get_collection(self, name: str):
        """Return the named collection, or None if it does not exist."""
        try:
            return self._client.get_collection(name)
        except Exception:
            return None

    def upsert(self, collection: str, entries: list[MemoryEntry]) -> None:
        col = self._col(collection)
        col.upsert(
            documents=[entry.text for entry in entries],
            ids=[entry.id for entry in entries],
            metadatas=[entry.metadata for entry in entries],
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

        embedding = self._embedding_cache.get_or_compute(text, lambda t: self._embedding_function([t])[0])
        kwargs: dict = {"query_embeddings": [embedding], "n_results": min(n_results, count)}
        if where:
            kwargs["where"] = where

        results = col.query(**kwargs)
        docs = results["documents"][0]
        ids = results["ids"][0]
        metas = (results.get("metadatas") or [[]])[0] or [{}] * len(docs)
        distances = (results.get("distances") or [[]])[0]

        return [
            QueryResult(
                rank=idx + 1,
                id=id_,
                # get_or_create_collection defaults to `l2` (squared L2) space, not cosine.
                # For normalized embeddings, dist = 2 - 2*cos_sim, so cos_sim = 1 - dist/2.
                score=round(1.0 - dist / 2.0, 4),
                text=doc,
                metadata=meta or {},
            )
            for idx, (doc, id_, meta, dist) in enumerate(zip(docs, ids, metas, distances))
        ]

    def get_by_ids(self, collection: str, ids: list[str]) -> list[MemoryEntry]:
        col = self._safe_get_collection(collection)
        if col is None:
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
        return [CollectionInfo(name=col_info.name, count=col_info.count()) for col_info in self._client.list_collections()]

    def delete(self, collection: str, ids: list[str] | None) -> int:
        if ids:
            col = self._col(collection)
            col.delete(ids=ids)
            return len(ids)
        self._client.delete_collection(collection)
        return -1  # whole collection dropped

    def delete_expired(self, collection: str) -> int:
        col = self._safe_get_collection(collection)
        if col is None:
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
        col = self._safe_get_collection(collection)
        if col is None:
            return

        with self._write_lock:
            existing = col.get(ids=ids)
            existing_ids = existing.get("ids") or []
            existing_metas = existing.get("metadatas") or []
            if not existing_ids:
                return

            now_ts = datetime.now(tz=timezone.utc).timestamp()
            new_metas = []
            for meta in existing_metas:
                meta_copy = dict(meta or {})
                meta_copy["last_accessed_at"] = now_ts
                meta_copy["access_count"] = int(meta_copy.get("access_count", 0)) + 1
                new_metas.append(meta_copy)

            col.update(ids=existing_ids, metadatas=new_metas)

    def get_all(self, collection: str) -> list[MemoryEntry]:
        col = self._safe_get_collection(collection)
        if col is None:
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
        col = self._safe_get_collection(collection)
        if col is None:
            return 0

        cutoff = datetime.now(tz=timezone.utc).timestamp() - stale_after_days * 86400
        result = col.get(where={"last_accessed_at": {"$lt": cutoff}})
        ids = _exclude_patterns(result)
        if ids:
            col.delete(ids=ids)
        return len(ids)

    def add_edge(self, collection: str, source_id: str, edge: MemoryEdge) -> None:
        col = self._safe_get_collection(collection)
        if col is None:
            return

        result = col.get(ids=[source_id])
        if not result.get("ids"):
            return

        existing_meta = dict((result.get("metadatas") or [{}])[0] or {})
        edges = _parse_edges(existing_meta.get("edges", "[]"))

        # Dedup: skip if the same (target_id, edge_type) already exists
        for existing_edge in edges:
            if existing_edge.target_id == edge.target_id and existing_edge.edge_type == edge.edge_type:
                return

        edges.append(edge)
        existing_meta["edges"] = json.dumps(
            [{"target_id": existing_edge.target_id, "edge_type": existing_edge.edge_type} for existing_edge in edges]
        )
        col.update(ids=[source_id], metadatas=[existing_meta])

    def get_edges(self, collection: str, entry_id: str) -> list[MemoryEdge]:
        col = self._safe_get_collection(collection)
        if col is None:
            return []

        result = col.get(ids=[entry_id])
        if not result.get("ids"):
            return []

        meta = (result.get("metadatas") or [{}])[0] or {}
        return _parse_edges(meta.get("edges", "[]"))
