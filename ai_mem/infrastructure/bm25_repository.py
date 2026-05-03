"""BM25 + cosine hybrid wrapper for MemoryRepository."""
from __future__ import annotations

from rank_bm25 import BM25Okapi

from ai_mem.domain.memory import CollectionInfo, MemoryEntry, QueryResult

_BM25_FETCH = 50


def _normalize(scores: list[float]) -> list[float]:
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0] * len(scores)
    span = hi - lo
    return [(s - lo) / span for s in scores]


class BM25MemoryRepository:
    """Wraps any MemoryRepository, re-ranking results with BM25+cosine fusion.

    query() fetches _BM25_FETCH candidates from the inner repo, applies BM25
    over those documents, normalises both score sets to [0,1], and combines
    them as: hybrid = alpha * cosine_norm + (1-alpha) * bm25_norm.
    """

    def __init__(self, inner, alpha: float = 0.5) -> None:
        self._inner = inner
        self._alpha = alpha

    def query(
        self,
        collection: str,
        text: str,
        n_results: int,
        max_age_days: float | None,
        type_filter: str | None = None,
    ) -> list[QueryResult]:
        candidates = self._inner.query(collection, text, _BM25_FETCH, max_age_days, type_filter)
        if not candidates:
            return []

        corpus = [r.text.lower().split() for r in candidates]
        query_tokens = text.lower().split()

        try:
            bm25_raw = BM25Okapi(corpus).get_scores(query_tokens).tolist()
        except ZeroDivisionError:
            bm25_raw = [0.0] * len(candidates)

        cosine_raw = [r.score for r in candidates]

        cosine_norm = _normalize(cosine_raw)
        bm25_norm = _normalize(bm25_raw)

        alpha = self._alpha
        fused = [
            alpha * c + (1 - alpha) * b
            for c, b in zip(cosine_norm, bm25_norm)
        ]

        ranked = sorted(
            zip(fused, candidates),
            key=lambda pair: pair[0],
            reverse=True,
        )

        out = []
        for rank, (score, result) in enumerate(ranked[:n_results], start=1):
            out.append(
                QueryResult(
                    rank=rank,
                    id=result.id,
                    score=round(score, 4),
                    text=result.text,
                    metadata=result.metadata,
                )
            )
        return out

    def upsert(self, collection: str, entries: list[MemoryEntry]) -> None:
        self._inner.upsert(collection, entries)

    def get_by_ids(self, collection: str, ids: list[str]) -> list[MemoryEntry]:
        return self._inner.get_by_ids(collection, ids)

    def list_collections(self) -> list[CollectionInfo]:
        return self._inner.list_collections()

    def delete(self, collection: str, ids: list[str] | None) -> int:
        return self._inner.delete(collection, ids)

    def delete_expired(self, collection: str) -> int:
        return self._inner.delete_expired(collection)

    def record_access(self, collection: str, ids: list[str]) -> None:
        self._inner.record_access(collection, ids)

    def get_all(self, collection: str) -> list[MemoryEntry]:
        return self._inner.get_all(collection)

    def delete_stale(self, collection: str, stale_after_days: float) -> int:
        return self._inner.delete_stale(collection, stale_after_days)
