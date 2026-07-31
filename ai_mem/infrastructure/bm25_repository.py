"""BM25 + cosine hybrid wrapper for MemoryRepository."""
from __future__ import annotations

from rank_bm25 import BM25Okapi

from ai_mem.domain.memory import CollectionInfo, MemoryEdge, MemoryEntry, QueryResult

_BM25_FETCH = 50
# Empirically derived from the live-DB median nonzero raw BM25 score (~1.6), not a guess —
# see ai-mem entry bm25_saturating_transform_research_decision_2026_07_31 for the full derivation.
_BM25_SATURATION_K = 2.0


def _saturate(scores: list[float]) -> list[float]:
    # rank_bm25 floors negative IDFs at 0.25 * average_idf, which is itself negative for
    # small/near-duplicate corpora — so raw BM25 scores can be negative. raw/(raw+k) is only
    # monotonic on one side of its pole at raw=-k (and divides by zero exactly there), so
    # clamp to 0 first: a non-positive raw score means "no real match," which should saturate
    # to 0 rather than risk crossing the pole and inverting rank order.
    return [max(s, 0.0) / (max(s, 0.0) + _BM25_SATURATION_K) for s in scores]


class BM25MemoryRepository:
    """Wraps any MemoryRepository, re-ranking results with BM25+cosine fusion.

    query() fetches _BM25_FETCH candidates from the inner repo, applies BM25
    over those documents, and passes the raw BM25 scores through a saturating
    transform (cosine is already a meaningful absolute scale and is used as-is),
    combining them as: hybrid = alpha * cosine_raw + (1-alpha) * bm25_sat.
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

        corpus = [candidate.text.lower().split() for candidate in candidates]
        query_tokens = text.lower().split()

        try:
            bm25_raw = BM25Okapi(corpus).get_scores(query_tokens).tolist()
        except ZeroDivisionError:
            bm25_raw = [0.0] * len(candidates)

        # cosine is already a fixed, meaningful similarity scale (post score-scaling fix in
        # chroma_repository.py) — min-max normalizing it here would throw that away and replace
        # it with a purely rank-relative score again. BM25's raw score is unbounded and
        # corpus/query-length dependent, so it goes through a saturating transform
        # (raw / (raw + k)) instead: an absolute, pool-independent mapping that depends only
        # on the candidate's own raw score, not on what else is in the pool — same rationale
        # as the cosine-side fix, just applied to BM25's half of the fusion.
        cosine_raw = [candidate.score for candidate in candidates]
        bm25_norm = _saturate(bm25_raw)

        alpha = self._alpha
        fused = [
            alpha * cosine_val + (1 - alpha) * bm25_val
            for cosine_val, bm25_val in zip(cosine_raw, bm25_norm)
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

    def add_edge(self, collection: str, source_id: str, edge: MemoryEdge) -> None:
        self._inner.add_edge(collection, source_id, edge)

    def get_edges(self, collection: str, entry_id: str) -> list[MemoryEdge]:
        return self._inner.get_edges(collection, entry_id)
