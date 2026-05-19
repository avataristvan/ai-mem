"""Convert QueryResult list to RankingFeatures for the re-ranker."""
from __future__ import annotations

from ai_mem.domain.learning import RankingFeatures
from ai_mem.domain.memory import QueryResult

# Raw access_count is unbounded; capping prevents high-access incumbents from
# drowning out cosine similarity in the learned ranker's feature vector.
_ACCESS_COUNT_CAP = 50


class BuildFeaturesUseCase:
    def execute(
        self,
        results: list[QueryResult],
        now: float,
        session_hits: set[str] | None = None,
    ) -> list[RankingFeatures]:
        hits = session_hits or set()
        return [self._from_result(r, now, r.id in hits) for r in results]

    def _from_result(self, result: QueryResult, now: float, session_hit: bool = False) -> RankingFeatures:
        meta = result.metadata
        created_at: float = meta.get("created_at", now)
        access_count: int = min(int(meta.get("access_count", 0)), _ACCESS_COUNT_CAP)
        last_accessed_at: float = meta.get("last_accessed_at", created_at)
        expires_at: float | None = meta.get("expires_at")

        age_days = max(0.0, (now - created_at) / 86400)
        last_access_days = max(0.0, (now - last_accessed_at) / 86400)

        return RankingFeatures(
            cosine_similarity=result.score,
            age_days=age_days,
            last_access_days=last_access_days,
            access_count=access_count,
            has_ttl=expires_at is not None,
            session_hit=session_hit,
        )
