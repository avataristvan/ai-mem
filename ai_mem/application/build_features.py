"""Convert QueryResult list to RankingFeatures for the re-ranker."""
from __future__ import annotations

from ai_mem.domain.learning import RankingFeatures
from ai_mem.domain.memory import QueryResult


class BuildFeaturesUseCase:
    def execute(self, results: list[QueryResult], now: float) -> list[RankingFeatures]:
        return [self._from_result(r, now) for r in results]

    def _from_result(self, result: QueryResult, now: float) -> RankingFeatures:
        meta = result.metadata
        created_at: float = meta.get("created_at", now)
        access_count: int = int(meta.get("access_count", 0))
        last_accessed_at: float = meta.get("last_accessed_at", created_at)
        expires_at: float | None = meta.get("expires_at")

        age_days = max(0.0, (now - created_at) / 86400)
        last_access_days = max(0.0, (now - last_accessed_at) / 86400)
        has_ttl = expires_at is not None
        expires_in_days = max(0.0, (expires_at - now) / 86400) if has_ttl else 0.0

        return RankingFeatures(
            cosine_similarity=result.score,
            age_days=age_days,
            last_access_days=last_access_days,
            access_count=access_count,
            has_ttl=has_ttl,
            expires_in_days=expires_in_days,
        )
