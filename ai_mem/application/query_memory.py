"""Query memory semantically; records access for the confidence lifecycle."""
from __future__ import annotations

from collections import defaultdict

from ai_mem.application.track_access import TrackAccessUseCase
from ai_mem.domain.memory import MemoryRepository, QueryResult


class QueryMemoryUseCase:
    def __init__(
        self,
        repo: MemoryRepository,
        track_access: TrackAccessUseCase,
    ) -> None:
        self._repo = repo
        self._track_access = track_access
        self._session_hits: dict[str, set[str]] = defaultdict(set)

    def execute(
        self,
        collection: str,
        query: str,
        n_results: int = 5,
        max_age_days: float | None = None,
        type_filter: str | None = None,
        min_score_for_tracking: float | None = None,
    ) -> list[QueryResult]:
        results = self._repo.query(collection, query, n_results, max_age_days, type_filter)
        if not results:
            return []

        for idx, result in enumerate(results):
            result.rank = idx + 1

        returned_ids = {result.id for result in results}
        self._session_hits[collection].update(returned_ids)

        # access_count means deliberate use, not passive surfacing (see
        # arch_decision_push_vs_pull_2026_07_31 chunk 3) -- a result merely appearing in a
        # semantic search does not, by itself, count as access. min_score_for_tracking is an
        # explicit opt-in for callers with a real usage signal beyond "it matched": posttool_hook
        # passes a threshold because a strong match to the file just edited is itself evidence of
        # relevance, distinct from a query merely returning a candidate. Without a threshold, no
        # tracking happens here at all -- deliberate retrieval is tracked by GetMemoryUseCase
        # (mem_get by id) and BoostConfidenceUseCase (mem_boost, the /reflect citation signal).
        if min_score_for_tracking is not None:
            tracked_ids = {result.id for result in results if result.score >= min_score_for_tracking}
            self._track_access.execute(collection, list(tracked_ids))

        results = self._append_linked(collection, results, returned_ids)
        return results

    def _append_linked(
        self, collection: str, results: list[QueryResult], result_ids: set[str]
    ) -> list[QueryResult]:
        """Follow 1-hop edges from result entries and append linked entries (budget: 2)."""
        linked: list[QueryResult] = []
        try:
            for result in results:
                if len(linked) >= 2:
                    break
                edges = self._repo.get_edges(collection, result.id)
                for edge in edges:
                    if len(linked) >= 2:
                        break
                    if edge.target_id in result_ids:
                        continue
                    entries = self._repo.get_by_ids(collection, [edge.target_id])
                    if not entries:
                        continue
                    entry = entries[0]
                    linked_meta = dict(entry.metadata)
                    linked_meta["via_edge"] = edge.edge_type
                    linked_meta["via_source"] = result.id
                    linked.append(
                        QueryResult(
                            rank=len(results) + len(linked) + 1,
                            id=entry.id,
                            score=0.0,
                            text=entry.text,
                            metadata=linked_meta,
                        )
                    )
                    result_ids.add(edge.target_id)
        except Exception:
            pass
        return results + linked
