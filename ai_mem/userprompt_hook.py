#!/usr/bin/env python3
"""UserPromptSubmit hook — injects relevant memories when the ranker is trained enough."""
import json
import os
import sys
from pathlib import Path

from ai_mem.repo_context import GLOBAL_COLLECTION, WORKSPACE_COLLECTION, detect_repo_context

DB_PATH = Path(os.environ.get("AI_MEM_PATH", Path.home() / ".local" / "share" / "ai-mem"))
_STATS_PATH = DB_PATH / "session_stats.json"
TOP_K = 3
MAX_CHARS_PER_HIT = 600
MIN_LABELED_EXAMPLES = 10
MIN_AVG_SCORE = 0.55


def _build_deps():
    from ai_mem.application.build_features import BuildFeaturesUseCase
    from ai_mem.application.load_ranker_config import LoadRankerConfigUseCase
    from ai_mem.application.query_memory import QueryMemoryUseCase
    from ai_mem.application.ranker_registry import RankerRegistry
    from ai_mem.application.track_access import TrackAccessUseCase
    from ai_mem.application.train_ranker import TrainRankerUseCase
    from ai_mem.domain.learning import RankerScope
    from ai_mem.infrastructure.chroma_repository import ChromaMemoryRepository
    from ai_mem.infrastructure.ranker_storage import RankerStorage

    try:
        from ai_mem.infrastructure.torch_ranker import TorchMicroRanker as RankerClass
    except ImportError:
        from ai_mem.infrastructure.null_ranker import NullRanker as RankerClass  # type: ignore[assignment]

    repo = ChromaMemoryRepository(DB_PATH)
    try:
        from ai_mem.infrastructure.bm25_repository import BM25MemoryRepository
        repo = BM25MemoryRepository(repo)
    except ImportError:
        pass
    storage = RankerStorage(DB_PATH / "rankers")
    scope_map = LoadRankerConfigUseCase(DB_PATH / "ranker_config.json").execute()
    scope_resolver = lambda c: scope_map.get(c, RankerScope(name=c, mode="isolated"))
    registry = RankerRegistry(scope_resolver=scope_resolver, ranker_factory=RankerClass, storage=storage)
    query_uc = QueryMemoryUseCase(
        repo,
        TrackAccessUseCase(repo),
        BuildFeaturesUseCase(),
        TrainRankerUseCase(repo, storage, RankerClass, scope_resolver=scope_resolver),
        registry,
    )
    return query_uc, storage, registry


def _hits(query_uc, collection: str, query: str):
    try:
        return query_uc.execute(collection=collection, query=query, n_results=TOP_K)
    except Exception:
        return []


def _labeled_count(storage, registry, collection: str) -> int:
    try:
        scope_key = registry.scope_key(collection)
        return sum(1 for e in storage.load_buffer(scope_key) if e.label is not None)
    except Exception:
        return 0


def _avg_score(results) -> float:
    if not results:
        return 0.0
    top = results[:TOP_K]
    scores = [r.score for r in top if r.score is not None]
    return sum(scores) / len(scores) if scores else 0.0


def _qualifies(storage, registry, results, collection: str) -> bool:
    return (
        _labeled_count(storage, registry, collection) >= MIN_LABELED_EXAMPLES
        and _avg_score(results) >= MIN_AVG_SCORE
    )


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return

    query = payload.get("prompt", "").strip()
    if not query or not DB_PATH.exists():
        return

    try:
        query_uc, storage, registry = _build_deps()
    except Exception:
        return

    global_results = _hits(query_uc, GLOBAL_COLLECTION, query)
    global_ok = _qualifies(storage, registry, global_results, GLOBAL_COLLECTION)

    repo_results = []
    repo_collection = None
    repo_ok = False
    try:
        ctx = detect_repo_context()
        if ctx.collection not in (GLOBAL_COLLECTION, WORKSPACE_COLLECTION):
            repo_collection = ctx.collection
            repo_results = _hits(query_uc, repo_collection, query)
            repo_ok = _qualifies(storage, registry, repo_results, repo_collection)
    except Exception:
        pass

    if not global_ok and not repo_ok:
        return

    collected: list[tuple[str, object]] = []
    if global_ok:
        collected.extend((GLOBAL_COLLECTION, r) for r in global_results)
    if repo_ok and repo_collection:
        collected.extend((repo_collection, r) for r in repo_results)

    try:
        from ai_mem.session_stats import record_injection
        record_injection(_STATS_PATH, GLOBAL_COLLECTION, injected=bool(collected))
    except Exception:
        pass

    if not collected:
        return

    lines = ["[ai-mem] Relevant context for your prompt:"]
    for coll, r in collected:
        text = r.text[:MAX_CHARS_PER_HIT]
        if len(r.text) > MAX_CHARS_PER_HIT:
            text += "..."
        score = f"{r.score:.2f}" if r.score is not None else "n/a"
        lines.append(f"- [{coll} score={score}] {text}")

    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": "\n".join(lines),
                }
            }
        )
    )


if __name__ == "__main__":
    main()
