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
MAX_CHARS_PER_HIT = 300
MAX_TOTAL_CHARS = 1500
MIN_LABELED_EXAMPLES = 10
MIN_AVG_SCORE = 0.55
SESSION_TTL_HOURS = 4
ANTIPATTERN_TOP_K = 2
MAX_CHARS_PER_ANTIPATTERN = 200
ANTIPATTERN_MIN_SCORE = 0.4
DILEMMA_TOP_K = 2
MAX_CHARS_PER_DILEMMA = 250
DILEMMA_MIN_SCORE = 0.4
ANTICIPATION_QUESTION = "  → Anticipation: Who holds the same role now? What would happen to them?"


def _load_session_injected(db_path: Path) -> set[str]:
    """Return the set of entry IDs already injected this session.

    Returns an empty set if the file is absent, unreadable, or older than SESSION_TTL_HOURS.
    """
    import time

    path = db_path / "session_injected.json"
    try:
        data = json.loads(path.read_text())
        age_hours = (time.time() - data["session_ts"]) / 3600
        if age_hours > SESSION_TTL_HOURS:
            return set()
        return set(data.get("ids", []))
    except Exception:
        return set()


def _save_session_injected(db_path: Path, ids: set[str]) -> None:
    """Persist the current session's injected IDs; silently ignores I/O errors."""
    import time

    path = db_path / "session_injected.json"
    try:
        existing = _load_session_injected(db_path)
        merged = existing | ids
        ts = time.time()
        # Preserve the original session_ts if the file is still valid.
        try:
            data = json.loads(path.read_text())
            age_hours = (time.time() - data["session_ts"]) / 3600
            if age_hours <= SESSION_TTL_HOURS:
                ts = data["session_ts"]
        except Exception:
            pass
        path.write_text(json.dumps({"session_ts": ts, "ids": list(merged)}))
    except Exception:
        pass


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


def _antipattern_hits(query_uc, collection: str, query: str):
    try:
        results = query_uc.execute(
            collection=collection,
            query=query,
            n_results=ANTIPATTERN_TOP_K,
            type_filter="anti-pattern",
        )
        return [r for r in results if (r.score or 0.0) >= ANTIPATTERN_MIN_SCORE]
    except Exception:
        return []


def _dilemma_hits(query_uc, collection: str, query: str):
    try:
        results = query_uc.execute(
            collection=collection,
            query=query,
            n_results=DILEMMA_TOP_K,
            type_filter="dilemma",
        )
        return [r for r in results if (r.score or 0.0) >= DILEMMA_MIN_SCORE]
    except Exception:
        return []


def _labeled_count(storage, registry, collection: str) -> int:
    try:
        scope_key = registry.scope_key(collection)
        return storage.labeled_count(scope_key)
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

    try:
        from ai_mem.agent_context import detect_for_hook
        if not detect_for_hook(payload).should_inject:
            return
    except Exception:
        pass

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
    antipattern_results = []
    dilemma_results = []
    try:
        ctx = detect_repo_context()
        if ctx.collection not in (GLOBAL_COLLECTION, WORKSPACE_COLLECTION):
            repo_collection = ctx.collection
            repo_results = _hits(query_uc, repo_collection, query)
            repo_ok = _qualifies(storage, registry, repo_results, repo_collection)
            antipattern_results = _antipattern_hits(query_uc, repo_collection, query)
            dilemma_results = _dilemma_hits(query_uc, repo_collection, query)
    except Exception:
        pass

    if not global_ok and not repo_ok and not antipattern_results and not dilemma_results:
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

    if not collected and not antipattern_results and not dilemma_results:
        return

    # Per-session dedup: skip entries already injected in this session.
    already_injected = _load_session_injected(DB_PATH)
    collected = [(coll, r) for coll, r in collected if getattr(r, "id", None) not in already_injected or getattr(r, "id", None) is None]
    antipattern_results = [r for r in antipattern_results if getattr(r, "id", None) not in already_injected]
    dilemma_results = [r for r in dilemma_results if getattr(r, "id", None) not in already_injected]

    # Combined budget cap: include entries until MAX_TOTAL_CHARS is reached.
    budget_collected: list[tuple[str, object]] = []
    chars_used = 0
    for coll, r in collected:
        entry_len = min(len(r.text), MAX_CHARS_PER_HIT)
        if chars_used + entry_len > MAX_TOTAL_CHARS:
            break
        budget_collected.append((coll, r))
        chars_used += entry_len
    collected = budget_collected

    if not collected and not antipattern_results and not dilemma_results:
        return

    lines: list[str] = []
    injected_ids: set[str] = set()

    if dilemma_results:
        lines.append("[ai-mem dilemmas]")
        for r in dilemma_results:
            text = r.text[:MAX_CHARS_PER_DILEMMA]
            if len(r.text) > MAX_CHARS_PER_DILEMMA:
                text += "..."
            lines.append(f"⚖ {text}")
            entry_id = getattr(r, "id", None)
            if entry_id is not None:
                injected_ids.add(entry_id)

    if antipattern_results:
        if lines:
            lines.append("")
        lines.append("[ai-mem warnings]")
        for r in antipattern_results:
            text = r.text[:MAX_CHARS_PER_ANTIPATTERN]
            if len(r.text) > MAX_CHARS_PER_ANTIPATTERN:
                text += "..."
            lines.append(f"⚠ {text}")
            if "Affected:" in r.text:
                lines.append(ANTICIPATION_QUESTION)
            entry_id = getattr(r, "id", None)
            if entry_id is not None:
                injected_ids.add(entry_id)

    if collected:
        if lines:
            lines.append("")
        lines.append("[ai-mem] Relevant context for your prompt:")
        for coll, r in collected:
            text = r.text[:MAX_CHARS_PER_HIT]
            if len(r.text) > MAX_CHARS_PER_HIT:
                text += "..."
            score = f"{r.score:.2f}" if r.score is not None else "n/a"
            lines.append(f"- [{coll} score={score}] {text}")
            entry_id = getattr(r, "id", None)
            if entry_id is not None:
                injected_ids.add(entry_id)

    _save_session_injected(DB_PATH, injected_ids)

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
