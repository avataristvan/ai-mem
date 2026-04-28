#!/usr/bin/env python3
"""PreToolUse hook for Write|Edit — injects relevant past experiences from ai-mem."""
import json
import os
import sys
from pathlib import Path

DB_PATH = Path(os.environ.get("AI_MEM_PATH", Path.home() / ".local" / "share" / "ai-mem"))
TOP_K = 2
MAX_CHARS_PER_HIT = 500


def _build_query_use_case():
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
    storage = RankerStorage(DB_PATH / "rankers")
    scope_map = LoadRankerConfigUseCase(DB_PATH / "ranker_config.json").execute()
    scope_resolver = lambda c: scope_map.get(c, RankerScope(name=c, mode="isolated"))
    registry = RankerRegistry(scope_resolver=scope_resolver, ranker_factory=RankerClass, storage=storage)
    return QueryMemoryUseCase(
        repo,
        TrackAccessUseCase(repo),
        BuildFeaturesUseCase(),
        TrainRankerUseCase(repo, storage, RankerClass, scope_resolver=scope_resolver),
        registry,
    )


def _hits(query_uc, collection: str, query: str):
    try:
        return query_uc.execute(collection=collection, query=query, n_results=TOP_K)
    except Exception:
        return []


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return

    if payload.get("tool_name") not in ("Write", "Edit"):
        return

    file_path = payload.get("tool_input", {}).get("file_path", "")
    if not file_path or not DB_PATH.exists():
        return

    try:
        from ai_mem.repo_context import GLOBAL_COLLECTION, WORKSPACE_COLLECTION, detect_repo_context

        query_uc = _build_query_use_case()
    except Exception:
        return

    tool_name = payload["tool_name"]
    file_name = Path(file_path).name
    query = f"{tool_name} {file_name} {file_path}"

    collected: list[tuple[str, object]] = [(GLOBAL_COLLECTION, r) for r in _hits(query_uc, GLOBAL_COLLECTION, query)]

    try:
        ctx = detect_repo_context()
        if ctx.collection not in (GLOBAL_COLLECTION, WORKSPACE_COLLECTION):
            collected.extend((ctx.collection, r) for r in _hits(query_uc, ctx.collection, query))
    except Exception:
        pass

    if not collected:
        return

    lines = [f"[ai-mem] Past experiences for {tool_name} on {file_name}:"]
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
                    "hookEventName": "PreToolUse",
                    "additionalContext": "\n".join(lines),
                }
            }
        )
    )


if __name__ == "__main__":
    main()
