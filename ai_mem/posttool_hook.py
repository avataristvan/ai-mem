#!/usr/bin/env python3
"""PostToolUse hook for Write|Edit — records a passive training signal in ai-mem.

After a file is written or edited, the file path is used as a query signal.
Matching memory entries get their last_accessed_at updated (via TrackAccessUseCase
inside QueryMemoryUseCase), which causes them to be labeled positive when
train_step() runs within the 7-day window.

No output is produced — the hook is fully silent.
"""
import json
import os
import sys
import time
from pathlib import Path

from ai_mem.repo_context import GLOBAL_COLLECTION, WORKSPACE_COLLECTION, detect_repo_context

DB_PATH = Path(os.environ.get("AI_MEM_PATH", Path.home() / ".local" / "share" / "ai-mem"))
_QUERY_K = 5  # enough candidates to surface relevant entries; no injection needed


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
    train_uc = TrainRankerUseCase(repo, storage, RankerClass, scope_resolver=scope_resolver)
    return query_uc, train_uc


def _signal_collection(query_uc, train_uc, collection: str, query: str, now: float) -> None:
    """Query a collection (updating last_accessed_at) then run a train_step."""
    try:
        query_uc.execute(collection=collection, query=query, n_results=_QUERY_K)
    except Exception:
        return
    try:
        train_uc.train_step(collection=collection, now=now)
    except Exception:
        pass


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
        query_uc, train_uc = _build_deps()
    except Exception:
        return

    file_name = Path(file_path).name
    query = f"{file_name} {file_path}"
    now = time.time()

    _signal_collection(query_uc, train_uc, GLOBAL_COLLECTION, query, now)

    try:
        ctx = detect_repo_context()
        if ctx.collection not in (GLOBAL_COLLECTION, WORKSPACE_COLLECTION):
            _signal_collection(query_uc, train_uc, ctx.collection, query, now)
    except Exception:
        pass


if __name__ == "__main__":
    main()
