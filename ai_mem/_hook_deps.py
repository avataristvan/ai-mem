"""Shared infrastructure wiring for Claude Code lifecycle hooks.

All three hooks (UserPromptSubmit, PreToolUse, PostToolUse) need the same
ChromaDB / BM25 / Ranker stack. This module keeps that wiring in one place.
"""
from __future__ import annotations

from pathlib import Path


def _resolve_ranker_class():
    try:
        from ai_mem.infrastructure.torch_ranker import TorchMicroRanker
        return TorchMicroRanker, True
    except ImportError:
        from ai_mem.infrastructure.null_ranker import NullRanker  # type: ignore[assignment]
        return NullRanker, False


def _build_core(db_path: Path, with_bm25: bool = False):
    """Return (repo, storage, scope_resolver, registry, RankerClass).

    Pass with_bm25=True for hooks that need high-precision retrieval.
    PostToolUse and PreToolUse should pass False — BM25 adds latency
    the silent/passive hooks cannot afford.
    """
    from ai_mem.application.load_ranker_config import LoadRankerConfigUseCase
    from ai_mem.application.ranker_registry import RankerRegistry
    from ai_mem.domain.learning import RankerScope
    from ai_mem.infrastructure.chroma_repository import ChromaMemoryRepository
    from ai_mem.infrastructure.ranker_storage import RankerStorage

    RankerClass, has_torch = _resolve_ranker_class()
    repo = ChromaMemoryRepository(db_path)
    if with_bm25:
        try:
            from ai_mem.infrastructure.bm25_repository import BM25MemoryRepository
            repo = BM25MemoryRepository(repo)
        except ImportError:
            pass
    storage = RankerStorage(db_path / "rankers")
    scope_map = LoadRankerConfigUseCase(db_path / "ranker_config.json").execute()
    scope_resolver = lambda c: scope_map.get(c, RankerScope(name=c, mode="isolated"))
    fallback_factory = None
    if has_torch:
        from ai_mem.infrastructure.null_ranker import NullRanker
        fallback_factory = NullRanker
    registry = RankerRegistry(
        scope_resolver=scope_resolver,
        ranker_factory=RankerClass,
        storage=storage,
        fallback_factory=fallback_factory,
    )
    return repo, storage, scope_resolver, registry, RankerClass


def _make_query_uc(repo, storage, scope_resolver, registry, RankerClass):
    """Assemble a QueryMemoryUseCase from pre-built infrastructure components."""
    from ai_mem.application.build_features import BuildFeaturesUseCase
    from ai_mem.application.query_memory import QueryMemoryUseCase
    from ai_mem.application.track_access import TrackAccessUseCase
    from ai_mem.application.train_ranker import TrainRankerUseCase

    return QueryMemoryUseCase(
        repo,
        TrackAccessUseCase(repo),
        BuildFeaturesUseCase(),
        TrainRankerUseCase(repo, storage, RankerClass, scope_resolver=scope_resolver),
        registry,
    )
