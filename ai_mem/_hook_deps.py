"""Shared infrastructure wiring for Claude Code lifecycle hooks."""
from __future__ import annotations

from pathlib import Path


def _build_query_uc(db_path: Path, with_bm25: bool = False):
    """Return a QueryMemoryUseCase wired to ChromaDB (optionally BM25-wrapped).

    Pass with_bm25=True for hooks that need high-precision retrieval.
    PostToolUse should pass False — BM25 adds latency the silent hook cannot afford.
    """
    from ai_mem.application.query_memory import QueryMemoryUseCase
    from ai_mem.application.track_access import TrackAccessUseCase
    from ai_mem.infrastructure.chroma_repository import ChromaMemoryRepository

    repo = ChromaMemoryRepository(db_path)
    if with_bm25:
        try:
            from ai_mem.infrastructure.bm25_repository import BM25MemoryRepository
            repo = BM25MemoryRepository(repo)
        except ImportError:
            pass
    return QueryMemoryUseCase(repo, TrackAccessUseCase(repo))
