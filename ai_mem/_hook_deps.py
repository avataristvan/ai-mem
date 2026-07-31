"""Shared infrastructure wiring for Claude Code lifecycle hooks."""
from __future__ import annotations

from pathlib import Path


def _build_query_uc(db_path: Path, with_bm25: bool = False):
    """Return something exposing .execute(...) like QueryMemoryUseCase, wired to ChromaDB
    (optionally BM25-wrapped).

    Pass with_bm25=True for hooks that need high-precision retrieval.
    PostToolUse should pass False — BM25 adds latency the silent hook cannot afford.

    Tries the warm background daemon first (see daemon_client.py) — every hook call is a
    fresh subprocess, so importing chromadb + loading the embedding model from scratch costs
    ~1.1s that a daemon avoids by staying warm across calls. Falls back to building
    everything in-process (the original, always-correct path) when the daemon is unavailable.
    """
    try:
        from ai_mem.daemon_client import get_daemon_query_uc
        daemon_uc = get_daemon_query_uc(db_path, with_bm25)
        if daemon_uc is not None:
            return daemon_uc
    except Exception:
        pass

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
