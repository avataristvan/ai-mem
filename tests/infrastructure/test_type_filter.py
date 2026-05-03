"""type_filter propagation through ChromaMemoryRepository and BM25MemoryRepository."""
from __future__ import annotations

import pytest

from ai_mem.application.add_memory import AddMemoryUseCase


def _seed(repo, collection: str) -> None:
    add = AddMemoryUseCase(repo)
    add.execute(
        collection=collection,
        documents=["feedback about the UI", "reference to the API docs", "project goal statement"],
        ids=["f1", "r1", "p1"],
        metadatas=[{"type": "feedback"}, {"type": "reference"}, {"type": "project"}],
    )


def test_type_filter_restricts_results(tmp_repo):
    _seed(tmp_repo, "col")
    results = tmp_repo.query("col", "text", n_results=10, max_age_days=None, type_filter="feedback")
    assert len(results) == 1
    assert results[0].id == "f1"


def test_type_filter_none_returns_all(tmp_repo):
    _seed(tmp_repo, "col")
    results = tmp_repo.query("col", "text", n_results=10, max_age_days=None, type_filter=None)
    assert len(results) == 3


def test_type_filter_no_match_returns_empty(tmp_repo):
    _seed(tmp_repo, "col")
    results = tmp_repo.query("col", "text", n_results=10, max_age_days=None, type_filter="nonexistent")
    assert results == []


def test_bm25_passes_type_filter_through(tmp_repo):
    pytest.importorskip("rank_bm25", reason="rank_bm25 not installed")
    from ai_mem.infrastructure.bm25_repository import BM25MemoryRepository

    bm25 = BM25MemoryRepository(tmp_repo)
    _seed(bm25, "col")
    results = bm25.query("col", "text", n_results=10, max_age_days=None, type_filter="reference")
    assert len(results) == 1
    assert results[0].id == "r1"
