"""Integration tests for BM25MemoryRepository."""
from __future__ import annotations

import pytest

from ai_mem.application.add_memory import AddMemoryUseCase
from ai_mem.infrastructure.bm25_repository import BM25MemoryRepository

pytest.importorskip("rank_bm25", reason="rank_bm25 not installed")


@pytest.fixture
def bm25_repo(tmp_repo):
    return BM25MemoryRepository(tmp_repo)


def _upsert(repo, collection: str, entries: list[tuple[str, str]]) -> None:
    add = AddMemoryUseCase(repo)
    docs = [text for _, text in entries]
    ids = [id_ for id_, _ in entries]
    add.execute(collection=collection, documents=docs, ids=ids)


def test_query_returns_hybrid_scores(bm25_repo):
    _upsert(
        bm25_repo,
        "test",
        [
            ("a", "TorchMicroRanker is an MLP neural network"),
            ("b", "ChromaDB is a vector database for embeddings"),
            ("c", "Python is a general purpose programming language"),
        ],
    )
    results = bm25_repo.query("test", "TorchMicroRanker", n_results=3, max_age_days=None)
    assert len(results) == 3
    assert results[0].id == "a"


def test_exact_term_boosted(tmp_repo):
    """alpha=0.0 (pure BM25) must rank exact-term match above a semantically related doc."""
    bm25_only = BM25MemoryRepository(tmp_repo, alpha=0.0)
    _upsert(
        bm25_only,
        "boost",
        [
            ("exact", "xyzzy_token appears in this document"),
            ("semantic", "vector database embeddings for AI memory retrieval"),
        ],
    )
    results = bm25_only.query("boost", "xyzzy_token", n_results=2, max_age_days=None)
    assert results[0].id == "exact"


def test_scores_in_range(bm25_repo):
    _upsert(
        bm25_repo,
        "range",
        [
            ("x", "semantic memory for AI agents"),
            ("y", "gradient descent optimisation step"),
            ("z", "chromadb persistent client path"),
        ],
    )
    results = bm25_repo.query("range", "AI agent memory", n_results=3, max_age_days=None)
    for r in results:
        assert 0.0 <= r.score <= 1.0


def test_respects_n_results(bm25_repo):
    _upsert(
        bm25_repo,
        "nres",
        [
            ("1", "apple fruit"),
            ("2", "banana fruit"),
            ("3", "cherry fruit"),
            ("4", "date fruit"),
            ("5", "elderberry fruit"),
        ],
    )
    results = bm25_repo.query("nres", "fruit", n_results=2, max_age_days=None)
    assert len(results) == 2


def test_delegates_upsert_and_list(bm25_repo, tmp_repo):
    _upsert(bm25_repo, "delegate", [("d1", "hello world"), ("d2", "foo bar")])
    # verify inner repo can see entries via direct query
    inner_results = tmp_repo.query("delegate", "hello", n_results=2, max_age_days=None)
    assert len(inner_results) == 2


def test_empty_collection_returns_empty(bm25_repo):
    results = bm25_repo.query("empty_col", "anything", n_results=5, max_age_days=None)
    assert results == []


def test_single_entry(bm25_repo):
    _upsert(bm25_repo, "solo", [("only", "the only document in this collection")])
    results = bm25_repo.query("solo", "document", n_results=1, max_age_days=None)
    assert len(results) == 1
    assert results[0].id == "only"
    assert 0.0 <= results[0].score <= 1.0
