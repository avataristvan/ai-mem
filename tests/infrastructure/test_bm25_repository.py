"""Integration tests for BM25MemoryRepository."""
from __future__ import annotations

import pytest

from ai_mem.application.add_memory import AddMemoryUseCase
from ai_mem.domain.memory import QueryResult
from ai_mem.infrastructure.bm25_repository import BM25MemoryRepository

pytest.importorskip("rank_bm25", reason="rank_bm25 not installed")


class _FixedScoreRepo:
    """Fake inner repo returning pre-set candidates with controlled cosine scores,
    so fusion behavior can be tested independent of actual embedding values."""

    def __init__(self, candidates: list[QueryResult]) -> None:
        self._candidates = candidates

    def query(self, collection, text, n_results, max_age_days, type_filter=None):
        return self._candidates


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
        # cosine is no longer min-max normalized (post-fix), so it can be slightly
        # negative for dissimilar embeddings; bm25_norm stays in [0,1]. Fused range
        # is therefore [-1, 1] with the current alpha, not the old tight [0, 1].
        assert -1.0 <= r.score <= 1.0


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


def test_weak_pool_cosine_not_rescaled_to_one():
    """A candidate with a real cosine score of 0.1 (weak) must not be rescaled to
    1.0 just because it's the best-ranked of an even weaker pool (0.05, 0.02).
    With alpha=1.0 (pure cosine, no BM25 contribution) the fused score must equal
    the raw cosine value exactly — proving cosine is no longer min-max normalized."""
    candidates = [
        QueryResult(rank=1, id="best", score=0.1, text="apple fruit basket", metadata={}),
        QueryResult(rank=2, id="mid", score=0.05, text="banana fruit basket", metadata={}),
        QueryResult(rank=3, id="worst", score=0.02, text="cherry fruit basket", metadata={}),
    ]
    repo = BM25MemoryRepository(_FixedScoreRepo(candidates), alpha=1.0)
    results = repo.query("weak", "fruit basket", n_results=3, max_age_days=None)
    assert results[0].id == "best"
    assert results[0].score == pytest.approx(0.1, abs=1e-4)


def test_strong_pool_fusion_ranking_preserved():
    """A candidate with a genuinely strong cosine score, reinforced by exact BM25
    term matches, must still rank above a weak, unrelated candidate — proving the
    fusion formula still combines both signals sensibly."""
    candidates = [
        QueryResult(
            rank=1,
            id="strong",
            score=0.9,
            text="TorchMicroRanker neural network model",
            metadata={},
        ),
        QueryResult(
            rank=2,
            id="weak",
            score=0.1,
            text="completely unrelated database text",
            metadata={},
        ),
    ]
    repo = BM25MemoryRepository(_FixedScoreRepo(candidates), alpha=0.5)
    results = repo.query(
        "strong", "TorchMicroRanker neural network", n_results=2, max_age_days=None
    )
    assert results[0].id == "strong"
    assert results[0].score > results[1].score
