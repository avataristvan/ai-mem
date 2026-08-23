"""Unit tests for ChromaMemoryRepository.query() score computation.

Collections are created via get_or_create_collection() with no explicit space
metadata, which defaults to ChromaDB's `l2` (squared L2) space, not cosine.
For normalized embeddings, dist = 2 - 2*cos_sim, so cos_sim = 1 - dist/2.
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from ai_mem.application.add_memory import AddMemoryUseCase


def _mock_query_result(distances: list[float]) -> dict:
    n = len(distances)
    return {
        "documents": [[f"doc{i}" for i in range(n)]],
        "ids": [[f"id{i}" for i in range(n)]],
        "metadatas": [[{} for _ in range(n)]],
        "distances": [distances],
    }


def _mock_collection(distances: list[float]) -> MagicMock:
    col = MagicMock()
    col.count.return_value = len(distances)
    col.query.return_value = _mock_query_result(distances)
    return col


def test_query_score_identical_vectors_dist_zero(tmp_repo):
    tmp_repo._col = MagicMock(return_value=_mock_collection([0.0]))

    results = tmp_repo.query("col", "text", n_results=1, max_age_days=None)

    assert results[0].score == 1.0


def test_query_score_orthogonal_vectors_dist_two(tmp_repo):
    tmp_repo._col = MagicMock(return_value=_mock_collection([2.0]))

    results = tmp_repo.query("col", "text", n_results=1, max_age_days=None)

    assert results[0].score == 0.0


def test_query_score_matches_verified_real_world_distance(tmp_repo):
    tmp_repo._col = MagicMock(return_value=_mock_collection([1.758348822593689]))

    results = tmp_repo.query("col", "text", n_results=1, max_age_days=None)

    assert results[0].score == pytest.approx(0.1208, abs=1e-4)


# ---------------------------------------------------------------------------
# Embedding cache: the same query text must be embedded at most once, even
# across different collections/type_filters (userprompt_hook's 4-way fan-out)
# or concurrent callers (the daemon serving multiple threads at once).
# ---------------------------------------------------------------------------

def _spy_on_embedding_function(tmp_repo):
    """Wrap tmp_repo's real embedding function with a call-recording proxy;
    returns the list of calls (each entry is the `texts` list passed in)."""
    real_ef = tmp_repo._embedding_function
    calls: list[list[str]] = []

    def spy(texts):
        calls.append(list(texts))
        return real_ef(texts)

    tmp_repo._embedding_function = spy
    return calls


def test_query_embeds_same_text_only_once_across_collections_and_filters(tmp_repo):
    AddMemoryUseCase(tmp_repo).execute(collection="global", documents=["alpha fact"], ids=["a1"])
    AddMemoryUseCase(tmp_repo).execute(
        collection="repo.x", documents=["beta fact"], ids=["b1"], metadatas=[{"type": "anti-pattern"}]
    )
    calls = _spy_on_embedding_function(tmp_repo)

    # Mirrors userprompt_hook's 4-way fan-out: same text, different collection/type_filter.
    tmp_repo.query("global", "shared prompt text", n_results=3, max_age_days=None)
    tmp_repo.query("repo.x", "shared prompt text", n_results=3, max_age_days=None)
    tmp_repo.query("repo.x", "shared prompt text", n_results=2, max_age_days=None, type_filter="anti-pattern")
    tmp_repo.query("repo.x", "shared prompt text", n_results=2, max_age_days=None, type_filter="dilemma")

    assert len(calls) == 1
    assert calls[0] == ["shared prompt text"]


def test_query_embeds_distinct_text_separately(tmp_repo):
    AddMemoryUseCase(tmp_repo).execute(collection="global", documents=["alpha fact"], ids=["a1"])
    calls = _spy_on_embedding_function(tmp_repo)

    tmp_repo.query("global", "first text", n_results=3, max_age_days=None)
    tmp_repo.query("global", "second text", n_results=3, max_age_days=None)

    assert len(calls) == 2


def test_concurrent_identical_text_queries_embed_only_once(tmp_repo):
    """N threads querying the exact same text concurrently must share one embedding
    computation -- the first computes, the rest wait and reuse it (per-key locking),
    not each compute independently."""
    AddMemoryUseCase(tmp_repo).execute(collection="global", documents=["alpha fact"], ids=["a1"])

    real_ef = tmp_repo._embedding_function
    calls: list[list[str]] = []
    calls_lock = threading.Lock()

    def slow_spy(texts):
        with calls_lock:
            calls.append(list(texts))
        time.sleep(0.1)  # widen the race window so concurrent callers overlap
        return real_ef(texts)

    tmp_repo._embedding_function = slow_spy

    n_threads = 5
    threads = [
        threading.Thread(target=tmp_repo.query, args=("global", "same concurrent text"), kwargs={"n_results": 3, "max_age_days": None})
        for _ in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert len(calls) == 1
