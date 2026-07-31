"""QueryMemoryUseCase: empty collection, basic query, access tracking."""
from __future__ import annotations

from pathlib import Path

from ai_mem.application.add_memory import AddMemoryUseCase
from ai_mem.application.get_memory import GetMemoryUseCase
from ai_mem.application.query_memory import QueryMemoryUseCase
from ai_mem.application.track_access import TrackAccessUseCase


def _make_query_uc(tmp_repo, track_access) -> QueryMemoryUseCase:
    return QueryMemoryUseCase(repo=tmp_repo, track_access=track_access)


def test_query_empty_collection_returns_empty(tmp_repo, track_access):
    results = _make_query_uc(tmp_repo, track_access).execute(
        collection="empty", query="anything"
    )
    assert results == []


def test_query_returns_results_after_add(tmp_repo, track_access):
    AddMemoryUseCase(tmp_repo).execute(
        collection="test_col",
        documents=["the cat sat on the mat", "completely unrelated text"],
        ids=["a", "b"],
    )
    results = _make_query_uc(tmp_repo, track_access).execute(
        collection="test_col", query="cat", n_results=2
    )
    assert len(results) == 2
    assert results[0].id == "a"


def test_query_type_filter_restricts_results(tmp_repo, track_access):
    AddMemoryUseCase(tmp_repo).execute(
        collection="typed_col",
        documents=["feedback entry", "reference entry"],
        ids=["f1", "r1"],
        metadatas=[{"type": "feedback"}, {"type": "reference"}],
    )
    results = _make_query_uc(tmp_repo, track_access).execute(
        collection="typed_col", query="entry", n_results=5, type_filter="feedback"
    )
    assert len(results) == 1
    assert results[0].id == "f1"


def test_query_without_min_score_for_tracking_does_not_track_access(tmp_repo, track_access):
    """access_count means deliberate use, not passive surfacing (chunk 3 of
    arch_decision_push_vs_pull_2026_07_31) -- a plain query, with no explicit tracking
    threshold, must never bump access_count just because a result was returned."""
    AddMemoryUseCase(tmp_repo).execute(
        collection="test_col",
        documents=["the cat sat on the mat"],
        ids=["a"],
    )
    query_uc = _make_query_uc(tmp_repo, track_access)
    for _ in range(3):
        query_uc.execute(collection="test_col", query="cat", n_results=1)

    fetched = GetMemoryUseCase(tmp_repo).execute("test_col", ["a"])
    assert fetched[0].metadata.get("access_count", 0) == 0


def test_query_with_min_score_for_tracking_increments_access_count(tmp_repo, track_access):
    """An explicit min_score_for_tracking opt-in (posttool_hook's real usage signal) still
    tracks access on each qualifying call, repeated calls accumulate."""
    AddMemoryUseCase(tmp_repo).execute(
        collection="test_col",
        documents=["the cat sat on the mat"],
        ids=["a"],
    )
    query_uc = _make_query_uc(tmp_repo, track_access)
    for _ in range(3):
        query_uc.execute(collection="test_col", query="cat", n_results=1, min_score_for_tracking=0.0)

    fetched = GetMemoryUseCase(tmp_repo).execute("test_col", ["a"])
    assert fetched[0].metadata["access_count"] == 3
    assert fetched[0].metadata["last_accessed_at"] >= fetched[0].metadata["created_at"]


def test_min_score_for_tracking_suppresses_weak_matches(tmp_repo, track_access):
    """Entries below min_score_for_tracking must not have last_accessed_at updated."""
    AddMemoryUseCase(tmp_repo).execute(
        collection="test_col",
        documents=["the cat sat on the mat"],
        ids=["a"],
    )
    get_uc = GetMemoryUseCase(tmp_repo)
    created_at = get_uc.execute("test_col", ["a"])[0].metadata["created_at"]

    # Query with an impossibly high threshold — no real result will score ≥ 1.0
    _make_query_uc(tmp_repo, track_access).execute(
        collection="test_col",
        query="cat",
        n_results=1,
        min_score_for_tracking=1.0,
    )

    fetched = get_uc.execute("test_col", ["a"])[0]
    # Access was NOT tracked: last_accessed_at still equals created_at
    assert fetched.metadata.get("access_count", 0) == 0
    assert fetched.metadata.get("last_accessed_at", created_at) == created_at


def test_min_score_for_tracking_allows_strong_matches(tmp_repo, track_access):
    """Entries at or above min_score_for_tracking DO get last_accessed_at updated."""
    AddMemoryUseCase(tmp_repo).execute(
        collection="test_col",
        documents=["the cat sat on the mat"],
        ids=["a"],
    )
    # Threshold of 0.0 allows everything through
    _make_query_uc(tmp_repo, track_access).execute(
        collection="test_col",
        query="cat",
        n_results=1,
        min_score_for_tracking=0.0,
    )

    fetched = GetMemoryUseCase(tmp_repo).execute("test_col", ["a"])[0]
    assert fetched.metadata.get("access_count", 0) == 1
