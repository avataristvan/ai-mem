"""QueryMemoryUseCase: empty collection, ranking, access tracking."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from ai_mem.application.add_memory import AddMemoryUseCase
from ai_mem.application.build_features import BuildFeaturesUseCase
from ai_mem.application.get_memory import GetMemoryUseCase
from ai_mem.application.query_memory import QueryMemoryUseCase
from ai_mem.application.ranker_registry import RankerRegistry
from ai_mem.application.track_access import TrackAccessUseCase
from ai_mem.application.train_ranker import TrainRankerUseCase
from ai_mem.domain.learning import RankerScope
from ai_mem.infrastructure.null_ranker import NullRanker
from ai_mem.infrastructure.ranker_storage import RankerStorage


def _make_query_uc(tmp_repo, track_access, tmp_path: Path) -> QueryMemoryUseCase:
    storage = RankerStorage(tmp_path / "rankers")
    train_ranker = TrainRankerUseCase(tmp_repo, storage, NullRanker)
    registry = RankerRegistry(
        scope_resolver=lambda c: RankerScope(name=c, mode="isolated"),
        ranker_factory=NullRanker,
        storage=storage,
    )
    return QueryMemoryUseCase(
        repo=tmp_repo,
        track_access=track_access,
        build_features=BuildFeaturesUseCase(),
        train_ranker=train_ranker,
        ranker_provider=registry,
    )


def test_query_empty_collection_returns_empty(tmp_repo, track_access, tmp_path):
    results = _make_query_uc(tmp_repo, track_access, tmp_path).execute(
        collection="empty", query="anything"
    )
    assert results == []


def test_query_returns_results_after_add(tmp_repo, track_access, tmp_path):
    AddMemoryUseCase(tmp_repo).execute(
        collection="test_col",
        documents=["the cat sat on the mat", "completely unrelated text"],
        ids=["a", "b"],
    )
    results = _make_query_uc(tmp_repo, track_access, tmp_path).execute(
        collection="test_col", query="cat", n_results=2
    )
    assert len(results) == 2
    assert results[0].id == "a"


def test_query_type_filter_restricts_results(tmp_repo, track_access, tmp_path):
    AddMemoryUseCase(tmp_repo).execute(
        collection="typed_col",
        documents=["feedback entry", "reference entry"],
        ids=["f1", "r1"],
        metadatas=[{"type": "feedback"}, {"type": "reference"}],
    )
    results = _make_query_uc(tmp_repo, track_access, tmp_path).execute(
        collection="typed_col", query="entry", n_results=5, type_filter="feedback"
    )
    assert len(results) == 1
    assert results[0].id == "f1"


def test_query_increments_access_count(tmp_repo, track_access, tmp_path):
    AddMemoryUseCase(tmp_repo).execute(
        collection="test_col",
        documents=["the cat sat on the mat"],
        ids=["a"],
    )
    query_uc = _make_query_uc(tmp_repo, track_access, tmp_path)
    for _ in range(3):
        query_uc.execute(collection="test_col", query="cat", n_results=1)

    fetched = GetMemoryUseCase(tmp_repo).execute("test_col", ["a"])
    assert fetched[0].metadata["access_count"] == 3
    assert fetched[0].metadata["last_accessed_at"] >= fetched[0].metadata["created_at"]


def test_min_score_for_tracking_suppresses_weak_matches(tmp_repo, track_access, tmp_path):
    """Entries below min_score_for_tracking must not have last_accessed_at updated."""
    AddMemoryUseCase(tmp_repo).execute(
        collection="test_col",
        documents=["the cat sat on the mat"],
        ids=["a"],
    )
    get_uc = GetMemoryUseCase(tmp_repo)
    created_at = get_uc.execute("test_col", ["a"])[0].metadata["created_at"]

    # Query with an impossibly high threshold — no real result will score ≥ 1.0
    _make_query_uc(tmp_repo, track_access, tmp_path).execute(
        collection="test_col",
        query="cat",
        n_results=1,
        min_score_for_tracking=1.0,
    )

    fetched = get_uc.execute("test_col", ["a"])[0]
    # Access was NOT tracked: last_accessed_at still equals created_at
    assert fetched.metadata.get("access_count", 0) == 0
    assert fetched.metadata.get("last_accessed_at", created_at) == created_at


def test_min_score_for_tracking_allows_strong_matches(tmp_repo, track_access, tmp_path):
    """Entries at or above min_score_for_tracking DO get last_accessed_at updated."""
    AddMemoryUseCase(tmp_repo).execute(
        collection="test_col",
        documents=["the cat sat on the mat"],
        ids=["a"],
    )
    # Threshold of 0.0 allows everything through
    _make_query_uc(tmp_repo, track_access, tmp_path).execute(
        collection="test_col",
        query="cat",
        n_results=1,
        min_score_for_tracking=0.0,
    )

    fetched = GetMemoryUseCase(tmp_repo).execute("test_col", ["a"])[0]
    assert fetched.metadata.get("access_count", 0) == 1


def test_null_ranker_session_hit_penalty():
    from ai_mem.domain.learning import RankingFeatures
    from ai_mem.infrastructure.null_ranker import (
        _EXPLORE_BONUS, _SESSION_HIT_PENALTY, _SESSION_HIT_SATURATION,
        _session_hit_multiplier,
    )
    ranker = NullRanker()

    # Low-access entry (new): full penalty applied
    low_access = RankingFeatures(
        cosine_similarity=0.9, age_days=0, last_access_days=0,
        access_count=0, session_hit=True,
    )
    # High-access entry (load-bearing): penalty fully lifted
    high_access = RankingFeatures(
        cosine_similarity=0.9, age_days=0, last_access_days=0,
        access_count=_SESSION_HIT_SATURATION, session_hit=True,
    )
    # No session hit: no penalty
    no_hit = RankingFeatures(
        cosine_similarity=0.85, age_days=0, last_access_days=0,
        access_count=0, session_hit=False,
    )

    scores = ranker.rank([low_access, high_access, no_hit])

    # Low-access: full _SESSION_HIT_PENALTY
    expected_low = 0.9 * _SESSION_HIT_PENALTY + _EXPLORE_BONUS
    assert abs(scores[0] - expected_low) < 1e-6

    # High-access: multiplier == 1.0 (no penalty)
    assert _session_hit_multiplier(_SESSION_HIT_SATURATION) == 1.0
    expected_high = 0.9 * 1.0 + _EXPLORE_BONUS / math.sqrt(_SESSION_HIT_SATURATION)
    assert abs(scores[1] - expected_high) < 1e-6

    # High-access session hit scores higher than low-access session hit
    assert scores[1] > scores[0]

    # No-hit scores without penalty
    expected_no_hit = 0.85 + _EXPLORE_BONUS
    assert abs(scores[2] - expected_no_hit) < 1e-6
