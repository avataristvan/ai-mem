"""QueryMemoryUseCase: empty collection, ranking, access tracking."""
from __future__ import annotations

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
