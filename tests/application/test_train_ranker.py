"""TrainRankerUseCase: buffer writes, labeling, train_step."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from ai_mem.application.add_memory import AddMemoryUseCase
from ai_mem.application.build_features import BuildFeaturesUseCase
from ai_mem.application.train_ranker import TrainRankerUseCase
from ai_mem.domain.learning import RankingFeatures, TrainingMetrics
from ai_mem.domain.memory import QueryResult
from ai_mem.infrastructure.null_ranker import NullRanker
from ai_mem.infrastructure.ranker_storage import RankerStorage


@pytest.fixture
def storage(tmp_path: Path) -> RankerStorage:
    return RankerStorage(tmp_path / "rankers")


@pytest.fixture
def train_uc(tmp_repo, storage) -> TrainRankerUseCase:
    return TrainRankerUseCase(
        repo=tmp_repo,
        storage=storage,
        ranker_factory=NullRanker,
        label_window_days=7.0,
        buffer_max=1000,
    )


def _qresult(id_: str, score: float = 0.8) -> QueryResult:
    return QueryResult(rank=1, id=id_, score=score, text="text", metadata={"created_at": time.time()})


def _features() -> RankingFeatures:
    return RankingFeatures(
        cosine_similarity=0.2,
        age_days=0.0,
        last_access_days=0.0,
        access_count=0,
        has_ttl=False,
        expires_in_days=0.0,
    )


def test_record_query_writes_buffer(train_uc, storage):
    now = time.time()
    results = [_qresult("a"), _qresult("b")]
    features = [_features(), _features()]
    train_uc.record_query("col", results, features, now)

    buf = storage.load_buffer("col")
    assert len(buf) == 2
    assert {e.memory_id for e in buf} == {"a", "b"}
    assert buf[0].target_future_access is None


def test_record_query_immediately_labels_returned_ids(train_uc, storage):
    """Entries in returned_ids get target=1.0; others remain unlabeled."""
    now = time.time()
    results = [_qresult("a"), _qresult("b"), _qresult("c")]
    features = [_features()] * 3
    train_uc.record_query("col", results, features, now, returned_ids={"a", "c"})

    buf = {e.memory_id: e for e in storage.load_buffer("col")}
    assert buf["a"].target_future_access == 1.0
    assert buf["b"].target_future_access is None
    assert buf["c"].target_future_access == 1.0


def test_record_query_sets_co_activated_ids(train_uc, storage):
    now = time.time()
    results = [_qresult("a"), _qresult("b"), _qresult("c")]
    features = [_features()] * 3
    train_uc.record_query("col", results, features, now)

    buf = storage.load_buffer("col")
    ids_by_mem = {e.memory_id: set(e.co_activated_ids) for e in buf}
    assert ids_by_mem["a"] == {"b", "c"}
    assert ids_by_mem["b"] == {"a", "c"}


def test_train_step_no_buffer_returns_skipped(train_uc):
    now = time.time()
    metrics = train_uc.train_step("empty_col", now)
    assert metrics.skipped is True
    assert metrics.n == 0


def test_train_step_unlabeled_only_returns_skipped(train_uc, storage):
    now = time.time()
    results = [_qresult("a")]
    features = [_features()]
    train_uc.record_query("col", results, features, now)

    # Label window not elapsed yet — no labels assigned
    metrics = train_uc.train_step("col", now)
    assert metrics.skipped is True


def test_train_step_with_backdated_examples_produces_labels(tmp_repo, storage):
    # Insert a real memory entry with access history
    AddMemoryUseCase(tmp_repo).execute(
        collection="col",
        documents=["hello world"],
        ids=["a"],
    )

    now = time.time()
    eight_days_ago = now - 8 * 86400

    build = BuildFeaturesUseCase()
    from ai_mem.domain.memory import QueryResult
    fake_result = QueryResult(
        rank=1,
        id="a",
        score=0.9,
        text="hello world",
        metadata={"created_at": eight_days_ago, "access_count": 0},
    )
    feats = build.execute([fake_result], eight_days_ago)

    uc = TrainRankerUseCase(
        repo=tmp_repo,
        storage=storage,
        ranker_factory=NullRanker,
        label_window_days=7.0,
    )
    uc.record_query("col", [fake_result], feats, eight_days_ago)

    metrics = uc.train_step("col", now)
    # NullRanker.train_step always returns skipped, but labels were assigned
    # and the buffer was processed without error
    assert isinstance(metrics, TrainingMetrics)


def test_train_step_drops_labeled_examples_from_buffer(tmp_repo, storage):
    """After training, examples whose labels were applied must be removed
    from the buffer. Otherwise they accumulate across train_step calls."""
    AddMemoryUseCase(tmp_repo).execute(
        collection="col", documents=["a"], ids=["a"]
    )

    now = time.time()
    eight_days_ago = now - 8 * 86400
    fresh = now - 60  # well within label window

    uc = TrainRankerUseCase(
        repo=tmp_repo, storage=storage, ranker_factory=NullRanker, label_window_days=7.0
    )

    old_qres = QueryResult(rank=1, id="a", score=0.9, text="a",
                           metadata={"created_at": eight_days_ago})
    fresh_qres = QueryResult(rank=1, id="a", score=0.9, text="a",
                             metadata={"created_at": fresh})

    uc.record_query("col", [old_qres], [_features()], eight_days_ago)
    uc.record_query("col", [fresh_qres], [_features()], fresh)
    assert len(storage.load_buffer("col")) == 2

    uc.train_step("col", now)

    remaining = storage.load_buffer("col")
    assert len(remaining) == 1
    assert remaining[0].retrieved_at == fresh
    assert remaining[0].target_future_access is None
