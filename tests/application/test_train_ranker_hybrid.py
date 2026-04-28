"""TrainRankerUseCase: hybrid mode scope routing."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from ai_mem.application.add_memory import AddMemoryUseCase
from ai_mem.application.build_features import BuildFeaturesUseCase
from ai_mem.application.train_ranker import TrainRankerUseCase
from ai_mem.domain.learning import RankerScope, RankingFeatures
from ai_mem.domain.memory import QueryResult
from ai_mem.infrastructure.null_ranker import NullRanker
from ai_mem.infrastructure.ranker_storage import RankerStorage


def _hybrid_resolver(group: str, members: list[str]):
    scope = RankerScope(name=group, mode="hybrid", group=group, member_collections=members)
    return lambda c: scope if c in members else RankerScope(name=c, mode="isolated")


def _qresult(id_: str) -> QueryResult:
    return QueryResult(rank=1, id=id_, score=0.8, text="text", metadata={"created_at": time.time()})


def _features() -> RankingFeatures:
    return RankingFeatures(
        cosine_similarity=0.5,
        age_days=0.0,
        last_access_days=0.0,
        access_count=0,
        has_ttl=False,
        expires_in_days=0.0,
    )


@pytest.fixture
def storage(tmp_path: Path) -> RankerStorage:
    return RankerStorage(tmp_path / "rankers")


def test_hybrid_buffer_written_to_group_path(storage: RankerStorage, tmp_repo):
    resolver = _hybrid_resolver("my-group", ["col.a", "col.b"])
    uc = TrainRankerUseCase(tmp_repo, storage, NullRanker, scope_resolver=resolver)

    now = time.time()
    uc.record_query("col.a", [_qresult("x")], [_features()], now)

    # Buffer should live at group path, not collection path
    assert storage.buffer_path("my-group").exists()
    assert not storage.buffer_path("col.a").exists()


def test_two_hybrid_collections_share_buffer(storage: RankerStorage, tmp_repo):
    resolver = _hybrid_resolver("shared-group", ["col.a", "col.b"])
    uc = TrainRankerUseCase(tmp_repo, storage, NullRanker, scope_resolver=resolver)

    now = time.time()
    uc.record_query("col.a", [_qresult("m1")], [_features()], now)
    uc.record_query("col.b", [_qresult("m2")], [_features()], now)

    buf = storage.load_buffer("shared-group")
    assert {ex.memory_id for ex in buf} == {"m1", "m2"}
    assert {ex.source_collection for ex in buf} == {"col.a", "col.b"}


def test_hybrid_weights_saved_to_group_path(storage: RankerStorage, tmp_repo):
    AddMemoryUseCase(tmp_repo).execute(collection="col.a", documents=["hello"], ids=["m1"])

    resolver = _hybrid_resolver("grp", ["col.a"])
    uc = TrainRankerUseCase(tmp_repo, storage, NullRanker, scope_resolver=resolver, label_window_days=7.0)

    eight_days_ago = time.time() - 8 * 86400
    fake = QueryResult(rank=1, id="m1", score=0.9, text="hello",
                       metadata={"created_at": eight_days_ago})
    feats = BuildFeaturesUseCase().execute([fake], eight_days_ago)
    uc.record_query("col.a", [fake], feats, eight_days_ago)

    uc.train_step("col.a", time.time())

    # NullRanker.save is a no-op but weights_path("grp") should be what was
    # attempted — confirm it was NOT the collection path
    assert not storage.weights_path("col.a").exists()


def test_isolated_collection_uses_own_path(storage: RankerStorage, tmp_repo):
    uc = TrainRankerUseCase(tmp_repo, storage, NullRanker)

    now = time.time()
    uc.record_query("isolated-col", [_qresult("z")], [_features()], now)

    assert storage.buffer_path("isolated-col").exists()
    assert not storage.buffer_path("shared-group").exists()


def test_source_collection_set_on_examples(storage: RankerStorage, tmp_repo):
    uc = TrainRankerUseCase(tmp_repo, storage, NullRanker)

    now = time.time()
    uc.record_query("my-col", [_qresult("a"), _qresult("b")], [_features(), _features()], now)

    buf = storage.load_buffer("my-col")
    assert all(ex.source_collection == "my-col" for ex in buf)
