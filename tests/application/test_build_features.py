"""BuildFeaturesUseCase: feature computation from QueryResult metadata."""
from __future__ import annotations

import time

from ai_mem.application.build_features import BuildFeaturesUseCase
from ai_mem.domain.memory import QueryResult


def _result(id_: str = "x", score: float = 0.9, metadata: dict | None = None) -> QueryResult:
    return QueryResult(rank=1, id=id_, score=score, text="text", metadata=metadata or {})


def test_defaults_when_metadata_empty():
    now = time.time()
    uc = BuildFeaturesUseCase()
    feats = uc.execute([_result(metadata={})], now)
    f = feats[0]
    assert f.access_count == 0
    assert f.has_ttl is False
    assert f.age_days == 0.0
    assert f.last_access_days == 0.0


def test_age_days_computed():
    now = time.time()
    five_days_ago = now - 5 * 86400
    uc = BuildFeaturesUseCase()
    feats = uc.execute([_result(metadata={"created_at": five_days_ago})], now)
    assert abs(feats[0].age_days - 5.0) < 0.01


def test_last_access_falls_back_to_created_when_missing():
    now = time.time()
    created = now - 86400
    uc = BuildFeaturesUseCase()
    feats = uc.execute([_result(metadata={"created_at": created})], now)
    assert abs(feats[0].last_access_days - 1.0) < 0.01


def test_has_ttl_when_expires_at_set():
    now = time.time()
    uc = BuildFeaturesUseCase()
    feats = uc.execute(
        [_result(metadata={"created_at": now, "expires_at": now + 10 * 86400})], now
    )
    assert feats[0].has_ttl is True


def test_as_vector_length():
    now = time.time()
    uc = BuildFeaturesUseCase()
    feats = uc.execute([_result()], now)
    assert len(feats[0].as_vector()) == 10


def test_session_hit_false_by_default():
    now = time.time()
    uc = BuildFeaturesUseCase()
    feats = uc.execute([_result(id_="x")], now)
    assert feats[0].session_hit is False
    assert feats[0].as_vector()[9] == 0.0


def test_session_hit_true_when_id_in_hits():
    now = time.time()
    uc = BuildFeaturesUseCase()
    feats = uc.execute([_result(id_="x"), _result(id_="y")], now, session_hits={"x"})
    assert feats[0].session_hit is True
    assert feats[1].session_hit is False


def test_access_count_populated():
    now = time.time()
    uc = BuildFeaturesUseCase()
    feats = uc.execute([_result(metadata={"created_at": now, "access_count": 7})], now)
    assert feats[0].access_count == 7


def test_access_count_capped_at_50():
    now = time.time()
    uc = BuildFeaturesUseCase()
    feats = uc.execute([_result(metadata={"created_at": now, "access_count": 200})], now)
    assert feats[0].access_count == 50


def test_access_count_below_cap_unchanged():
    now = time.time()
    uc = BuildFeaturesUseCase()
    feats = uc.execute([_result(metadata={"created_at": now, "access_count": 49})], now)
    assert feats[0].access_count == 49
