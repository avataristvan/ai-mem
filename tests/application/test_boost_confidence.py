"""BoostConfidenceUseCase: confidence delta, clamping, unknown-id skipping."""
from __future__ import annotations

from ai_mem.application.add_memory import AddMemoryUseCase
from ai_mem.application.boost_confidence import BoostConfidenceUseCase
from ai_mem.application.get_memory import GetMemoryUseCase


def test_boost_clamped_at_1(tmp_repo):
    AddMemoryUseCase(tmp_repo).execute(collection="col", documents=["x"], ids=["a"])
    n = BoostConfidenceUseCase(tmp_repo).execute("col", ["a"], 0.5)
    assert n == 1
    meta = GetMemoryUseCase(tmp_repo).execute("col", ["a"])[0].metadata
    assert meta["confidence"] == 1.0


def test_boost_increases_confidence(tmp_repo):
    AddMemoryUseCase(tmp_repo).execute(
        collection="col", documents=["x"], ids=["a"], metadatas=[{"confidence": 0.5}]
    )
    n = BoostConfidenceUseCase(tmp_repo).execute("col", ["a"], 0.2)
    assert n == 1
    meta = GetMemoryUseCase(tmp_repo).execute("col", ["a"])[0].metadata
    assert abs(meta["confidence"] - 0.7) < 1e-9


def test_decay_clamped_at_0(tmp_repo):
    AddMemoryUseCase(tmp_repo).execute(
        collection="col", documents=["x"], ids=["a"], metadatas=[{"confidence": 0.2}]
    )
    n = BoostConfidenceUseCase(tmp_repo).execute("col", ["a"], -0.3)
    assert n == 1
    meta = GetMemoryUseCase(tmp_repo).execute("col", ["a"])[0].metadata
    assert meta["confidence"] == 0.0


def test_unknown_ids_silently_skipped(tmp_repo):
    n = BoostConfidenceUseCase(tmp_repo).execute("col", ["does-not-exist"], 0.1)
    assert n == 0


def test_return_count_matches_found_entries(tmp_repo):
    add = AddMemoryUseCase(tmp_repo)
    add.execute(collection="col", documents=["x", "y"], ids=["a", "b"],
                metadatas=[{"confidence": 0.5}, {"confidence": 0.5}])
    n = BoostConfidenceUseCase(tmp_repo).execute("col", ["a", "b", "missing"], 0.1)
    assert n == 2


# ---------------------------------------------------------------------------
# track_access wiring: a positive boost is the /reflect citation signal
# (arch_decision_push_vs_pull_2026_07_31 chunk 3) -- a decay must not count as usage.
# ---------------------------------------------------------------------------

def test_without_track_access_does_not_error(tmp_repo):
    """Default (no track_access injected) -- existing callers that don't care about
    access tracking keep working unchanged."""
    AddMemoryUseCase(tmp_repo).execute(collection="col", documents=["x"], ids=["a"])
    n = BoostConfidenceUseCase(tmp_repo).execute("col", ["a"], 0.1)
    assert n == 1


def test_positive_boost_bumps_access_count(tmp_repo, track_access):
    AddMemoryUseCase(tmp_repo).execute(collection="col", documents=["x"], ids=["a"])
    BoostConfidenceUseCase(tmp_repo, track_access).execute("col", ["a"], 0.1)
    meta = GetMemoryUseCase(tmp_repo).execute("col", ["a"])[0].metadata
    assert meta["access_count"] == 1


def test_decay_does_not_bump_access_count(tmp_repo, track_access):
    AddMemoryUseCase(tmp_repo).execute(
        collection="col", documents=["x"], ids=["a"], metadatas=[{"confidence": 0.5}]
    )
    BoostConfidenceUseCase(tmp_repo, track_access).execute("col", ["a"], -0.1)
    meta = GetMemoryUseCase(tmp_repo).execute("col", ["a"])[0].metadata
    assert meta.get("access_count", 0) == 0
