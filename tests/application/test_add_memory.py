"""AddMemoryUseCase: timestamp injection, TTL, history preservation on update."""
from __future__ import annotations

from datetime import datetime, timezone

from ai_mem.application.add_memory import AddMemoryUseCase
from ai_mem.application.get_memory import GetMemoryUseCase
from ai_mem.application.track_access import TrackAccessUseCase


def test_created_at_is_injected(tmp_repo):
    use_case = AddMemoryUseCase(tmp_repo)
    before = datetime.now(tz=timezone.utc).timestamp()
    use_case.execute(collection="test_col", documents=["hi"], ids=["a"])
    after = datetime.now(tz=timezone.utc).timestamp()

    fetched = GetMemoryUseCase(tmp_repo).execute("test_col", ["a"])
    assert len(fetched) == 1
    meta = fetched[0].metadata
    assert before <= meta["created_at"] <= after
    assert meta["last_accessed_at"] == meta["created_at"]
    assert meta["access_count"] == 0
    assert "expires_at" not in meta


def test_ttl_sets_expires_at(tmp_repo):
    AddMemoryUseCase(tmp_repo).execute(
        collection="test_col", documents=["hi"], ids=["a"], ttl_days=1.0
    )
    fetched = GetMemoryUseCase(tmp_repo).execute("test_col", ["a"])
    delta = fetched[0].metadata["expires_at"] - fetched[0].metadata["created_at"]
    assert abs(delta - 86400) < 1.0


def test_readd_preserves_access_history(tmp_repo):
    """Re-adding an existing id keeps created_at, last_accessed_at, access_count."""
    add = AddMemoryUseCase(tmp_repo)
    track = TrackAccessUseCase(tmp_repo)

    add.execute(collection="test_col", documents=["v1"], ids=["a"])
    track.execute("test_col", ["a"])
    track.execute("test_col", ["a"])

    before = GetMemoryUseCase(tmp_repo).execute("test_col", ["a"])[0].metadata
    assert before["access_count"] == 2

    add.execute(collection="test_col", documents=["v2 updated text"], ids=["a"])

    after = GetMemoryUseCase(tmp_repo).execute("test_col", ["a"])[0]
    assert after.text == "v2 updated text"
    assert after.metadata["created_at"] == before["created_at"]
    assert after.metadata["last_accessed_at"] == before["last_accessed_at"]
    assert after.metadata["access_count"] == 2
