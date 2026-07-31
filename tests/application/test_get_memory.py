"""GetMemoryUseCase: fetch by id, and the optional deliberate-use access tracking."""
from __future__ import annotations

from ai_mem.application.add_memory import AddMemoryUseCase
from ai_mem.application.get_memory import GetMemoryUseCase


def test_fetches_entries_by_id(tmp_repo):
    AddMemoryUseCase(tmp_repo).execute(collection="col", documents=["x", "y"], ids=["a", "b"])
    result = GetMemoryUseCase(tmp_repo).execute("col", ["a", "b"])
    assert {e.id for e in result} == {"a", "b"}


def test_missing_ids_silently_skipped(tmp_repo):
    AddMemoryUseCase(tmp_repo).execute(collection="col", documents=["x"], ids=["a"])
    result = GetMemoryUseCase(tmp_repo).execute("col", ["a", "does-not-exist"])
    assert len(result) == 1
    assert result[0].id == "a"


def test_without_track_access_does_not_bump_access_count(tmp_repo):
    """Default (no track_access injected) -- e.g. hook.py's SessionStart current_focus lookup,
    which is automatic plumbing, not a deliberate agent choice, and must not count as usage."""
    AddMemoryUseCase(tmp_repo).execute(collection="col", documents=["x"], ids=["a"])
    GetMemoryUseCase(tmp_repo).execute("col", ["a"])
    fetched = GetMemoryUseCase(tmp_repo).execute("col", ["a"])
    assert fetched[0].metadata.get("access_count", 0) == 0


def test_with_track_access_bumps_access_count_for_found_ids_only(tmp_repo, track_access):
    AddMemoryUseCase(tmp_repo).execute(collection="col", documents=["x"], ids=["a"])
    GetMemoryUseCase(tmp_repo, track_access).execute("col", ["a", "does-not-exist"])
    fetched = GetMemoryUseCase(tmp_repo).execute("col", ["a"])
    assert fetched[0].metadata["access_count"] == 1


def test_track_access_not_called_when_nothing_found(tmp_repo):
    calls = []
    fake_track_access = type("Fake", (), {"execute": lambda self, coll, ids: calls.append((coll, ids))})()
    GetMemoryUseCase(tmp_repo, fake_track_access).execute("col", ["does-not-exist"])
    assert calls == []
