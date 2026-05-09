"""CleanupMemoryUseCase: TTL expiry + usage-based stale removal."""
from __future__ import annotations

import time

from ai_mem.application.add_memory import AddMemoryUseCase
from ai_mem.application.cleanup_memory import CleanupMemoryUseCase
from ai_mem.application.get_memory import GetMemoryUseCase


def _backdate(repo, collection: str, id_: str, days_ago: float) -> None:
    """Rewrite last_accessed_at to simulate aging via the public repository interface."""
    existing = repo.get_by_ids(collection, [id_])
    if not existing:
        return
    entry = existing[0]
    backdated = time.time() - days_ago * 86400
    entry.metadata["last_accessed_at"] = backdated
    entry.metadata["created_at"] = backdated
    repo.upsert(collection, [entry])


def test_stale_cleanup_removes_old_untouched_entries(tmp_repo):
    AddMemoryUseCase(tmp_repo).execute(
        collection="test_col",
        documents=["old never-touched", "fresh entry"],
        ids=["old", "fresh"],
    )
    _backdate(tmp_repo, "test_col", "old", days_ago=60)

    report = CleanupMemoryUseCase(tmp_repo).execute(
        collection="test_col", stale_after_days=30
    )
    assert report.collections["test_col"].stale == 1

    remaining = GetMemoryUseCase(tmp_repo).execute("test_col", ["old", "fresh"])
    assert {e.id for e in remaining} == {"fresh"}


def test_cleanup_without_stale_param_only_runs_ttl(tmp_repo):
    AddMemoryUseCase(tmp_repo).execute(
        collection="test_col",
        documents=["entry"],
        ids=["a"],
    )
    report = CleanupMemoryUseCase(tmp_repo).execute(collection="test_col")
    assert report.collections["test_col"].expired == 0
    assert report.collections["test_col"].stale == 0


def test_pattern_entries_exempt_from_stale_cleanup(tmp_repo):
    add = AddMemoryUseCase(tmp_repo)
    add.execute(
        collection="test_col",
        documents=["stale fact", "durable pattern"],
        ids=["fact", "pat"],
        metadatas=[{"type": "fact"}, {"type": "pattern"}],
    )
    _backdate(tmp_repo, "test_col", "fact", days_ago=60)
    _backdate(tmp_repo, "test_col", "pat", days_ago=60)

    CleanupMemoryUseCase(tmp_repo).execute(collection="test_col", stale_after_days=30)

    remaining = GetMemoryUseCase(tmp_repo).execute("test_col", ["fact", "pat"])
    assert {e.id for e in remaining} == {"pat"}


def _backdate_expiry(repo, collection: str, id_: str) -> None:
    """Set expires_at to the past so the entry is considered expired."""
    existing = repo.get_by_ids(collection, [id_])
    if not existing:
        return
    entry = existing[0]
    entry.metadata["expires_at"] = time.time() - 1
    repo.upsert(collection, [entry])


def test_pattern_entries_exempt_from_ttl_expiry(tmp_repo):
    add = AddMemoryUseCase(tmp_repo)
    add.execute(
        collection="test_col",
        documents=["expiring fact", "permanent pattern"],
        ids=["fact", "pat"],
        ttl_days=1,
        metadatas=[{"type": "fact"}, {"type": "pattern"}],
    )
    _backdate_expiry(tmp_repo, "test_col", "fact")
    _backdate_expiry(tmp_repo, "test_col", "pat")

    CleanupMemoryUseCase(tmp_repo).execute(collection="test_col")

    remaining = GetMemoryUseCase(tmp_repo).execute("test_col", ["fact", "pat"])
    assert {e.id for e in remaining} == {"pat"}
