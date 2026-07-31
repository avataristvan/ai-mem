"""Tests for ai_mem/daemon_client.py — hook-side connect-or-spawn-or-fallback logic."""
from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import patch

from ai_mem import daemon
from ai_mem.application.add_memory import AddMemoryUseCase
from ai_mem.daemon_client import DaemonQueryClient, DaemonUnavailable, get_daemon_query_uc
from ai_mem.infrastructure.chroma_repository import ChromaMemoryRepository


def _wait_for_socket(db_path: Path, timeout: float = 5.0) -> None:
    """Poll until the daemon actually answers a ping, not just until the socket file exists —
    see the identical helper in test_daemon.py for why file-existence alone isn't enough
    under full-suite load."""
    import time
    from ai_mem.daemon_client import _send
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = _send(db_path, {"op": "ping"}, timeout=0.5)
        if resp is not None and resp.get("ok"):
            return
        time.sleep(0.02)
    raise TimeoutError("daemon socket never became ready")


def test_get_daemon_query_uc_returns_none_and_spawns_when_no_daemon(tmp_path: Path):
    db_path = tmp_path / "chroma"  # nothing running here
    with patch("ai_mem.daemon_client._spawn_daemon") as spawn:
        result = get_daemon_query_uc(db_path, with_bm25=False)
    assert result is None
    spawn.assert_called_once_with(db_path)


def test_get_daemon_query_uc_returns_client_when_daemon_warm(tmp_path: Path):
    db_path = tmp_path / "chroma"
    thread = threading.Thread(target=daemon.run, args=(db_path,), kwargs={"idle_timeout_seconds": 60}, daemon=True)
    thread.start()
    try:
        _wait_for_socket(db_path)
        with patch("ai_mem.daemon_client._spawn_daemon") as spawn:
            result = get_daemon_query_uc(db_path, with_bm25=True)
        assert isinstance(result, DaemonQueryClient)
        spawn.assert_not_called()
    finally:
        daemon.stop_daemon(db_path)
        thread.join(timeout=2.0)


def test_get_daemon_query_uc_respawns_on_version_mismatch(tmp_path: Path):
    db_path = tmp_path / "chroma"
    thread = threading.Thread(target=daemon.run, args=(db_path,), kwargs={"idle_timeout_seconds": 60}, daemon=True)
    thread.start()
    try:
        _wait_for_socket(db_path)
        with (
            patch("ai_mem.daemon_client.PROTOCOL_VERSION", 999),
            patch("ai_mem.daemon_client._spawn_daemon") as spawn,
        ):
            result = get_daemon_query_uc(db_path, with_bm25=False)
        assert result is None
        spawn.assert_called_once_with(db_path)
    finally:
        daemon.stop_daemon(db_path)
        thread.join(timeout=2.0)


def test_daemon_query_client_execute_roundtrip(tmp_path: Path):
    db_path = tmp_path / "chroma"
    repo = ChromaMemoryRepository(db_path)
    AddMemoryUseCase(repo).execute(collection="global", documents=["the sky is blue"], ids=["fact1"])
    del repo

    thread = threading.Thread(target=daemon.run, args=(db_path,), kwargs={"idle_timeout_seconds": 60}, daemon=True)
    thread.start()
    try:
        _wait_for_socket(db_path)
        client = DaemonQueryClient(db_path, with_bm25=False)
        results = client.execute(collection="global", query="sky color", n_results=5)
        assert any(r.id == "fact1" for r in results)
    finally:
        daemon.stop_daemon(db_path)
        thread.join(timeout=2.0)


def test_daemon_query_client_raises_when_daemon_unreachable(tmp_path: Path):
    client = DaemonQueryClient(tmp_path / "chroma", with_bm25=False)
    try:
        client.execute(collection="global", query="anything")
        raised = False
    except DaemonUnavailable:
        raised = True
    assert raised
