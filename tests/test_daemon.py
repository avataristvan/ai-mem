"""Tests for ai_mem/daemon.py — the warm background query server."""
from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from ai_mem import daemon
from ai_mem.application.add_memory import AddMemoryUseCase
from ai_mem.daemon_protocol import PROTOCOL_VERSION, pid_path, socket_path
from ai_mem.infrastructure.chroma_repository import ChromaMemoryRepository


def _send(db_path: Path, payload: dict, timeout: float = 2.0) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(str(socket_path(db_path)))
        sock.sendall(json.dumps(payload).encode() + b"\n")
        chunks = []
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
    return json.loads(b"".join(chunks).decode())


def _wait_for_socket(db_path: Path, timeout: float = 5.0) -> None:
    """Poll until the daemon actually answers a ping, not just until the socket file exists —
    bind() creates the file before the daemon is necessarily scheduled to accept()/respond,
    and under full-suite load (thread scheduling contention) that gap can exceed a short
    fixed sleep. Retrying an end-to-end ping is what the caller actually needs to be true."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if socket_path(db_path).exists():
            resp = _send(db_path, {"op": "ping"}, timeout=0.5)
            if resp is not None and resp.get("ok"):
                return
        time.sleep(0.02)
    raise TimeoutError("daemon socket never became ready")


@pytest.fixture
def running_daemon(tmp_path: Path):
    """Start a real daemon against tmp_path in a background thread; stop it afterward."""
    db_path = tmp_path / "chroma"
    thread = threading.Thread(target=daemon.run, args=(db_path,), kwargs={"idle_timeout_seconds": 60}, daemon=True)
    thread.start()
    _wait_for_socket(db_path)
    yield db_path
    daemon.stop_daemon(db_path)
    thread.join(timeout=2.0)


def test_ping_returns_version_and_pid(running_daemon: Path):
    resp = _send(running_daemon, {"op": "ping"})
    assert resp["ok"] is True
    assert resp["version"] == PROTOCOL_VERSION
    assert isinstance(resp["pid"], int)


def test_query_roundtrip(tmp_path: Path):
    db_path = tmp_path / "chroma"
    repo = ChromaMemoryRepository(db_path)
    AddMemoryUseCase(repo).execute(
        collection="global", documents=["the sky is blue"], ids=["fact1"],
    )
    del repo  # release before the daemon opens its own PersistentClient on the same path

    thread = threading.Thread(target=daemon.run, args=(db_path,), kwargs={"idle_timeout_seconds": 60}, daemon=True)
    thread.start()
    try:
        _wait_for_socket(db_path)
        resp = _send(db_path, {
            "op": "query", "collection": "global", "query": "sky color",
            "n_results": 5, "with_bm25": False,
        })
        assert resp["ok"] is True
        assert any(r["id"] == "fact1" for r in resp["results"])
    finally:
        daemon.stop_daemon(db_path)
        thread.join(timeout=2.0)


def test_query_with_bm25_wraps_repo(tmp_path: Path):
    """with_bm25=True must not error even though rank_bm25 may or may not be installed."""
    db_path = tmp_path / "chroma"
    repo = ChromaMemoryRepository(db_path)
    AddMemoryUseCase(repo).execute(collection="global", documents=["hello world"], ids=["e1"])
    del repo

    thread = threading.Thread(target=daemon.run, args=(db_path,), kwargs={"idle_timeout_seconds": 60}, daemon=True)
    thread.start()
    try:
        _wait_for_socket(db_path)
        resp = _send(db_path, {
            "op": "query", "collection": "global", "query": "hello",
            "n_results": 5, "with_bm25": True,
        })
        assert resp["ok"] is True
    finally:
        daemon.stop_daemon(db_path)
        thread.join(timeout=2.0)


def test_shutdown_stops_daemon(running_daemon: Path):
    resp = _send(running_daemon, {"op": "shutdown"})
    assert resp["ok"] is True
    deadline = time.time() + 2.0
    while time.time() < deadline and socket_path(running_daemon).exists():
        time.sleep(0.02)
    assert not socket_path(running_daemon).exists()
    assert not pid_path(running_daemon).exists()


def test_unknown_op_returns_error(running_daemon: Path):
    resp = _send(running_daemon, {"op": "not_a_real_op"})
    assert resp["ok"] is False


def test_second_daemon_on_same_path_is_a_noop(running_daemon: Path):
    """run() must detect an already-alive daemon at the same socket and return immediately
    instead of crashing on AddressInUse or serving a second, conflicting instance."""
    daemon.run(running_daemon, idle_timeout_seconds=60)  # returns fast, doesn't hang
    # the original daemon (from the fixture) must still be the one answering
    resp = _send(running_daemon, {"op": "ping"})
    assert resp["ok"] is True


def test_idle_timeout_exits_daemon(tmp_path: Path):
    db_path = tmp_path / "chroma"
    with patch.object(daemon, "ACCEPT_POLL_SECONDS", 0.05):
        thread = threading.Thread(
            target=daemon.run, args=(db_path,), kwargs={"idle_timeout_seconds": 0.1}, daemon=True
        )
        thread.start()
        _wait_for_socket(db_path)
        thread.join(timeout=3.0)
        assert not thread.is_alive()
        assert not socket_path(db_path).exists()


def test_stop_daemon_returns_false_when_nothing_running(tmp_path: Path):
    assert daemon.stop_daemon(tmp_path / "chroma") is False


def test_concurrent_connections_are_not_serialized(running_daemon: Path) -> None:
    """Two connections handled at once must overlap, not queue behind each other --
    the whole point of moving off the single-threaded accept-handle-close loop."""
    n_conns = 3
    delay = 0.3

    def slow_handle_query(repo, payload):
        time.sleep(delay)
        return {"ok": True, "results": []}

    results = [None] * n_conns

    def worker(idx: int) -> None:
        results[idx] = _send(running_daemon, {"op": "query", "collection": "global", "query": "x"})

    with patch.object(daemon, "_handle_query", side_effect=slow_handle_query):
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_conns)]
        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
        elapsed = time.time() - start

    assert all(r is not None and r["ok"] for r in results)
    # Serialized, this would take n_conns * delay (~0.9s); concurrent, ~delay (~0.3s) plus
    # scheduling slack. The threshold sits well below the serialized time to catch a regression
    # to single-threaded handling without being sensitive to normal scheduling jitter.
    assert elapsed < n_conns * delay * 0.75


def test_stop_daemon_via_pid_fallback_when_socket_stale(tmp_path: Path):
    """A daemon that dies without cleaning up (e.g. killed by signal) leaves a stale,
    unconnectable socket file; stop_daemon must fall back to killing the recorded pid."""
    import subprocess

    db_path = tmp_path / "chroma"
    db_path.mkdir(parents=True)
    socket_path(db_path).touch()  # present on disk, but nothing is listening
    proc = subprocess.Popen(["sleep", "30"])
    pid_path(db_path).write_text(str(proc.pid))

    try:
        assert daemon.stop_daemon(db_path) is True
        assert proc.wait(timeout=2.0) != 0  # killed by SIGTERM
    finally:
        proc.kill()
        proc.wait()
    assert not socket_path(db_path).exists()
    assert not pid_path(db_path).exists()
