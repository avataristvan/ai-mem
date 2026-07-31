#!/usr/bin/env python3
"""Background query daemon — keeps one warm ChromaMemoryRepository across hook calls.

Each Claude Code hook (UserPromptSubmit, PreToolUse, PostToolUse) runs as a fresh
subprocess per event, so nothing built inside a hook script survives to the next
call. Importing chromadb + loading the ONNX embedding model costs ~1.1s on a cold
process — that cost can't be cached away inside a hook, only by moving the warm
state into a process that outlives individual hook invocations. This module is
that process: a small unix-socket server, spawned lazily by daemon_client.py on
the first cache miss, that stays warm for IDLE_TIMEOUT_SECONDS and then exits on
its own so it doesn't linger across unrelated projects/sessions.

Single-threaded by design: the ChromaDB PersistentClient (sqlite-backed) isn't
verified safe for concurrent access from multiple threads, and hook traffic is
low-frequency enough (one connection per user prompt / tool edit) that serializing
requests costs nothing in practice.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import sys
import time
from pathlib import Path

from ai_mem.daemon_protocol import (
    ACCEPT_POLL_SECONDS,
    IDLE_TIMEOUT_SECONDS,
    PROTOCOL_VERSION,
    pid_path,
    socket_path,
)

DB_PATH = Path(os.environ.get("AI_MEM_PATH", Path.home() / ".local" / "share" / "ai-mem"))


def _is_socket_alive(path: Path) -> bool:
    """True if a daemon is listening at *path* and answers a ping."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            sock.connect(str(path))
            sock.sendall(json.dumps({"op": "ping"}).encode() + b"\n")
            return bool(_read_line(sock))
    except OSError:
        return False


def _read_line(sock: socket.socket) -> bytes:
    chunks = []
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        chunks.append(chunk)
        if b"\n" in chunk:
            break
    return b"".join(chunks)


def stop_daemon(db_path: Path = DB_PATH) -> bool:
    """Gracefully stop a running daemon; falls back to SIGTERM by pid. Returns True if a
    daemon was found and a stop was attempted (does not guarantee it has fully exited).
    """
    sock_path = socket_path(db_path)
    stopped = False
    if sock_path.exists():
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(1.0)
                sock.connect(str(sock_path))
                sock.sendall(json.dumps({"op": "shutdown"}).encode() + b"\n")
                _read_line(sock)
            stopped = True
        except OSError:
            pass

    pfile = pid_path(db_path)
    if not stopped and pfile.exists():
        try:
            pid = int(pfile.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            stopped = True
        except (OSError, ValueError):
            pass

    # Best-effort cleanup in case the daemon didn't get to remove its own files
    # (e.g. it was killed by signal rather than the graceful shutdown path).
    sock_path.unlink(missing_ok=True)
    pfile.unlink(missing_ok=True)
    return stopped


def _handle_query(repo, payload: dict) -> dict:
    from ai_mem.application.query_memory import QueryMemoryUseCase
    from ai_mem.application.track_access import TrackAccessUseCase

    target_repo = repo
    if payload.get("with_bm25"):
        try:
            from ai_mem.infrastructure.bm25_repository import BM25MemoryRepository
            target_repo = BM25MemoryRepository(repo)
        except ImportError:
            pass

    query_uc = QueryMemoryUseCase(target_repo, TrackAccessUseCase(target_repo))
    results = query_uc.execute(
        collection=payload["collection"],
        query=payload["query"],
        n_results=payload.get("n_results", 5),
        max_age_days=payload.get("max_age_days"),
        type_filter=payload.get("type_filter"),
        min_score_for_tracking=payload.get("min_score_for_tracking"),
    )
    return {
        "ok": True,
        "results": [
            {"rank": r.rank, "id": r.id, "score": r.score, "text": r.text, "metadata": r.metadata}
            for r in results
        ],
    }


def _handle_conn(conn: socket.socket, repo) -> bool:
    """Process one request on *conn*. Returns True if the daemon should stop afterward."""
    raw = _read_line(conn)
    if not raw:
        return False

    try:
        payload = json.loads(raw.decode())
    except (UnicodeDecodeError, json.JSONDecodeError):
        conn.sendall(json.dumps({"ok": False, "error": "bad request"}).encode() + b"\n")
        return False

    op = payload.get("op")
    if op == "ping":
        conn.sendall(json.dumps({"ok": True, "version": PROTOCOL_VERSION, "pid": os.getpid()}).encode() + b"\n")
        return False
    if op == "shutdown":
        conn.sendall(json.dumps({"ok": True}).encode() + b"\n")
        return True
    if op == "query":
        try:
            response = _handle_query(repo, payload)
        except Exception as exc:  # noqa: BLE001 - report to client, never crash the daemon
            response = {"ok": False, "error": str(exc)}
        conn.sendall(json.dumps(response).encode() + b"\n")
        return False

    conn.sendall(json.dumps({"ok": False, "error": f"unknown op: {op!r}"}).encode() + b"\n")
    return False


def run(db_path: Path, idle_timeout_seconds: float = IDLE_TIMEOUT_SECONDS) -> None:
    """Run the daemon loop against *db_path* until idle-timeout or a shutdown request.

    Split out from main() so tests can run a real daemon against a tmp_path with a short
    idle_timeout_seconds instead of relying on the AI_MEM_PATH env var / module-level DB_PATH.
    """
    from ai_mem.infrastructure.chroma_repository import ChromaMemoryRepository

    db_path.mkdir(parents=True, exist_ok=True)
    sock_path = socket_path(db_path)
    pfile = pid_path(db_path)

    if sock_path.exists():
        if _is_socket_alive(sock_path):
            return  # another daemon is already serving this DB_PATH
        sock_path.unlink(missing_ok=True)

    server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_sock.bind(str(sock_path))
    server_sock.listen(5)
    server_sock.settimeout(ACCEPT_POLL_SECONDS)
    pfile.write_text(str(os.getpid()))

    repo = ChromaMemoryRepository(db_path)  # the one warm, expensive-to-build object
    last_activity = time.time()

    try:
        while True:
            try:
                conn, _ = server_sock.accept()
            except socket.timeout:
                if time.time() - last_activity > idle_timeout_seconds:
                    break
                continue

            last_activity = time.time()
            try:
                should_stop = _handle_conn(conn, repo)
            except Exception:  # noqa: BLE001 - one bad connection must not kill the daemon
                should_stop = False
            finally:
                conn.close()
            if should_stop:
                break
    finally:
        server_sock.close()
        sock_path.unlink(missing_ok=True)
        pfile.unlink(missing_ok=True)


def main() -> None:
    run(DB_PATH)


if __name__ == "__main__":
    main()
