"""Client-side helper for the ai-mem background query daemon.

Hooks call get_daemon_query_uc(); it either returns a thin object whose .execute()
proxies to an already-warm daemon (fast path — see daemon.py), or returns None so
the caller falls back to today's cold in-process build. A cache miss also spawns a
fresh daemon detached in the background, on a best-effort basis, so the *next*
hook invocation in the same session gets the fast path — the current invocation
never waits for that startup.
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
from pathlib import Path

from ai_mem.daemon_protocol import (
    CONNECT_TIMEOUT_SECONDS,
    PROTOCOL_VERSION,
    REQUEST_TIMEOUT_SECONDS,
    socket_path,
)


class DaemonUnavailable(Exception):
    pass


class DaemonQueryClient:
    """Drop-in stand-in for QueryMemoryUseCase, backed by the warm daemon process."""

    def __init__(self, db_path: Path, with_bm25: bool) -> None:
        self._db_path = db_path
        self._with_bm25 = with_bm25

    def execute(
        self,
        collection: str,
        query: str,
        n_results: int = 5,
        max_age_days: float | None = None,
        type_filter: str | None = None,
        min_score_for_tracking: float | None = None,
    ):
        from ai_mem.domain.memory import QueryResult

        request = {
            "op": "query",
            "collection": collection,
            "query": query,
            "n_results": n_results,
            "max_age_days": max_age_days,
            "type_filter": type_filter,
            "min_score_for_tracking": min_score_for_tracking,
            "with_bm25": self._with_bm25,
        }
        response = _send(self._db_path, request, timeout=REQUEST_TIMEOUT_SECONDS)
        if response is None or not response.get("ok"):
            raise DaemonUnavailable(response.get("error") if response else "no response")
        return [QueryResult(**r) for r in response["results"]]


def _send(db_path: Path, payload: dict, timeout: float) -> dict | None:
    sock_path = socket_path(db_path)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(str(sock_path))
            sock.sendall(json.dumps(payload).encode() + b"\n")
            chunks = []
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
                if b"\n" in chunk:
                    break
            raw = b"".join(chunks)
        return json.loads(raw.decode()) if raw else None
    except (OSError, json.JSONDecodeError):
        return None


def _spawn_daemon(db_path: Path) -> None:
    """Best-effort, fire-and-forget: start the daemon detached so it outlives this hook."""
    try:
        subprocess.Popen(
            [sys.executable, "-m", "ai_mem.daemon"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        pass


def get_daemon_query_uc(db_path: Path, with_bm25: bool) -> DaemonQueryClient | None:
    """Return a warm daemon-backed client, or None if unavailable right now.

    Never blocks the caller waiting for a daemon to start — a miss spawns one in the
    background for next time and returns None immediately so the caller falls back
    to building its own (slower, but self-contained) query path for this call.
    """
    ping = _send(db_path, {"op": "ping"}, timeout=CONNECT_TIMEOUT_SECONDS)
    if ping is None or not ping.get("ok") or ping.get("version") != PROTOCOL_VERSION:
        _spawn_daemon(db_path)
        return None
    return DaemonQueryClient(db_path, with_bm25)
