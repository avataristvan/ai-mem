"""Shared constants and path helpers for the ai-mem background query daemon.

Bumping PROTOCOL_VERSION forces every client to treat an already-running daemon as
stale (see daemon_client.py) — bump it whenever daemon.py's request/response shape
changes, so a daemon left running from before a code update gets replaced instead
of silently serving an incompatible protocol forever.
"""
from __future__ import annotations

from pathlib import Path

PROTOCOL_VERSION = 1
IDLE_TIMEOUT_SECONDS = 1800
ACCEPT_POLL_SECONDS = 5
CONNECT_TIMEOUT_SECONDS = 0.15
REQUEST_TIMEOUT_SECONDS = 3.0


def socket_path(db_path: Path) -> Path:
    return db_path / "daemon.sock"


def pid_path(db_path: Path) -> Path:
    return db_path / "daemon.pid"
