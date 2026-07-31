"""Tests for uninstall.py — stop_running_daemon."""
from __future__ import annotations

from unittest.mock import patch

import uninstall


def test_stop_running_daemon_calls_stop_daemon_with_db_path():
    with patch("ai_mem.daemon.stop_daemon", return_value=True) as stop_daemon:
        uninstall.stop_running_daemon(dry_run=False)
    stop_daemon.assert_called_once_with(uninstall.DB_PATH)


def test_stop_running_daemon_dry_run_does_not_call_stop_daemon():
    with patch("ai_mem.daemon.stop_daemon") as stop_daemon:
        uninstall.stop_running_daemon(dry_run=True)
    stop_daemon.assert_not_called()


def test_stop_running_daemon_silent_when_package_not_importable():
    with patch.dict("sys.modules", {"ai_mem.daemon": None}):
        uninstall.stop_running_daemon(dry_run=False)  # must not raise
