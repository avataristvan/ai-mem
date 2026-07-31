"""Tests for install.py — _upsert_hook_command dedup logic."""
from __future__ import annotations

from unittest.mock import patch

import install


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hook(command: str, timeout: int = 5) -> dict:
    return {"type": "command", "command": command, "timeout": timeout}


def _group(hooks: list[dict], matcher: str | None = None) -> dict:
    entry = {"hooks": hooks}
    if matcher is not None:
        entry["matcher"] = matcher
    return entry


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_matcher_not_overwritten_when_group_has_unrelated_hook():
    """A hook group that bundles an ai-mem hook alongside a user's own hook must keep
    its own matcher untouched — ai-mem should not claim a group it doesn't own outright."""
    ai_mem_hook = _hook("python3 -m ai_mem.sometest_hook ")
    unrelated_hook = _hook("some-other-tool --do-thing")
    entries = [_group([ai_mem_hook, unrelated_hook], matcher="UserPromptSubmit")]

    install._upsert_hook_command(
        entries, "ai_mem.sometest_hook", "python3 -m ai_mem.sometest_hook ", 5, matcher="SessionStart"
    )

    assert entries[0]["matcher"] == "UserPromptSubmit"
    assert unrelated_hook in entries[0]["hooks"]
    assert len(entries[0]["hooks"]) == 2


def test_matcher_overwritten_when_group_is_ai_mem_only():
    """When the group ends up holding only the ai-mem hook, the matcher may be updated."""
    ai_mem_hook = _hook("python3 -m ai_mem.sometest_hook ")
    entries = [_group([ai_mem_hook], matcher="OldMatcher")]

    install._upsert_hook_command(
        entries, "ai_mem.sometest_hook", "python3 -m ai_mem.sometest_hook ", 5, matcher="NewMatcher"
    )

    assert entries[0]["matcher"] == "NewMatcher"


def test_remove_by_identity_not_value_equality():
    """Two dict-value-equal empty groups in `entries` must not cause list.remove to drop
    the wrong one — removal must target the specific emptied entry by identity."""
    unrelated_empty = {"hooks": []}
    first_dup = _group([_hook("python3 -m ai_mem.sometest_hook ")])
    second_dup = _group([_hook("python -m ai_mem.sometest_hook ")])
    entries = [unrelated_empty, first_dup, second_dup]

    install._upsert_hook_command(
        entries, "ai_mem.sometest_hook", "python3 -m ai_mem.sometest_hook ", 5
    )

    # first_dup is kept (updated in place), second_dup gets emptied and removed by
    # identity — the pre-existing unrelated empty entry must survive untouched.
    assert any(e is unrelated_empty for e in entries)
    assert any(e is first_dup for e in entries)
    assert not any(e is second_dup for e in entries)
    assert len(entries) == 2


# ---------------------------------------------------------------------------
# stop_running_daemon
# ---------------------------------------------------------------------------

def test_stop_running_daemon_calls_stop_daemon_with_db_path():
    with patch("ai_mem.daemon.stop_daemon", return_value=True) as stop_daemon:
        install.stop_running_daemon()
    stop_daemon.assert_called_once_with(install.DB_PATH)


def test_stop_running_daemon_silent_when_package_not_yet_installed():
    """A fresh machine has no ai_mem.daemon to import yet — must not raise."""
    with patch.dict("sys.modules", {"ai_mem.daemon": None}):
        install.stop_running_daemon()  # must not raise
