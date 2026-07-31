"""Tests for pretool_hook.py — PreToolUse past-experience injection.

Regression coverage for the dead-import bug: `_build_query_use_case` used to
import `_build_core`/`_make_query_uc` from `ai_mem._hook_deps`, neither of
which exists there (only `_build_query_uc` does). The ImportError was
silently swallowed by main()'s broad except, so the hook never produced any
output despite being registered. No test file existed to catch it.
"""
from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

from ai_mem import pretool_hook as hook


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(score: float, text: str = "some memory text", entry_id: str | None = None) -> MagicMock:
    r = MagicMock()
    r.score = score
    r.text = text
    r.id = entry_id
    return r


def _payload(tool_name: str = "Write", file_path: str = "/tmp/foo.py") -> StringIO:
    return StringIO(json.dumps({
        "hook_event_name": "PreToolUse",
        "tool_name": tool_name,
        "tool_input": {"file_path": file_path},
    }))


def _run_main(
    tmp_path: Path,
    stdin_stream: StringIO,
    global_results=None,
    repo_results=None,
    repo_collection: str = "repo.my-project",
    has_claude_md: bool = True,
) -> str:
    """Run hook.main() with mocked dependencies; return captured stdout."""
    global_results = global_results or []
    repo_results = repo_results or []

    query_uc = MagicMock()

    def fake_execute(collection, query, n_results):
        return global_results if collection == "global" else repo_results

    query_uc.execute.side_effect = fake_execute

    ctx = MagicMock()
    ctx.collection = repo_collection
    ctx.has_claude_md = has_claude_md

    with (
        patch.object(sys, "stdin", stdin_stream),
        patch.object(hook, "DB_PATH", tmp_path),
        patch.object(hook, "_build_query_use_case", return_value=query_uc),
        patch.object(hook, "detect_repo_context", return_value=ctx),
        patch.object(hook, "GLOBAL_COLLECTION", "global"),
    ):
        tmp_path.mkdir(parents=True, exist_ok=True)
        captured: list[str] = []
        with patch("builtins.print", side_effect=lambda s: captured.append(s)):
            hook.main()

    return "\n".join(captured)


# ---------------------------------------------------------------------------
# 1. The dead-import regression itself
# ---------------------------------------------------------------------------

def test_build_query_use_case_does_not_raise(tmp_path: Path) -> None:
    """Regression: _build_query_use_case must actually construct a use case,
    not raise ImportError for names that don't exist in _hook_deps."""
    with patch.object(hook, "DB_PATH", tmp_path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        use_case = hook._build_query_use_case()
    assert use_case is not None
    assert hasattr(use_case, "execute")


# ---------------------------------------------------------------------------
# 2. Fires only for Write and Edit tools
# ---------------------------------------------------------------------------

def test_fires_for_write(tmp_path: Path) -> None:
    out = _run_main(tmp_path, _payload("Write"), global_results=[_make_result(0.9, "hit")])
    assert "hit" in out


def test_fires_for_edit(tmp_path: Path) -> None:
    out = _run_main(tmp_path, _payload("Edit"), global_results=[_make_result(0.9, "hit")])
    assert "hit" in out


def test_does_not_fire_for_read(tmp_path: Path) -> None:
    out = _run_main(tmp_path, _payload("Read"), global_results=[_make_result(0.9, "hit")])
    assert out == ""


# ---------------------------------------------------------------------------
# 3. No output when nothing found
# ---------------------------------------------------------------------------

def test_no_output_when_no_hits(tmp_path: Path) -> None:
    out = _run_main(tmp_path, _payload())
    assert out == ""


# ---------------------------------------------------------------------------
# 4. Injects both global and repo hits, correctly labeled
# ---------------------------------------------------------------------------

def test_injects_global_and_repo_hits(tmp_path: Path) -> None:
    out = _run_main(
        tmp_path,
        _payload(file_path="/tmp/repo/module.py"),
        global_results=[_make_result(0.8, "global lesson")],
        repo_results=[_make_result(0.7, "repo lesson")],
        repo_collection="repo.my-project",
        has_claude_md=True,
    )
    assert "global lesson" in out
    assert "repo lesson" in out
    assert "[global score=0.80]" in out
    assert "[repo.my-project score=0.70]" in out


def test_repo_hits_skipped_without_claude_md(tmp_path: Path) -> None:
    out = _run_main(
        tmp_path,
        _payload(),
        global_results=[],
        repo_results=[_make_result(0.9, "should not appear")],
        has_claude_md=False,
    )
    assert "should not appear" not in out


# ---------------------------------------------------------------------------
# 5. Missing file_path or DB path
# ---------------------------------------------------------------------------

def test_no_output_when_file_path_missing(tmp_path: Path) -> None:
    out = _run_main(tmp_path, _payload(file_path=""), global_results=[_make_result(0.9, "hit")])
    assert out == ""


def test_silently_exits_when_db_path_missing(tmp_path: Path) -> None:
    absent = tmp_path / "no_such_db"
    query_uc = MagicMock()
    query_uc.execute.return_value = [_make_result(0.9, "hit")]
    ctx = MagicMock()
    ctx.has_claude_md = True

    with (
        patch.object(sys, "stdin", _payload()),
        patch.object(hook, "DB_PATH", absent),
        patch.object(hook, "_build_query_use_case", return_value=query_uc),
        patch.object(hook, "detect_repo_context", return_value=ctx),
        patch.object(hook, "GLOBAL_COLLECTION", "global"),
    ):
        captured: list[str] = []
        with patch("builtins.print", side_effect=lambda s: captured.append(s)):
            hook.main()

    assert captured == []


# ---------------------------------------------------------------------------
# 6. Text truncation
# ---------------------------------------------------------------------------

def test_text_truncated_to_max_chars_per_hit(tmp_path: Path) -> None:
    long_text = "X" * (hook.MAX_CHARS_PER_HIT + 100)
    out = _run_main(tmp_path, _payload(), global_results=[_make_result(0.9, long_text)])

    assert long_text not in out
    assert "..." in out
