"""Tests for posttool_hook.py — PostToolUse passive training signal."""
from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from ai_mem import posttool_hook as hook


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _payload(tool_name: str = "Write", file_path: str = "/tmp/foo.py") -> StringIO:
    return StringIO(json.dumps({
        "hook_event_name": "PostToolUse",
        "tool_name": tool_name,
        "tool_input": {"file_path": file_path},
        "tool_response": {},
    }))


def _run_main(
    tmp_path: Path,
    stdin_stream: StringIO,
    repo_collection: str = "repo.my-project",
) -> tuple:
    """Run hook.main() with mocked DI; return captured stdout (must be empty)."""
    query_uc = MagicMock()
    query_uc.execute.return_value = []

    train_uc = MagicMock()

    ctx = MagicMock()
    ctx.collection = repo_collection

    with (
        patch.object(sys, "stdin", stdin_stream),
        patch.object(hook, "DB_PATH", tmp_path),
        patch.object(hook, "_build_deps", return_value=(query_uc, train_uc)),
        patch.object(hook, "detect_repo_context", return_value=ctx),
        patch.object(hook, "GLOBAL_COLLECTION", "global"),
        patch.object(hook, "WORKSPACE_COLLECTION", "workspace"),
    ):
        tmp_path.mkdir(parents=True, exist_ok=True)
        captured = []
        with patch("builtins.print", side_effect=lambda *a, **kw: captured.append(a)):
            hook.main()

    return captured, query_uc, train_uc


# ---------------------------------------------------------------------------
# 1. Hook only fires for Write and Edit tools
# ---------------------------------------------------------------------------

def test_fires_for_write(tmp_path: Path) -> None:
    captured, query_uc, train_uc = _run_main(tmp_path, _payload("Write"))
    assert query_uc.execute.called


def test_fires_for_edit(tmp_path: Path) -> None:
    captured, query_uc, train_uc = _run_main(tmp_path, _payload("Edit"))
    assert query_uc.execute.called


def test_does_not_fire_for_read(tmp_path: Path) -> None:
    captured, query_uc, train_uc = _run_main(tmp_path, _payload("Read"))
    assert not query_uc.execute.called
    assert not train_uc.train_step.called


def test_does_not_fire_for_bash(tmp_path: Path) -> None:
    captured, query_uc, train_uc = _run_main(tmp_path, _payload("Bash"))
    assert not query_uc.execute.called


# ---------------------------------------------------------------------------
# 2. train_step is called after query
# ---------------------------------------------------------------------------

def test_train_step_called_for_global(tmp_path: Path) -> None:
    captured, query_uc, train_uc = _run_main(tmp_path, _payload(), repo_collection="global")
    train_uc.train_step.assert_called()
    call_collections = [c.kwargs.get("collection") or c.args[0] for c in train_uc.train_step.call_args_list]
    assert "global" in call_collections


def test_train_step_called_for_both_collections_when_repo_differs(tmp_path: Path) -> None:
    captured, query_uc, train_uc = _run_main(tmp_path, _payload(), repo_collection="repo.my-project")
    assert train_uc.train_step.call_count == 2
    call_collections = {
        (c.kwargs.get("collection") or c.args[0])
        for c in train_uc.train_step.call_args_list
    }
    assert "global" in call_collections
    assert "repo.my-project" in call_collections


def test_train_step_not_called_twice_for_global_only(tmp_path: Path) -> None:
    # When repo collection == global, we should only signal global once.
    captured, query_uc, train_uc = _run_main(tmp_path, _payload(), repo_collection="global")
    assert train_uc.train_step.call_count == 1


# ---------------------------------------------------------------------------
# 3. No output — hook is fully silent
# ---------------------------------------------------------------------------

def test_no_stdout_output_on_success(tmp_path: Path) -> None:
    captured, _, _ = _run_main(tmp_path, _payload())
    assert captured == []


def test_no_stdout_output_on_write(tmp_path: Path) -> None:
    captured, _, _ = _run_main(tmp_path, _payload("Write", "/repo/src/main.py"))
    assert captured == []


# ---------------------------------------------------------------------------
# 4. Silent failure cases
# ---------------------------------------------------------------------------

def test_silently_exits_on_bad_json(tmp_path: Path) -> None:
    with (
        patch.object(sys, "stdin", StringIO("not json {")),
        patch.object(hook, "DB_PATH", tmp_path),
    ):
        tmp_path.mkdir(parents=True, exist_ok=True)
        captured = []
        with patch("builtins.print", side_effect=lambda *a, **kw: captured.append(a)):
            hook.main()
    assert captured == []


def test_silently_exits_when_db_path_missing(tmp_path: Path) -> None:
    absent = tmp_path / "no_such_db"
    with (
        patch.object(sys, "stdin", _payload()),
        patch.object(hook, "DB_PATH", absent),
    ):
        captured = []
        with patch("builtins.print", side_effect=lambda *a, **kw: captured.append(a)):
            hook.main()
    assert captured == []


def test_silently_exits_when_build_deps_raises(tmp_path: Path) -> None:
    with (
        patch.object(sys, "stdin", _payload()),
        patch.object(hook, "DB_PATH", tmp_path),
        patch.object(hook, "_build_deps", side_effect=ImportError("no chromadb")),
    ):
        tmp_path.mkdir(parents=True, exist_ok=True)
        captured = []
        with patch("builtins.print", side_effect=lambda *a, **kw: captured.append(a)):
            hook.main()
    assert captured == []


def test_silently_exits_when_file_path_missing(tmp_path: Path) -> None:
    payload = StringIO(json.dumps({
        "hook_event_name": "PostToolUse",
        "tool_name": "Write",
        "tool_input": {},
        "tool_response": {},
    }))
    query_uc = MagicMock()
    train_uc = MagicMock()
    with (
        patch.object(sys, "stdin", payload),
        patch.object(hook, "DB_PATH", tmp_path),
        patch.object(hook, "_build_deps", return_value=(query_uc, train_uc)),
    ):
        tmp_path.mkdir(parents=True, exist_ok=True)
        captured = []
        with patch("builtins.print", side_effect=lambda *a, **kw: captured.append(a)):
            hook.main()
    assert captured == []
    assert not query_uc.execute.called


# ---------------------------------------------------------------------------
# 5. Query exception does not propagate (train_step is still skipped safely)
# ---------------------------------------------------------------------------

def test_query_exception_is_swallowed(tmp_path: Path) -> None:
    query_uc = MagicMock()
    query_uc.execute.side_effect = RuntimeError("chroma down")
    train_uc = MagicMock()
    ctx = MagicMock()
    ctx.collection = "global"

    with (
        patch.object(sys, "stdin", _payload()),
        patch.object(hook, "DB_PATH", tmp_path),
        patch.object(hook, "_build_deps", return_value=(query_uc, train_uc)),
        patch.object(hook, "detect_repo_context", return_value=ctx),
        patch.object(hook, "GLOBAL_COLLECTION", "global"),
        patch.object(hook, "WORKSPACE_COLLECTION", "workspace"),
    ):
        tmp_path.mkdir(parents=True, exist_ok=True)
        captured = []
        with patch("builtins.print", side_effect=lambda *a, **kw: captured.append(a)):
            hook.main()

    assert captured == []
    # train_step was not called because query failed
    assert not train_uc.train_step.called


# ---------------------------------------------------------------------------
# 6. Query is constructed from file name and path
# ---------------------------------------------------------------------------

def test_query_includes_filename_and_path(tmp_path: Path) -> None:
    captured, query_uc, _ = _run_main(
        tmp_path,
        _payload("Edit", "/workspace/ai-mem/ai_mem/server.py"),
        repo_collection="global",
    )
    assert query_uc.execute.called
    _, call_kwargs = query_uc.execute.call_args
    query_str = call_kwargs.get("query") or query_uc.execute.call_args.args[1] if query_uc.execute.call_args.args else ""
    # Fallback: check via any call
    all_queries = []
    for c in query_uc.execute.call_args_list:
        q = c.kwargs.get("query") or (c.args[1] if len(c.args) > 1 else "")
        all_queries.append(q)
    assert any("server.py" in q for q in all_queries)
