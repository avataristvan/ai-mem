"""Tests for userprompt_hook.py — UserPromptSubmit hook."""
from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai_mem.domain.learning import TrainingExample, RankingFeatures
from ai_mem import userprompt_hook as hook


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(score: float, text: str = "some memory text") -> MagicMock:
    r = MagicMock()
    r.score = score
    r.text = text
    return r


def _make_example(labeled: bool) -> MagicMock:
    ex = MagicMock(spec=TrainingExample)
    ex.label = 1.0 if labeled else None
    return ex


def _make_storage(labeled_count: int) -> MagicMock:
    storage = MagicMock()
    examples = [_make_example(i < labeled_count) for i in range(max(labeled_count, 0))]
    storage.load_buffer.return_value = examples
    return storage


def _make_registry(scope_key: str = "global") -> MagicMock:
    registry = MagicMock()
    registry.scope_key.return_value = scope_key
    return registry


def _stdin_json(prompt: str = "what is the current focus?") -> StringIO:
    return StringIO(json.dumps({"hook_event_name": "UserPromptSubmit", "prompt": prompt}))


def _run_main(
    tmp_path: Path,
    stdin_stream: StringIO,
    global_results=None,
    repo_results=None,
    storage=None,
    registry=None,
    repo_collection: str | None = None,
) -> str:
    """Run hook.main() with mocked dependencies; return captured stdout."""
    global_results = global_results or []
    repo_results = repo_results or []
    storage = storage or _make_storage(0)
    registry = registry or _make_registry()

    query_uc = MagicMock()

    def fake_hits(collection, query, n_results):
        if collection == "global":
            return global_results
        return repo_results

    query_uc.execute.side_effect = fake_hits

    ctx = MagicMock()
    ctx.collection = repo_collection or "global"

    with (
        patch.object(sys, "stdin", stdin_stream),
        patch.object(hook, "DB_PATH", tmp_path),
        patch.object(hook, "_build_deps", return_value=(query_uc, storage, registry)),
        patch.object(hook, "detect_repo_context", return_value=ctx),
        patch.object(hook, "GLOBAL_COLLECTION", "global"),
        patch.object(hook, "WORKSPACE_COLLECTION", "workspace"),
    ):
        tmp_path.mkdir(parents=True, exist_ok=True)
        captured = []
        with patch("builtins.print", side_effect=lambda s: captured.append(s)):
            hook.main()

    return "\n".join(captured)


# ---------------------------------------------------------------------------
# 1. No output when labeled examples < MIN_LABELED_EXAMPLES
# ---------------------------------------------------------------------------

def test_no_output_when_too_few_labeled_examples(tmp_path: Path) -> None:
    results = [_make_result(0.9), _make_result(0.85), _make_result(0.80)]
    storage = _make_storage(labeled_count=5)  # below MIN_LABELED_EXAMPLES=10
    registry = _make_registry("global")

    out = _run_main(tmp_path, _stdin_json(), global_results=results, storage=storage, registry=registry)

    assert out == ""


# ---------------------------------------------------------------------------
# 2. No output when average top-3 score < MIN_AVG_SCORE
# ---------------------------------------------------------------------------

def test_no_output_when_avg_score_below_threshold(tmp_path: Path) -> None:
    results = [_make_result(0.4), _make_result(0.3), _make_result(0.2)]
    storage = _make_storage(labeled_count=15)
    registry = _make_registry("global")

    out = _run_main(tmp_path, _stdin_json(), global_results=results, storage=storage, registry=registry)

    assert out == ""


# ---------------------------------------------------------------------------
# 3. Injects context when both conditions are met
# ---------------------------------------------------------------------------

def test_injects_context_when_conditions_met(tmp_path: Path) -> None:
    results = [_make_result(0.9, "memory A"), _make_result(0.8, "memory B"), _make_result(0.75, "memory C")]
    storage = _make_storage(labeled_count=12)
    registry = _make_registry("global")

    out = _run_main(tmp_path, _stdin_json(), global_results=results, storage=storage, registry=registry)

    parsed = json.loads(out)
    ctx = parsed["hookSpecificOutput"]["additionalContext"]
    assert "ai-mem" in ctx
    assert "memory A" in ctx
    assert parsed["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"


# ---------------------------------------------------------------------------
# 4. Missing DB path — exits silently
# ---------------------------------------------------------------------------

def test_silently_exits_when_db_path_missing(tmp_path: Path) -> None:
    absent = tmp_path / "no_such_db"

    with (
        patch.object(sys, "stdin", _stdin_json()),
        patch.object(hook, "DB_PATH", absent),
    ):
        captured = []
        with patch("builtins.print", side_effect=lambda s: captured.append(s)):
            hook.main()

    assert captured == []


# ---------------------------------------------------------------------------
# 5. Bad stdin JSON — exits silently
# ---------------------------------------------------------------------------

def test_silently_exits_on_bad_json(tmp_path: Path) -> None:
    with (
        patch.object(sys, "stdin", StringIO("not json {")),
        patch.object(hook, "DB_PATH", tmp_path),
    ):
        tmp_path.mkdir(parents=True, exist_ok=True)
        captured = []
        with patch("builtins.print", side_effect=lambda s: captured.append(s)):
            hook.main()

    assert captured == []


# ---------------------------------------------------------------------------
# 6. Queries both global and repo collection when they differ
# ---------------------------------------------------------------------------

def test_queries_both_global_and_repo_collection(tmp_path: Path) -> None:
    global_results = [_make_result(0.9, "global memory")]
    repo_results = [_make_result(0.85, "repo memory")]

    global_storage = _make_storage(labeled_count=12)
    repo_storage = _make_storage(labeled_count=15)

    # Storage returns different labeled counts depending on scope_key arg
    storage = MagicMock()
    def fake_load(scope_key):
        if scope_key == "global":
            return global_storage.load_buffer(scope_key)
        return repo_storage.load_buffer(scope_key)
    storage.load_buffer.side_effect = fake_load

    registry = MagicMock()
    def fake_scope_key(collection):
        return collection
    registry.scope_key.side_effect = fake_scope_key

    out = _run_main(
        tmp_path,
        _stdin_json(),
        global_results=global_results,
        repo_results=repo_results,
        storage=storage,
        registry=registry,
        repo_collection="repo.my-project",
    )

    parsed = json.loads(out)
    ctx = parsed["hookSpecificOutput"]["additionalContext"]
    assert "global memory" in ctx
    assert "repo memory" in ctx
