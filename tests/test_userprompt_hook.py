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

def _make_result(score: float, text: str = "some memory text", entry_id: str | None = None) -> MagicMock:
    r = MagicMock()
    r.score = score
    r.text = text
    r.id = entry_id
    return r


def _make_example(labeled: bool) -> MagicMock:
    ex = MagicMock(spec=TrainingExample)
    ex.target_future_access = 1.0 if labeled else None
    return ex


def _make_storage(labeled_count: int) -> MagicMock:
    storage = MagicMock()
    examples = [_make_example(i < labeled_count) for i in range(max(labeled_count, 0))]
    storage.load_buffer.return_value = examples
    storage.labeled_count.return_value = labeled_count
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
    antipattern_results=None,
    dilemma_results=None,
    storage=None,
    registry=None,
    repo_collection: str | None = None,
) -> str:
    """Run hook.main() with mocked dependencies; return captured stdout."""
    global_results = global_results or []
    repo_results = repo_results or []
    antipattern_results = antipattern_results or []
    dilemma_results = dilemma_results or []
    storage = storage or _make_storage(0)
    registry = registry or _make_registry()

    query_uc = MagicMock()

    def fake_hits(collection, query, n_results, type_filter=None, max_age_days=None):
        if type_filter == "anti-pattern":
            return antipattern_results
        if type_filter == "dilemma":
            return dilemma_results
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

    def fake_labeled_count(scope_key):
        if scope_key == "global":
            return global_storage.labeled_count(scope_key)
        return repo_storage.labeled_count(scope_key)
    storage.labeled_count.side_effect = fake_labeled_count

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


# ---------------------------------------------------------------------------
# 7. Per-session dedup: already-injected IDs are filtered out
# ---------------------------------------------------------------------------

def test_dedup_filters_already_injected_ids(tmp_path: Path) -> None:
    import json, time

    results = [
        _make_result(0.9, "memory A", entry_id="id-1"),
        _make_result(0.85, "memory B", entry_id="id-2"),
        _make_result(0.80, "memory C", entry_id="id-3"),
    ]
    storage = _make_storage(labeled_count=12)
    registry = _make_registry("global")

    # Pre-populate the session file with id-1 already seen.
    session_file = tmp_path / "session_injected.json"
    session_file.write_text(json.dumps({"session_ts": time.time(), "ids": ["id-1"]}))

    out = _run_main(tmp_path, _stdin_json(), global_results=results, storage=storage, registry=registry)

    parsed = json.loads(out)
    ctx = parsed["hookSpecificOutput"]["additionalContext"]
    assert "memory A" not in ctx
    assert "memory B" in ctx


# ---------------------------------------------------------------------------
# 8. Budget cap stops injection after MAX_TOTAL_CHARS
# ---------------------------------------------------------------------------

def test_budget_cap_stops_after_max_total_chars(tmp_path: Path) -> None:
    # Each entry text is 400 chars; MAX_CHARS_PER_HIT=300, MAX_TOTAL_CHARS=1500
    # → entry_len per hit = 300; 5 entries = 1500 chars (exactly at limit), 6th would exceed
    long_text = "x" * 400
    results = [_make_result(0.9 - i * 0.01, long_text, entry_id=f"id-{i}") for i in range(7)]
    storage = _make_storage(labeled_count=12)
    registry = _make_registry("global")

    out = _run_main(tmp_path, _stdin_json(), global_results=results, storage=storage, registry=registry)

    parsed = json.loads(out)
    ctx = parsed["hookSpecificOutput"]["additionalContext"]
    # 5 entries fit exactly (5 × 300 = 1500), 6th and 7th are cut.
    assert ctx.count("- [global") == 5


# ---------------------------------------------------------------------------
# 9. Expired session file is treated as a new session (dedup ignored)
# ---------------------------------------------------------------------------

def test_expired_session_file_is_ignored(tmp_path: Path) -> None:
    import json, time

    results = [
        _make_result(0.9, "memory A", entry_id="id-1"),
        _make_result(0.85, "memory B", entry_id="id-2"),
    ]
    storage = _make_storage(labeled_count=12)
    registry = _make_registry("global")

    # Write a session file that is 5 hours old (beyond SESSION_TTL_HOURS=4).
    old_ts = time.time() - 5 * 3600
    session_file = tmp_path / "session_injected.json"
    session_file.write_text(json.dumps({"session_ts": old_ts, "ids": ["id-1", "id-2"]}))

    out = _run_main(tmp_path, _stdin_json(), global_results=results, storage=storage, registry=registry)

    parsed = json.loads(out)
    ctx = parsed["hookSpecificOutput"]["additionalContext"]
    # Both entries must appear because the old session file is expired.
    assert "memory A" in ctx
    assert "memory B" in ctx


# ---------------------------------------------------------------------------
# 11. Anti-pattern warnings fire without ranker qualification
# ---------------------------------------------------------------------------

def test_antipattern_fires_without_ranker_qualification(tmp_path: Path) -> None:
    storage = _make_storage(labeled_count=0)  # far below MIN_LABELED_EXAMPLES
    registry = _make_registry("repo.my-project")
    ap = _make_result(0.85, "Tried: X\nFailed because: Y\nInstead: Z", entry_id="ap-1")

    out = _run_main(
        tmp_path,
        _stdin_json(),
        antipattern_results=[ap],
        storage=storage,
        registry=registry,
        repo_collection="repo.my-project",
    )

    parsed = json.loads(out)
    ctx = parsed["hookSpecificOutput"]["additionalContext"]
    assert "[ai-mem warnings]" in ctx
    assert "⚠" in ctx
    assert "Tried: X" in ctx


# ---------------------------------------------------------------------------
# 12. Anti-pattern warnings appear before main context block
# ---------------------------------------------------------------------------

def test_antipattern_block_precedes_context_block(tmp_path: Path) -> None:
    storage = _make_storage(labeled_count=15)
    registry = _make_registry("repo.my-project")
    ap = _make_result(0.85, "Tried: A\nFailed because: B\nInstead: C", entry_id="ap-1")
    regular = _make_result(0.9, "regular memory", entry_id="r-1")

    out = _run_main(
        tmp_path,
        _stdin_json(),
        repo_results=[regular],
        antipattern_results=[ap],
        storage=storage,
        registry=registry,
        repo_collection="repo.my-project",
    )

    parsed = json.loads(out)
    ctx = parsed["hookSpecificOutput"]["additionalContext"]
    assert ctx.index("[ai-mem warnings]") < ctx.index("[ai-mem]")


# ---------------------------------------------------------------------------
# 13. Anti-patterns below ANTIPATTERN_MIN_SCORE are filtered
# ---------------------------------------------------------------------------

def test_antipattern_below_min_score_not_injected(tmp_path: Path) -> None:
    storage = _make_storage(labeled_count=0)
    registry = _make_registry("repo.my-project")
    ap = _make_result(0.2, "Tried: low score\nFailed because: X\nInstead: Y", entry_id="ap-low")

    out = _run_main(
        tmp_path,
        _stdin_json(),
        antipattern_results=[ap],
        storage=storage,
        registry=registry,
        repo_collection="repo.my-project",
    )

    assert out == ""


# ---------------------------------------------------------------------------
# 14. Anti-patterns deduplicated against already-injected session IDs
# ---------------------------------------------------------------------------

def test_antipattern_dedup_against_session_injected(tmp_path: Path) -> None:
    import time

    storage = _make_storage(labeled_count=0)
    registry = _make_registry("repo.my-project")
    ap = _make_result(0.85, "Tried: seen before\nFailed because: X\nInstead: Y", entry_id="ap-seen")

    session_file = tmp_path / "session_injected.json"
    session_file.write_text(json.dumps({"session_ts": time.time(), "ids": ["ap-seen"]}))

    out = _run_main(
        tmp_path,
        _stdin_json(),
        antipattern_results=[ap],
        storage=storage,
        registry=registry,
        repo_collection="repo.my-project",
    )

    assert out == ""


# ---------------------------------------------------------------------------
# 10. Corrupt session file is ignored (does not crash the hook)
# ---------------------------------------------------------------------------

def test_corrupt_session_file_is_ignored(tmp_path: Path) -> None:
    results = [_make_result(0.9, "memory A", entry_id="id-1")]
    storage = _make_storage(labeled_count=12)
    registry = _make_registry("global")

    session_file = tmp_path / "session_injected.json"
    session_file.write_text("{corrupt json{{")

    out = _run_main(tmp_path, _stdin_json(), global_results=results, storage=storage, registry=registry)

    parsed = json.loads(out)
    ctx = parsed["hookSpecificOutput"]["additionalContext"]
    assert "memory A" in ctx


# ---------------------------------------------------------------------------
# 15. Dilemma warnings fire without ranker qualification
# ---------------------------------------------------------------------------

def test_dilemma_fires_without_ranker_qualification(tmp_path: Path) -> None:
    storage = _make_storage(labeled_count=0)
    registry = _make_registry("repo.my-project")
    d = _make_result(0.75, "Tension: A vs. B\nContext A: ...\nContext B: ...\nQuestions: ?", entry_id="d-1")

    out = _run_main(
        tmp_path,
        _stdin_json(),
        dilemma_results=[d],
        storage=storage,
        registry=registry,
        repo_collection="repo.my-project",
    )

    parsed = json.loads(out)
    ctx = parsed["hookSpecificOutput"]["additionalContext"]
    assert "[ai-mem dilemmas]" in ctx
    assert "⚖" in ctx
    assert "Tension: A vs. B" in ctx


# ---------------------------------------------------------------------------
# 16. Dilemma block precedes warnings and context blocks
# ---------------------------------------------------------------------------

def test_dilemma_block_precedes_warnings_and_context(tmp_path: Path) -> None:
    storage = _make_storage(labeled_count=15)
    registry = _make_registry("repo.my-project")
    d = _make_result(0.75, "Tension: X vs. Y\nQuestions: ?", entry_id="d-1")
    ap = _make_result(0.85, "Tried: A\nFailed because: B\nInstead: C", entry_id="ap-1")
    regular = _make_result(0.9, "regular memory", entry_id="r-1")

    out = _run_main(
        tmp_path,
        _stdin_json(),
        repo_results=[regular],
        antipattern_results=[ap],
        dilemma_results=[d],
        storage=storage,
        registry=registry,
        repo_collection="repo.my-project",
    )

    parsed = json.loads(out)
    ctx = parsed["hookSpecificOutput"]["additionalContext"]
    assert ctx.index("[ai-mem dilemmas]") < ctx.index("[ai-mem warnings]")
    assert ctx.index("[ai-mem warnings]") < ctx.index("[ai-mem]")


# ---------------------------------------------------------------------------
# 17. Dilemma below DILEMMA_MIN_SCORE is filtered
# ---------------------------------------------------------------------------

def test_dilemma_below_min_score_not_injected(tmp_path: Path) -> None:
    storage = _make_storage(labeled_count=0)
    registry = _make_registry("repo.my-project")
    d = _make_result(0.2, "Tension: low score", entry_id="d-low")

    out = _run_main(
        tmp_path,
        _stdin_json(),
        dilemma_results=[d],
        storage=storage,
        registry=registry,
        repo_collection="repo.my-project",
    )

    assert out == ""


# ---------------------------------------------------------------------------
# 18. Anti-pattern with Affected: field appends anticipation question
# ---------------------------------------------------------------------------

def test_antipattern_with_affected_appends_anticipation(tmp_path: Path) -> None:
    storage = _make_storage(labeled_count=0)
    registry = _make_registry("repo.my-project")
    ap = _make_result(
        0.85,
        "Tried: X\nFailed because: Y\nAffected: maintainer lost trust\nInstead: Z",
        entry_id="ap-affected",
    )

    out = _run_main(
        tmp_path,
        _stdin_json(),
        antipattern_results=[ap],
        storage=storage,
        registry=registry,
        repo_collection="repo.my-project",
    )

    parsed = json.loads(out)
    ctx = parsed["hookSpecificOutput"]["additionalContext"]
    assert "Anticipation" in ctx
    assert "Who holds the same role" in ctx


# ---------------------------------------------------------------------------
# 19. Anti-pattern without Affected: field does NOT append anticipation question
# ---------------------------------------------------------------------------

def test_antipattern_without_affected_no_anticipation(tmp_path: Path) -> None:
    storage = _make_storage(labeled_count=0)
    registry = _make_registry("repo.my-project")
    ap = _make_result(
        0.85,
        "Tried: X\nFailed because: Y\nInstead: Z",
        entry_id="ap-no-affected",
    )

    out = _run_main(
        tmp_path,
        _stdin_json(),
        antipattern_results=[ap],
        storage=storage,
        registry=registry,
        repo_collection="repo.my-project",
    )

    parsed = json.loads(out)
    ctx = parsed["hookSpecificOutput"]["additionalContext"]
    assert "Anticipation" not in ctx
