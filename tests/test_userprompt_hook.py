"""Tests for userprompt_hook.py — UserPromptSubmit hook."""
from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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


def _stdin_json(prompt: str = "what is the current focus?") -> StringIO:
    return StringIO(json.dumps({"hook_event_name": "UserPromptSubmit", "prompt": prompt}))


def _run_main(
    tmp_path: Path,
    stdin_stream: StringIO,
    global_results=None,
    repo_results=None,
    antipattern_results=None,
    dilemma_results=None,
    repo_collection: str | None = None,
) -> str:
    """Run hook.main() with mocked dependencies; return captured stdout."""
    global_results = global_results or []
    repo_results = repo_results or []
    antipattern_results = antipattern_results or []
    dilemma_results = dilemma_results or []

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
        patch.object(hook, "_build_deps", return_value=query_uc),
        patch.object(hook, "detect_repo_context", return_value=ctx),
        patch.object(hook, "GLOBAL_COLLECTION", "global"),
    ):
        tmp_path.mkdir(parents=True, exist_ok=True)
        captured = []
        with patch("builtins.print", side_effect=lambda s: captured.append(s)):
            hook.main()

    return "\n".join(captured)


# ---------------------------------------------------------------------------
# 0. Short/low-content prompts are skipped before any query runs
# ---------------------------------------------------------------------------

def test_short_prompt_skipped_without_querying(tmp_path: Path) -> None:
    results = [_make_result(0.99, "should never surface")]

    out = _run_main(tmp_path, _stdin_json("ja, ablegen"), global_results=results)

    assert out == ""


def test_prompt_at_min_chars_boundary_is_queried(tmp_path: Path) -> None:
    prompt = "x" * hook.MIN_QUERY_CHARS
    results = [_make_result(0.9, "memory at boundary")]

    out = _run_main(tmp_path, _stdin_json(prompt), global_results=results)

    parsed = json.loads(out)
    assert "memory at boundary" in parsed["hookSpecificOutput"]["additionalContext"]


def test_prompt_one_under_min_chars_is_skipped(tmp_path: Path) -> None:
    prompt = "x" * (hook.MIN_QUERY_CHARS - 1)
    results = [_make_result(0.9, "should not surface")]

    out = _run_main(tmp_path, _stdin_json(prompt), global_results=results)

    assert out == ""


def test_empty_prompt_skipped(tmp_path: Path) -> None:
    out = _run_main(tmp_path, _stdin_json(""), global_results=[_make_result(0.9)])

    assert out == ""


# ---------------------------------------------------------------------------
# 1. No output when all results are below CONTEXT_MIN_SCORE
# ---------------------------------------------------------------------------

def test_no_output_when_all_results_below_min_score(tmp_path: Path) -> None:
    results = [_make_result(0.2), _make_result(0.1), _make_result(0.05)]

    out = _run_main(tmp_path, _stdin_json(), global_results=results)

    assert out == ""


# ---------------------------------------------------------------------------
# 2. Injects context when score is above CONTEXT_MIN_SCORE
# ---------------------------------------------------------------------------

def test_injects_context_when_score_above_threshold(tmp_path: Path) -> None:
    results = [_make_result(0.9, "memory A"), _make_result(0.8, "memory B"), _make_result(0.75, "memory C")]

    out = _run_main(tmp_path, _stdin_json(), global_results=results)

    parsed = json.loads(out)
    ctx = parsed["hookSpecificOutput"]["additionalContext"]
    assert "ai-mem" in ctx
    assert "memory A" in ctx
    assert parsed["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"


# ---------------------------------------------------------------------------
# 3. Mixed scores — only entries above threshold are injected
# ---------------------------------------------------------------------------

def test_only_entries_above_threshold_injected(tmp_path: Path) -> None:
    results = [_make_result(0.9, "high score"), _make_result(0.1, "low score")]

    out = _run_main(tmp_path, _stdin_json(), global_results=results)

    parsed = json.loads(out)
    ctx = parsed["hookSpecificOutput"]["additionalContext"]
    assert "high score" in ctx
    assert "low score" not in ctx


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

    out = _run_main(
        tmp_path,
        _stdin_json(),
        global_results=global_results,
        repo_results=repo_results,
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

    # Pre-populate the session file with id-1 already seen.
    session_file = tmp_path / "session_injected.json"
    session_file.write_text(json.dumps({"session_ts": time.time(), "ids": ["id-1"]}))

    out = _run_main(tmp_path, _stdin_json(), global_results=results)

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

    out = _run_main(tmp_path, _stdin_json(), global_results=results)

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

    # Write a session file that is 5 hours old (beyond SESSION_TTL_HOURS=4).
    old_ts = time.time() - 5 * 3600
    session_file = tmp_path / "session_injected.json"
    session_file.write_text(json.dumps({"session_ts": old_ts, "ids": ["id-1", "id-2"]}))

    out = _run_main(tmp_path, _stdin_json(), global_results=results)

    parsed = json.loads(out)
    ctx = parsed["hookSpecificOutput"]["additionalContext"]
    # Both entries must appear because the old session file is expired.
    assert "memory A" in ctx
    assert "memory B" in ctx


# ---------------------------------------------------------------------------
# 11. Anti-pattern warnings fire unconditionally (no ranker needed)
# ---------------------------------------------------------------------------

def test_antipattern_fires_without_ranker_qualification(tmp_path: Path) -> None:
    ap = _make_result(0.85, "Tried: X\nFailed because: Y\nInstead: Z", entry_id="ap-1")

    out = _run_main(
        tmp_path,
        _stdin_json(),
        antipattern_results=[ap],
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
    ap = _make_result(0.85, "Tried: A\nFailed because: B\nInstead: C", entry_id="ap-1")
    regular = _make_result(0.9, "regular memory", entry_id="r-1")

    out = _run_main(
        tmp_path,
        _stdin_json(),
        repo_results=[regular],
        antipattern_results=[ap],
        repo_collection="repo.my-project",
    )

    parsed = json.loads(out)
    ctx = parsed["hookSpecificOutput"]["additionalContext"]
    assert ctx.index("[ai-mem warnings]") < ctx.index("[ai-mem]")


# ---------------------------------------------------------------------------
# 13. Anti-patterns below ANTIPATTERN_MIN_SCORE are filtered
# ---------------------------------------------------------------------------

def test_antipattern_below_min_score_not_injected(tmp_path: Path) -> None:
    ap = _make_result(0.2, "Tried: low score\nFailed because: X\nInstead: Y", entry_id="ap-low")

    out = _run_main(
        tmp_path,
        _stdin_json(),
        antipattern_results=[ap],
        repo_collection="repo.my-project",
    )

    assert out == ""


# ---------------------------------------------------------------------------
# 14. Anti-patterns deduplicated against already-injected session IDs
# ---------------------------------------------------------------------------

def test_antipattern_dedup_against_session_injected(tmp_path: Path) -> None:
    import time

    ap = _make_result(0.85, "Tried: seen before\nFailed because: X\nInstead: Y", entry_id="ap-seen")

    session_file = tmp_path / "session_injected.json"
    session_file.write_text(json.dumps({"session_ts": time.time(), "ids": ["ap-seen"]}))

    out = _run_main(
        tmp_path,
        _stdin_json(),
        antipattern_results=[ap],
        repo_collection="repo.my-project",
    )

    assert out == ""


# ---------------------------------------------------------------------------
# 10. Corrupt session file is ignored (does not crash the hook)
# ---------------------------------------------------------------------------

def test_corrupt_session_file_is_ignored(tmp_path: Path) -> None:
    results = [_make_result(0.9, "memory A", entry_id="id-1")]

    session_file = tmp_path / "session_injected.json"
    session_file.write_text("{corrupt json{{")

    out = _run_main(tmp_path, _stdin_json(), global_results=results)

    parsed = json.loads(out)
    ctx = parsed["hookSpecificOutput"]["additionalContext"]
    assert "memory A" in ctx


# ---------------------------------------------------------------------------
# 15. Dilemma warnings fire unconditionally (no ranker needed)
# ---------------------------------------------------------------------------

def test_dilemma_fires_without_ranker_qualification(tmp_path: Path) -> None:
    d = _make_result(0.75, "Tension: A vs. B\nContext A: ...\nContext B: ...\nQuestions: ?", entry_id="d-1")

    out = _run_main(
        tmp_path,
        _stdin_json(),
        dilemma_results=[d],
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
    d = _make_result(0.75, "Tension: X vs. Y\nQuestions: ?", entry_id="d-1")
    ap = _make_result(0.85, "Tried: A\nFailed because: B\nInstead: C", entry_id="ap-1")
    regular = _make_result(0.9, "regular memory", entry_id="r-1")

    out = _run_main(
        tmp_path,
        _stdin_json(),
        repo_results=[regular],
        antipattern_results=[ap],
        dilemma_results=[d],
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
    d = _make_result(0.2, "Tension: low score", entry_id="d-low")

    out = _run_main(
        tmp_path,
        _stdin_json(),
        dilemma_results=[d],
        repo_collection="repo.my-project",
    )

    assert out == ""


# ---------------------------------------------------------------------------
# 18. Anti-pattern with Affected: field appends anticipation question
# ---------------------------------------------------------------------------

def test_antipattern_with_affected_appends_anticipation(tmp_path: Path) -> None:
    ap = _make_result(
        0.85,
        "Tried: X\nFailed because: Y\nAffected: maintainer lost trust\nInstead: Z",
        entry_id="ap-affected",
    )

    out = _run_main(
        tmp_path,
        _stdin_json(),
        antipattern_results=[ap],
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
    ap = _make_result(
        0.85,
        "Tried: X\nFailed because: Y\nInstead: Z",
        entry_id="ap-no-affected",
    )

    out = _run_main(
        tmp_path,
        _stdin_json(),
        antipattern_results=[ap],
        repo_collection="repo.my-project",
    )

    parsed = json.loads(out)
    ctx = parsed["hookSpecificOutput"]["additionalContext"]
    assert "Anticipation" not in ctx


# ---------------------------------------------------------------------------
# 20. Repo-scoped lookup gates on has_claude_md, not on collection membership
# ---------------------------------------------------------------------------

def test_repo_lookup_fires_when_workspace_collection_has_claude_md(tmp_path: Path) -> None:
    """CLAUDE.md at the workspace root itself yields collection == WORKSPACE_COLLECTION
    with has_claude_md == True — the repo-scoped query must still fire in that case."""
    global_results = [_make_result(0.9, "global memory")]
    repo_results = [_make_result(0.85, "repo memory")]

    query_uc = MagicMock()

    def fake_hits(collection, query, n_results, type_filter=None, max_age_days=None):
        if type_filter is not None:
            return []
        if collection == "global":
            return global_results
        return repo_results

    query_uc.execute.side_effect = fake_hits

    ctx = MagicMock()
    ctx.collection = "workspace"
    ctx.has_claude_md = True

    with (
        patch.object(sys, "stdin", _stdin_json()),
        patch.object(hook, "DB_PATH", tmp_path),
        patch.object(hook, "_build_deps", return_value=query_uc),
        patch.object(hook, "detect_repo_context", return_value=ctx),
        patch.object(hook, "GLOBAL_COLLECTION", "global"),
    ):
        tmp_path.mkdir(parents=True, exist_ok=True)
        captured = []
        with patch("builtins.print", side_effect=lambda s: captured.append(s)):
            hook.main()

    out = "\n".join(captured)
    parsed = json.loads(out)
    ctx_text = parsed["hookSpecificOutput"]["additionalContext"]
    assert "repo memory" in ctx_text


# ---------------------------------------------------------------------------
# 21. Regression: off-topic hybrid scores in the 0.45-0.60 band (measured
# irrelevant-query ceiling against the live DB, pre-fix threshold was 0.3)
# must not inject. This is the exact failure mode the CONTEXT_MIN_SCORE
# recalibration exists to close.
# ---------------------------------------------------------------------------

def test_irrelevant_hybrid_scores_do_not_inject(tmp_path: Path) -> None:
    """Scores in this band were observed for genuinely off-topic prompts
    (e.g. "what is the weather in Paris today?") against the real DB after
    the Chunk 1/2 scoring fixes — the old CONTEXT_MIN_SCORE=0.3 would have
    let all of these through."""
    noise_scores = [0.4795, 0.5659, 0.6023]
    results = [_make_result(score, f"unrelated memory {score}") for score in noise_scores]

    out = _run_main(tmp_path, _stdin_json(), global_results=results, repo_results=results)

    assert out == ""


def test_relevant_hybrid_score_still_injects(tmp_path: Path) -> None:
    """A genuinely relevant hit (measured ~0.71-0.78 floor in repo.ai-mem
    against real matching entries) must still clear the new threshold."""
    results = [_make_result(0.7653, "ChromaDB cosine similarity scoring bug fix")]

    out = _run_main(tmp_path, _stdin_json(), global_results=results)

    parsed = json.loads(out)
    ctx = parsed["hookSpecificOutput"]["additionalContext"]
    assert "ChromaDB cosine similarity scoring bug fix" in ctx


def test_repo_lookup_skipped_when_not_has_claude_md(tmp_path: Path) -> None:
    global_results = [_make_result(0.9, "global memory")]
    repo_results = [_make_result(0.85, "repo memory")]

    query_uc = MagicMock()

    def fake_hits(collection, query, n_results, type_filter=None, max_age_days=None):
        if type_filter is not None:
            return []
        if collection == "global":
            return global_results
        return repo_results

    query_uc.execute.side_effect = fake_hits

    ctx = MagicMock()
    ctx.collection = "workspace"
    ctx.has_claude_md = False

    with (
        patch.object(sys, "stdin", _stdin_json()),
        patch.object(hook, "DB_PATH", tmp_path),
        patch.object(hook, "_build_deps", return_value=query_uc),
        patch.object(hook, "detect_repo_context", return_value=ctx),
        patch.object(hook, "GLOBAL_COLLECTION", "global"),
    ):
        tmp_path.mkdir(parents=True, exist_ok=True)
        captured = []
        with patch("builtins.print", side_effect=lambda s: captured.append(s)):
            hook.main()

    out = "\n".join(captured)
    parsed = json.loads(out)
    ctx_text = parsed["hookSpecificOutput"]["additionalContext"]
    assert "repo memory" not in ctx_text
