"""Tests for hook.py — SessionStart hook expert-agent collection injection."""
from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai_mem import hook


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent_ctx(agent_type: str | None, should_inject: bool = True) -> MagicMock:
    ctx = MagicMock()
    ctx.agent_type = agent_type
    ctx.should_inject = should_inject
    return ctx


def _make_repo_ctx(
    collection: str = "workspace",
    has_claude_md: bool = False,
    scope_name: str = "test",
) -> MagicMock:
    ctx = MagicMock()
    ctx.collection = collection
    ctx.has_claude_md = has_claude_md
    ctx.scope_name = scope_name
    ctx.claude_md_dir = None
    return ctx


def _make_entry(id: str, text: str) -> MagicMock:
    e = MagicMock()
    e.id = id
    e.text = text
    return e


def _make_mem_entry(id: str, text: str, confidence: float = 1.0, access_count: int = 0) -> MagicMock:
    e = MagicMock()
    e.id = id
    e.text = text
    e.metadata = {"confidence": str(confidence), "access_count": str(access_count)}
    return e


def _run_main(
    tmp_path: Path,
    agent_type: str | None = None,
    should_inject: bool = True,
    focus_map: dict[str, str | None] | None = None,
    repo_collection: str = "workspace",
    session_delta: int | None = None,
    git_commits: list[str] | None = None,
    expert_query_results: list | None = None,
    high_confidence_entries: list | None = None,
) -> str:
    """Run hook.main() with mocked dependencies; return captured stdout.

    All imports in hook.py are now module-level, so patch.object(hook, ...) works.
    _focus_text, _session_delta, and _write_prev_session are patched directly
    to control output without standing up ChromaDB.

    expert_query_results: list of mock entries returned by repo.query() for the expert
    collection. Use _make_entry(id, text) to build them. Default: empty list (MagicMock
    default iteration yields nothing).
    """
    focus_map = focus_map or {}

    def fake_focus_text(get_memory_uc, collection: str) -> str | None:
        return focus_map.get(collection)

    mock_list_uc = MagicMock()
    mock_list_uc.return_value.execute.return_value = []

    mock_chroma_cls = MagicMock()
    mock_repo_instance = mock_chroma_cls.return_value
    mock_repo_instance.query.return_value = expert_query_results or []

    agent_ctx = _make_agent_ctx(agent_type, should_inject)
    repo_ctx = _make_repo_ctx(collection=repo_collection)

    stdin_stream = StringIO(json.dumps({"transcript_path": None}))

    with (
        patch.object(sys, "stdin", stdin_stream),
        patch.object(hook, "DB_PATH", tmp_path),
        patch.object(hook, "_focus_text", side_effect=fake_focus_text),
        patch.object(hook, "_session_delta", return_value=session_delta),
        patch.object(hook, "_git_commits_since", return_value=git_commits or []),
        patch.object(hook, "_write_prev_session"),
        patch.object(hook, "detect_for_session_start", return_value=agent_ctx),
        patch.object(hook, "write_to_env_file"),
        patch.object(hook, "ChromaMemoryRepository", mock_chroma_cls),
        patch.object(hook, "GetMemoryUseCase"),
        patch.object(hook, "detect_repo_context", return_value=repo_ctx),
        patch.object(hook, "record_injection"),
        patch.object(hook, "GLOBAL_COLLECTION", "global"),
        patch.object(hook, "WORKSPACE_COLLECTION", "workspace"),
        patch.object(hook, "ListCollectionsUseCase", mock_list_uc),
        patch.object(hook, "_high_confidence_entries", return_value=high_confidence_entries or []),
    ):
        tmp_path.mkdir(parents=True, exist_ok=True)
        captured: list[str] = []
        with patch("builtins.print", side_effect=lambda s: captured.append(s)):
            hook.main()

    return "\n".join(captured)


# ---------------------------------------------------------------------------
# 1. No agent_type → no expert block, no routing hint
# ---------------------------------------------------------------------------

def test_no_agent_type_produces_no_expert_output(tmp_path: Path) -> None:
    out = _run_main(tmp_path, agent_type=None, focus_map={"global": "global memory content"})

    assert out  # something is printed (global focus)
    assert "expertise" not in out
    assert "Expert collection" not in out
    assert "subagent" not in out


# ---------------------------------------------------------------------------
# 2. agent_type set, expert collection empty → routing hint still appears
# ---------------------------------------------------------------------------

def test_routing_hint_appears_even_when_expert_collection_empty(tmp_path: Path) -> None:
    out = _run_main(
        tmp_path,
        agent_type="the-coder",
        focus_map={"global": "global mem"},  # expert collection absent → None
    )

    assert out
    parsed = json.loads(out)
    ctx = parsed["hookSpecificOutput"]["additionalContext"]
    assert 'Expert collection: "subagent.the-coder"' in ctx
    assert 'collection="subagent.the-coder"' in ctx
    # expertise block absent since collection is empty
    assert "[the-coder expertise]" not in ctx


# ---------------------------------------------------------------------------
# 3. agent_type set, expert focus present → both expertise block and hint appear
# ---------------------------------------------------------------------------

def test_expert_focus_and_routing_hint_both_appear(tmp_path: Path) -> None:
    expert_text = "Prefer integration tests over mocks in Python projects."
    out = _run_main(
        tmp_path,
        agent_type="the-coder",
        focus_map={
            "global": "global mem",
            "subagent.the-coder": expert_text,
        },
    )

    parsed = json.loads(out)
    ctx = parsed["hookSpecificOutput"]["additionalContext"]
    assert "[the-coder expertise]" in ctx
    assert expert_text in ctx
    assert 'Expert collection: "subagent.the-coder"' in ctx


# ---------------------------------------------------------------------------
# 4. expert_focus longer than _FOCUS_PREVIEW_CHARS is truncated
# ---------------------------------------------------------------------------

def test_expert_focus_is_truncated_to_preview_limit(tmp_path: Path) -> None:
    long_text = "A" * 300  # well beyond _FOCUS_PREVIEW_CHARS = 150
    out = _run_main(
        tmp_path,
        agent_type="general-purpose",
        focus_map={"subagent.general-purpose": long_text},
    )

    parsed = json.loads(out)
    ctx = parsed["hookSpecificOutput"]["additionalContext"]
    # The truncated text ends with the ellipsis character, not the full 300 chars
    assert "AAAA…" in ctx
    assert long_text not in ctx


# ---------------------------------------------------------------------------
# 4b. expert_query_results injected under "Relevant past learnings"
# ---------------------------------------------------------------------------

def test_expert_query_results_appear_in_expertise_block(tmp_path: Path) -> None:
    entry = _make_entry("learned_ddd", "Always push domain logic down — never let infra leak upward.")
    out = _run_main(
        tmp_path,
        agent_type="the-coder",
        focus_map={"global": "global mem"},
        expert_query_results=[entry],
    )

    parsed = json.loads(out)
    ctx = parsed["hookSpecificOutput"]["additionalContext"]
    assert "[the-coder expertise]" in ctx
    assert "Relevant past learnings" in ctx
    assert "Always push domain logic down" in ctx


def test_current_focus_excluded_from_expert_query_results(tmp_path: Path) -> None:
    focus_entry = _make_entry("current_focus", "This is the focus entry.")
    other_entry = _make_entry("learned_xyz", "Use dependency injection consistently.")
    out = _run_main(
        tmp_path,
        agent_type="the-coder",
        focus_map={"global": "global mem"},
        expert_query_results=[focus_entry, other_entry],
    )

    parsed = json.loads(out)
    ctx = parsed["hookSpecificOutput"]["additionalContext"]
    assert "This is the focus entry." not in ctx  # focus entry filtered out
    assert "Use dependency injection consistently." in ctx


# ---------------------------------------------------------------------------
# 5. should_inject=False → hook exits early, no output at all
# ---------------------------------------------------------------------------

def test_should_inject_false_produces_no_output(tmp_path: Path) -> None:
    out = _run_main(
        tmp_path,
        agent_type="the-coder",
        should_inject=False,
        focus_map={
            "global": "global mem",
            "subagent.the-coder": "expert mem",
        },
    )

    assert out == ""


# ---------------------------------------------------------------------------
# 9. Cross-session delta — positive delta shows the line
# ---------------------------------------------------------------------------

def test_session_delta_positive_shows_new_entries_line(tmp_path: Path) -> None:
    out = _run_main(
        tmp_path,
        focus_map={"global": "some context"},
        session_delta=5,
    )

    parsed = json.loads(out)
    ctx = parsed["hookSpecificOutput"]["additionalContext"]
    assert "Since last session: 5 new entries added" in ctx


def test_session_delta_singular_uses_entry_not_entries(tmp_path: Path) -> None:
    out = _run_main(
        tmp_path,
        focus_map={"global": "some context"},
        session_delta=1,
    )

    parsed = json.loads(out)
    ctx = parsed["hookSpecificOutput"]["additionalContext"]
    assert "Since last session: 1 new entry added" in ctx


# ---------------------------------------------------------------------------
# 10. Cross-session delta — None (no change or no prior) suppresses the line
# ---------------------------------------------------------------------------

def test_session_delta_none_suppresses_line(tmp_path: Path) -> None:
    out = _run_main(
        tmp_path,
        focus_map={"global": "some context"},
        session_delta=None,
    )

    parsed = json.loads(out)
    ctx = parsed["hookSpecificOutput"]["additionalContext"]
    assert "Since last session" not in ctx


# ---------------------------------------------------------------------------
# 11. Git commits since last session — commits present
# ---------------------------------------------------------------------------

def test_git_commits_shown_when_present(tmp_path: Path) -> None:
    commits = ["abc1234 feat: add dilemma warnings", "def5678 fix: early exit guard"]
    out = _run_main(
        tmp_path,
        focus_map={"global": "some context"},
        git_commits=commits,
    )

    parsed = json.loads(out)
    ctx = parsed["hookSpecificOutput"]["additionalContext"]
    assert "Git commits since last session" in ctx
    assert "abc1234 feat: add dilemma warnings" in ctx
    assert "def5678 fix: early exit guard" in ctx


# ---------------------------------------------------------------------------
# 12. Git commits since last session — no commits → no block
# ---------------------------------------------------------------------------

def test_no_git_commits_suppresses_block(tmp_path: Path) -> None:
    out = _run_main(
        tmp_path,
        focus_map={"global": "some context"},
        git_commits=[],
    )

    parsed = json.loads(out)
    ctx = parsed["hookSpecificOutput"]["additionalContext"]
    assert "Git commits since last session" not in ctx


# ---------------------------------------------------------------------------
# 13. High-confidence proactive injection ([always-present] block)
# ---------------------------------------------------------------------------

def test_high_confidence_entries_appear_in_always_present_block(tmp_path: Path) -> None:
    entry = _make_mem_entry("pattern_ddd", "Always push domain logic down, never let infra leak upward.")
    out = _run_main(
        tmp_path,
        focus_map={"global": "global mem"},
        repo_collection="repo.ai-mem",
        high_confidence_entries=[entry],
    )

    parsed = json.loads(out)
    ctx = parsed["hookSpecificOutput"]["additionalContext"]
    assert "[always-present]" in ctx
    assert "Always push domain logic down" in ctx


def test_no_always_present_block_when_high_confidence_empty(tmp_path: Path) -> None:
    out = _run_main(
        tmp_path,
        focus_map={"global": "global mem"},
        repo_collection="repo.ai-mem",
        high_confidence_entries=[],
    )

    parsed = json.loads(out)
    ctx = parsed["hookSpecificOutput"]["additionalContext"]
    assert "[always-present]" not in ctx


def test_always_present_block_skipped_for_workspace_collection(tmp_path: Path) -> None:
    entry = _make_mem_entry("pattern_x", "This should not appear.")
    out = _run_main(
        tmp_path,
        focus_map={"global": "global mem"},
        repo_collection="workspace",
        high_confidence_entries=[entry],
    )

    parsed = json.loads(out)
    ctx = parsed["hookSpecificOutput"]["additionalContext"]
    assert "[always-present]" not in ctx


def test_high_confidence_gate_excludes_low_confidence_entry() -> None:
    """confidence=0.7 + access_count=3 must NOT pass the gate (threshold is > 0.9)."""
    mock_repo = MagicMock()
    entry = _make_mem_entry("candidate", "Should not appear.", confidence=0.7, access_count=3)
    mock_repo.get_all.return_value = [entry]

    result = hook._high_confidence_entries(mock_repo, "repo.ai-mem", exclude_ids=set())

    assert result == []


def test_high_confidence_gate_passes_for_qualifying_entry() -> None:
    """confidence=0.95 + access_count=3 must pass the gate."""
    mock_repo = MagicMock()
    entry = _make_mem_entry("pattern_x", "Qualifying entry.", confidence=0.95, access_count=3)
    mock_repo.get_all.return_value = [entry]

    result = hook._high_confidence_entries(mock_repo, "repo.ai-mem", exclude_ids=set())

    assert len(result) == 1
    assert result[0].id == "pattern_x"


def test_high_confidence_gate_excludes_insufficient_access_count() -> None:
    """confidence=0.95 + access_count=2 must NOT pass (minimum is 3)."""
    mock_repo = MagicMock()
    entry = _make_mem_entry("pattern_y", "High confidence but few accesses.", confidence=0.95, access_count=2)
    mock_repo.get_all.return_value = [entry]

    result = hook._high_confidence_entries(mock_repo, "repo.ai-mem", exclude_ids=set())

    assert result == []


def test_always_present_text_truncated_to_high_confidence_chars(tmp_path: Path) -> None:
    long_text = "B" * 500  # well beyond _HIGH_CONFIDENCE_CHARS = 300
    entry = _make_mem_entry("pattern_long", long_text)
    out = _run_main(
        tmp_path,
        focus_map={"global": "global mem"},
        repo_collection="repo.ai-mem",
        high_confidence_entries=[entry],
    )

    parsed = json.loads(out)
    ctx = parsed["hookSpecificOutput"]["additionalContext"]
    assert "[always-present]" in ctx
    assert long_text not in ctx
    assert "BBBB…" in ctx
