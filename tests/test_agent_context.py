"""Tests for agent_context.py — subagent detection and hook routing."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from ai_mem import agent_context as ctx_mod
from ai_mem.agent_context import (
    AgentContext,
    detect_for_hook,
    detect_for_session_start,
    write_to_env_file,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _transcript(content: str, tmp_path: Path) -> str:
    p = tmp_path / "transcript.jsonl"
    p.write_text(content)
    return str(p)


def _clean_env(monkeypatch):
    monkeypatch.delenv("AI_MEM_HOOK_DEPTH", raising=False)
    monkeypatch.delenv("AI_MEM_AGENT_TYPE", raising=False)


# ---------------------------------------------------------------------------
# detect_for_session_start
# ---------------------------------------------------------------------------

class TestDetectForSessionStart:
    def test_main_session_no_transcript(self, monkeypatch):
        _clean_env(monkeypatch)
        result = detect_for_session_start({})
        assert result.depth == 0
        assert result.is_subagent is False
        assert result.should_inject is True

    def test_main_session_neutral_transcript(self, monkeypatch, tmp_path):
        _clean_env(monkeypatch)
        t = _transcript('{"role": "user", "content": "hello"}', tmp_path)
        result = detect_for_session_start({"transcript_path": t})
        assert result.depth == 0
        assert result.is_subagent is False
        assert result.should_inject is True

    def test_d1_the_coder(self, monkeypatch, tmp_path):
        _clean_env(monkeypatch)
        monkeypatch.setenv("AI_MEM_HOOK_DEPTH", "0")  # inherited from d=0
        t = _transcript(
            "chunk-first delivery capability-centric DDD always runnable", tmp_path
        )
        result = detect_for_session_start({"transcript_path": t})
        assert result.depth == 1
        assert result.is_subagent is True
        assert result.agent_type == "the-coder"
        assert result.should_inject is True

    def test_d1_explore(self, monkeypatch, tmp_path):
        _clean_env(monkeypatch)
        monkeypatch.setenv("AI_MEM_HOOK_DEPTH", "0")
        t = _transcript("Fast read-only search agent for locating code", tmp_path)
        result = detect_for_session_start({"transcript_path": t})
        assert result.depth == 1
        assert result.agent_type == "Explore"
        assert result.should_inject is False

    def test_d1_unknown_subagent_skips(self, monkeypatch, tmp_path):
        _clean_env(monkeypatch)
        monkeypatch.setenv("AI_MEM_HOOK_DEPTH", "0")
        t = _transcript("This is a custom ad-hoc agent with no known markers", tmp_path)
        result = detect_for_session_start({"transcript_path": t})
        assert result.depth == 1
        assert result.agent_type is None
        assert result.should_inject is False  # unknown subagent → skip

    def test_d2_inherits_increments(self, monkeypatch, tmp_path):
        _clean_env(monkeypatch)
        monkeypatch.setenv("AI_MEM_HOOK_DEPTH", "1")  # inherited from d=1
        t = _transcript("chunk-first delivery capability-centric DDD", tmp_path)
        result = detect_for_session_start({"transcript_path": t})
        assert result.depth == 2
        assert result.agent_type == "the-coder"
        assert result.should_inject is True

    def test_agent_type_not_inherited_from_env(self, monkeypatch, tmp_path):
        """SessionStart must ignore AI_MEM_AGENT_TYPE from parent env."""
        _clean_env(monkeypatch)
        monkeypatch.setenv("AI_MEM_HOOK_DEPTH", "0")
        monkeypatch.setenv("AI_MEM_AGENT_TYPE", "the-coder")  # parent's type
        # Transcript shows Explore — should override parent type
        t = _transcript("Fast read-only search agent", tmp_path)
        result = detect_for_session_start({"transcript_path": t})
        assert result.agent_type == "Explore"
        assert result.should_inject is False

    def test_missing_transcript_file(self, monkeypatch):
        _clean_env(monkeypatch)
        monkeypatch.setenv("AI_MEM_HOOK_DEPTH", "0")
        result = detect_for_session_start({"transcript_path": "/nonexistent/path.jsonl"})
        assert result.depth == 1
        assert result.agent_type is None
        assert result.should_inject is False

    def test_stdin_json_none(self, monkeypatch):
        _clean_env(monkeypatch)
        result = detect_for_session_start(None)
        assert result.depth == 0
        assert result.should_inject is True


# ---------------------------------------------------------------------------
# detect_for_hook (fast path)
# ---------------------------------------------------------------------------

class TestDetectForHook:
    def test_fast_path_main_session(self, monkeypatch):
        _clean_env(monkeypatch)
        monkeypatch.setenv("AI_MEM_HOOK_DEPTH", "0")
        result = detect_for_hook({})
        assert result.depth == 0
        assert result.should_inject is True

    def test_fast_path_inject_agent(self, monkeypatch):
        _clean_env(monkeypatch)
        monkeypatch.setenv("AI_MEM_HOOK_DEPTH", "1")
        monkeypatch.setenv("AI_MEM_AGENT_TYPE", "the-coder")
        result = detect_for_hook({})
        assert result.is_subagent is True
        assert result.agent_type == "the-coder"
        assert result.should_inject is True

    def test_fast_path_skip_agent(self, monkeypatch):
        _clean_env(monkeypatch)
        monkeypatch.setenv("AI_MEM_HOOK_DEPTH", "1")
        monkeypatch.setenv("AI_MEM_AGENT_TYPE", "Explore")
        result = detect_for_hook({})
        assert result.should_inject is False

    def test_fast_path_unknown_subagent_skips(self, monkeypatch):
        _clean_env(monkeypatch)
        monkeypatch.setenv("AI_MEM_HOOK_DEPTH", "1")
        # No AI_MEM_AGENT_TYPE → unknown
        result = detect_for_hook({})
        assert result.is_subagent is True
        assert result.agent_type is None
        assert result.should_inject is False

    def test_fallback_to_slow_path_when_no_env(self, monkeypatch, tmp_path):
        _clean_env(monkeypatch)
        t = _transcript("Fast read-only search agent", tmp_path)
        result = detect_for_hook({"transcript_path": t})
        # Should behave like detect_for_session_start with depth=0 (main)
        # → no inherited depth, but transcript matches Explore... but depth=0 means main
        # Actually: no AI_MEM_HOOK_DEPTH → falls back → inherited_depth=-1 → depth=0
        # → is_subagent=False → should_inject=True (main session)
        assert result.depth == 0
        assert result.should_inject is True

    def test_fast_path_skips_transcript_io(self, monkeypatch):
        """Fast path must not read transcript_path at all."""
        _clean_env(monkeypatch)
        monkeypatch.setenv("AI_MEM_HOOK_DEPTH", "1")
        monkeypatch.setenv("AI_MEM_AGENT_TYPE", "the-coder")
        # Pass a non-existent transcript — should not raise
        result = detect_for_hook({"transcript_path": "/nonexistent/transcript.jsonl"})
        assert result.should_inject is True


# ---------------------------------------------------------------------------
# write_to_env_file
# ---------------------------------------------------------------------------

class TestWriteToEnvFile:
    def test_writes_depth_and_type(self, monkeypatch, tmp_path):
        env_file = tmp_path / "claude_env"
        monkeypatch.setenv("CLAUDE_ENV_FILE", str(env_file))
        c = AgentContext(is_subagent=True, depth=1, agent_type="the-coder", should_inject=True)
        write_to_env_file(c)
        lines = env_file.read_text().splitlines()
        assert "AI_MEM_HOOK_DEPTH=1" in lines
        assert "AI_MEM_AGENT_TYPE=the-coder" in lines

    def test_writes_empty_type_for_none(self, monkeypatch, tmp_path):
        env_file = tmp_path / "claude_env"
        monkeypatch.setenv("CLAUDE_ENV_FILE", str(env_file))
        c = AgentContext(is_subagent=False, depth=0, agent_type=None, should_inject=True)
        write_to_env_file(c)
        lines = env_file.read_text().splitlines()
        assert "AI_MEM_HOOK_DEPTH=0" in lines
        assert "AI_MEM_AGENT_TYPE=" in lines

    def test_no_env_file_set(self, monkeypatch):
        monkeypatch.delenv("CLAUDE_ENV_FILE", raising=False)
        c = AgentContext(is_subagent=True, depth=1, agent_type="the-coder", should_inject=True)
        write_to_env_file(c)  # must not raise

    def test_appends_not_overwrites(self, monkeypatch, tmp_path):
        env_file = tmp_path / "claude_env"
        env_file.write_text("EXISTING_VAR=foo\n")
        monkeypatch.setenv("CLAUDE_ENV_FILE", str(env_file))
        c = AgentContext(is_subagent=True, depth=1, agent_type="the-coder", should_inject=True)
        write_to_env_file(c)
        content = env_file.read_text()
        assert "EXISTING_VAR=foo" in content
        assert "AI_MEM_HOOK_DEPTH=1" in content
