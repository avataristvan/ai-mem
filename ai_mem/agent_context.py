"""Agent context detection for hook routing.

Determines whether the current Claude Code session is a top-level user session
(d=0) or a spawned subagent (d≥1), and which agent type it is, so hooks can
decide whether to inject ai-mem context.

Two entry points:
- detect_for_session_start() — called from SessionStart; parses transcript.
- detect_for_hook()          — called from other hooks; uses env set by SessionStart.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

INJECT_CONTEXT_AGENTS: frozenset[str] = frozenset({"the-coder", "general-purpose"})

_SIGNATURES: list[tuple[str, list[str]]] = [
    ("the-coder",             ["chunk-first delivery", "capability-centric DDD"]),
    ("general-purpose",       ["General-purpose agent for researching"]),
    ("Explore",               ["Fast read-only search agent"]),
    ("Plan",                  ["Software architect agent for designing"]),
    ("code-reviewer",         ["adherence to project guidelines", "style violations"]),
    ("silent-failure-hunter", ["silent failures", "inadequate error handling"]),
    ("type-design-analyzer",  ["type design", "invariant expression"]),
    ("pr-test-analyzer",      ["test coverage quality"]),
    ("code-simplifier",       ["simplifies and refines code for clarity"]),
    ("comment-analyzer",      ["analyze code comments for accuracy"]),
    ("agent-creator",         ["create an agent", "agent configuration"]),
    ("skill-reviewer",        ["skill quality"]),
    ("plugin-validator",      ["plugin structure", "plugin.json"]),
]

_ENV_DEPTH = "AI_MEM_HOOK_DEPTH"
_ENV_TYPE = "AI_MEM_AGENT_TYPE"


@dataclass(frozen=True)
class AgentContext:
    is_subagent: bool
    depth: int
    agent_type: str | None
    should_inject: bool


def detect_for_session_start(stdin_json: dict | None) -> AgentContext:
    """Detect agent context for use in SessionStart.

    Always parses the transcript — never trusts the inherited AI_MEM_AGENT_TYPE
    (that belongs to the parent session). Uses the inherited AI_MEM_HOOK_DEPTH
    to compute the current session's depth.
    """
    inherited_depth = int(os.environ.get(_ENV_DEPTH, "-1"))
    current_depth = max(inherited_depth + 1, 0)

    transcript_path = (stdin_json or {}).get("transcript_path")
    agent_type = _parse_agent_type(transcript_path) if transcript_path else None

    is_subagent = current_depth > 0
    should_inject = (not is_subagent) or (agent_type in INJECT_CONTEXT_AGENTS)

    return AgentContext(
        is_subagent=is_subagent,
        depth=current_depth,
        agent_type=agent_type,
        should_inject=should_inject,
    )


def detect_for_hook(stdin_json: dict | None) -> AgentContext:
    """Detect agent context for UserPromptSubmit / PreToolUse.

    Fast path: reads env vars written by this session's SessionStart.
    Falls back to detect_for_session_start() if env is absent.
    """
    raw = os.environ.get(_ENV_DEPTH, "")
    if not raw:
        return detect_for_session_start(stdin_json)

    try:
        depth = int(raw)
    except ValueError:
        return detect_for_session_start(stdin_json)

    agent_type = os.environ.get(_ENV_TYPE) or None
    is_subagent = depth > 0
    should_inject = (not is_subagent) or (agent_type in INJECT_CONTEXT_AGENTS)

    return AgentContext(
        is_subagent=is_subagent,
        depth=depth,
        agent_type=agent_type,
        should_inject=should_inject,
    )


def write_to_env_file(ctx: AgentContext) -> None:
    """Append depth and agent_type to $CLAUDE_ENV_FILE if available."""
    env_file = os.environ.get("CLAUDE_ENV_FILE", "")
    if not env_file:
        return
    try:
        with open(env_file, "a") as f:
            f.write(f"{_ENV_DEPTH}={ctx.depth}\n")
            f.write(f"{_ENV_TYPE}={ctx.agent_type or ''}\n")
    except Exception:
        pass


def _parse_agent_type(transcript_path: str) -> str | None:
    try:
        content = Path(transcript_path).read_text(errors="replace", encoding="utf-8")[:4096]
        lower = content.lower()
        for agent_type, markers in _SIGNATURES:
            if all(m.lower() in lower for m in markers):
                return agent_type
    except Exception:
        pass
    return None
