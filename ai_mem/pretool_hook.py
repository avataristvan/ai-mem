#!/usr/bin/env python3
"""PreToolUse hook for Write|Edit — injects relevant past experiences from ai-mem."""
import json
import os
import sys
from pathlib import Path

DB_PATH = Path(os.environ.get("AI_MEM_PATH", Path.home() / ".local" / "share" / "ai-mem"))
TOP_K = 2
MAX_CHARS_PER_HIT = 500


def _build_query_use_case():
    from ai_mem._hook_deps import _build_core, _make_query_uc

    return _make_query_uc(*_build_core(DB_PATH))


def _hits(query_uc, collection: str, query: str):
    try:
        return query_uc.execute(collection=collection, query=query, n_results=TOP_K)
    except Exception:
        return []


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return

    try:
        from ai_mem.agent_context import detect_for_hook
        if not detect_for_hook(payload).should_inject:
            return
    except Exception:
        pass

    if payload.get("tool_name") not in ("Write", "Edit"):
        return

    file_path = payload.get("tool_input", {}).get("file_path", "")
    if not file_path or not DB_PATH.exists():
        return

    try:
        from ai_mem.repo_context import GLOBAL_COLLECTION, WORKSPACE_COLLECTION, detect_repo_context

        query_uc = _build_query_use_case()
    except Exception:
        return

    tool_name = payload["tool_name"]
    file_name = Path(file_path).name
    query = f"{tool_name} {file_name} {file_path}"

    collected: list[tuple[str, object]] = [(GLOBAL_COLLECTION, result) for result in _hits(query_uc, GLOBAL_COLLECTION, query)]

    try:
        ctx = detect_repo_context()
        if ctx.collection not in (GLOBAL_COLLECTION, WORKSPACE_COLLECTION):
            collected.extend((ctx.collection, result) for result in _hits(query_uc, ctx.collection, query))
    except Exception:
        pass

    if not collected:
        return

    lines = [f"[ai-mem] Past experiences for {tool_name} on {file_name}:"]
    for coll, r in collected:
        text = r.text[:MAX_CHARS_PER_HIT]
        if len(r.text) > MAX_CHARS_PER_HIT:
            text += "..."
        score = f"{r.score:.2f}" if r.score is not None else "n/a"
        lines.append(f"- [{coll} score={score}] {text}")

    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "additionalContext": "\n".join(lines),
                }
            }
        )
    )


if __name__ == "__main__":
    main()
