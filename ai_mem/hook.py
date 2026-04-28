#!/usr/bin/env python3
"""SessionStart hook — injects repo/global current_focus and collection routing into Claude's context."""
import json
import os
from pathlib import Path

DB_PATH = Path(os.environ.get("AI_MEM_PATH", Path.home() / ".local" / "share" / "ai-mem"))
FOCUS_ID = "current_focus"


def _focus_text(get_memory, collection: str) -> str | None:
    try:
        entries = get_memory.execute(collection, [FOCUS_ID])
        return entries[0].text if entries and entries[0].text else None
    except Exception:
        return None


def main():
    try:
        from ai_mem.application.get_memory import GetMemoryUseCase
        from ai_mem.infrastructure.chroma_repository import ChromaMemoryRepository
        from ai_mem.repo_context import GLOBAL_COLLECTION, WORKSPACE_COLLECTION, detect_repo_context
    except ImportError:
        return

    if not DB_PATH.exists():
        return

    try:
        ctx = detect_repo_context()
        repo = ChromaMemoryRepository(DB_PATH)
        get_memory = GetMemoryUseCase(repo)

        repo_focus = _focus_text(get_memory, ctx.collection) if ctx.collection != WORKSPACE_COLLECTION else None
        global_focus = _focus_text(get_memory, GLOBAL_COLLECTION)

        parts = []
        if repo_focus:
            parts.append(f"[{ctx.scope_name} focus]\n{repo_focus}")
        if global_focus:
            parts.append(f"[global focus]\n{global_focus}")
        if ctx.has_claude_md:
            parts.append(
                f'Active collection: "{ctx.collection}". '
                f'Pass collection="{ctx.collection}" to mem_add and mem_query in this session.'
            )
            if not repo_focus:
                parts.append("Run /mem-init to set the initial focus for this scope.")

        if not parts:
            return

        output = {
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": "[ai-mem]\n" + "\n\n".join(parts),
            }
        }
        print(json.dumps(output))
    except Exception:
        pass


if __name__ == "__main__":
    main()
