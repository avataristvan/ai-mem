#!/usr/bin/env python3
"""SessionStop hook — reminds Claude to update current_focus after file changes."""
import os
import subprocess
import sys
from pathlib import Path

DB_PATH = Path(os.environ.get("AI_MEM_PATH", Path.home() / ".local" / "share" / "ai-mem"))


def _try_demote() -> None:
    """Silently migrate stale MEMORY.md entries to ai-mem when approaching the line limit."""
    try:
        from ai_mem.application.add_memory import AddMemoryUseCase
        from ai_mem.infrastructure.chroma_repository import ChromaMemoryRepository
        from ai_mem.memory_index import auto_demote, find_memory_dir
        from ai_mem.repo_context import GLOBAL_COLLECTION, detect_repo_context

        memory_dir = find_memory_dir()
        if not memory_dir or not DB_PATH.exists():
            return

        ctx = detect_repo_context()
        add_uc = AddMemoryUseCase(ChromaMemoryRepository(DB_PATH))
        auto_demote(
            memory_dir=memory_dir,
            add_uc=add_uc,
            global_collection=GLOBAL_COLLECTION,
            project_collection=ctx.collection,
        )
    except Exception:
        pass


def main():
    _try_demote()
    try:
        r = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            capture_output=True, text=True,
        )
        changed = len([l for l in r.stdout.splitlines() if l.strip()])
        if changed == 0:
            return

        from ai_mem.repo_context import detect_repo_context, WORKSPACE_COLLECTION

        ctx = detect_repo_context()
        if ctx.collection != WORKSPACE_COLLECTION:
            hint = f'mem_add id=current_focus collection="{ctx.collection}"'
        else:
            hint = "mem_add id=current_focus"

        print(
            f"Files changed this session ({changed} file(s)) — "
            f"update current_focus in ai-mem ({hint})."
        )
        sys.exit(2)
    except Exception:
        pass


if __name__ == "__main__":
    main()
