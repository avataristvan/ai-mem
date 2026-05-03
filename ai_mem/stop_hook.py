#!/usr/bin/env python3
"""SessionStop hook — reminds Claude to update current_focus and surface split hints."""
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


def _split_hint_ids(collection: str) -> list[str]:
    """Return IDs of entries that qualify for splitting (high access + long text)."""
    try:
        from ai_mem.application.detect_split_hints import SPLIT_MIN_TEXT_CHARS, SPLIT_THRESHOLD_ACCESSES
        from ai_mem.infrastructure.chroma_repository import ChromaMemoryRepository

        if not DB_PATH.exists():
            return []
        repo = ChromaMemoryRepository(DB_PATH)
        return [
            e.id
            for e in repo.get_all(collection)
            if int(e.metadata.get("access_count", 0)) >= SPLIT_THRESHOLD_ACCESSES
            and len(e.text) >= SPLIT_MIN_TEXT_CHARS
        ]
    except Exception:
        return []


def main():
    _try_demote()

    messages: list[str] = []

    try:
        from ai_mem.repo_context import WORKSPACE_COLLECTION, detect_repo_context

        ctx = detect_repo_context()

        r = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            capture_output=True, text=True,
        )
        changed = len([l for l in r.stdout.splitlines() if l.strip()])
        if changed > 0:
            if ctx.collection != WORKSPACE_COLLECTION:
                focus_hint = f'mem_add id=current_focus collection="{ctx.collection}"'
            else:
                focus_hint = "mem_add id=current_focus"
            messages.append(
                f"Files changed this session ({changed} file(s)) — "
                f"update current_focus in ai-mem ({focus_hint})."
            )

        split_ids = _split_hint_ids(ctx.collection)
        if split_ids:
            col_hint = f' collection="{ctx.collection}"' if ctx.collection != WORKSPACE_COLLECTION else ""
            messages.append(
                f"Split hints in '{ctx.collection}': {', '.join(split_ids)} — "
                f"call mem_split{col_hint}."
            )
    except Exception:
        pass

    if messages:
        print("\n".join(messages))
        sys.exit(2)


if __name__ == "__main__":
    main()
