#!/usr/bin/env python3
"""SessionStart hook — injects repo/global current_focus and collection routing into Claude's context."""
import json
import os
from pathlib import Path

DB_PATH = Path(os.environ.get("AI_MEM_PATH", Path.home() / ".local" / "share" / "ai-mem"))
FOCUS_ID = "current_focus"
_STATS_PATH = DB_PATH / "session_stats.json"


def _focus_text(get_memory, collection: str) -> str | None:
    try:
        entries = get_memory.execute(collection, [FOCUS_ID])
        return entries[0].text if entries and entries[0].text else None
    except Exception:
        return None


def _try_seed(ctx, repo, stats_path: Path) -> None:
    try:
        from ai_mem.application.add_memory import AddMemoryUseCase
        from ai_mem.application.list_collections import ListCollectionsUseCase
        from ai_mem.repo_context import WORKSPACE_COLLECTION
        from ai_mem.repo_seeder import seed_collection

        if ctx.collection == WORKSPACE_COLLECTION or ctx.claude_md_dir is None:
            return

        collections = ListCollectionsUseCase(repo).execute()
        existing = {c.name: c.count for c in collections}
        if existing.get(ctx.collection, 0) > 0:
            return

        add_uc = AddMemoryUseCase(repo)
        seed_collection(
            collection=ctx.collection,
            claude_md_path=ctx.claude_md_dir / "CLAUDE.md",
            add_uc=add_uc,
            stats_path=stats_path,
        )
    except Exception:
        pass


def main():
    try:
        from ai_mem.application.get_memory import GetMemoryUseCase
        from ai_mem.infrastructure.chroma_repository import ChromaMemoryRepository
        from ai_mem.repo_context import GLOBAL_COLLECTION, WORKSPACE_COLLECTION, detect_repo_context
        from ai_mem.session_stats import record_injection
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

        try:
            record_injection(_STATS_PATH, GLOBAL_COLLECTION, injected=global_focus is not None)
        except Exception:
            pass

        _try_seed(ctx, repo, _STATS_PATH)

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
