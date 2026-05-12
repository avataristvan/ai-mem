#!/usr/bin/env python3
"""SessionStart hook — injects repo/global current_focus and collection routing into Claude's context."""
import json
import os
import sys
from pathlib import Path

import time

DB_PATH = Path(os.environ.get("AI_MEM_PATH", Path.home() / ".local" / "share" / "ai-mem"))

# Module-level imports for patchability in tests (same pattern as posttool_hook.py).
try:
    from ai_mem.agent_context import detect_for_session_start, write_to_env_file
    from ai_mem.application.get_memory import GetMemoryUseCase
    from ai_mem.application.list_collections import ListCollectionsUseCase
    from ai_mem.infrastructure.chroma_repository import ChromaMemoryRepository
    from ai_mem.repo_context import GLOBAL_COLLECTION, WORKSPACE_COLLECTION, detect_repo_context
    from ai_mem.session_stats import record_injection
except ImportError:
    detect_for_session_start = None  # type: ignore[assignment]
    write_to_env_file = None  # type: ignore[assignment]
    GetMemoryUseCase = None  # type: ignore[assignment]
    ListCollectionsUseCase = None  # type: ignore[assignment]
    ChromaMemoryRepository = None  # type: ignore[assignment]
    GLOBAL_COLLECTION = "global"
    WORKSPACE_COLLECTION = "workspace"
    detect_repo_context = None  # type: ignore[assignment]
    record_injection = None  # type: ignore[assignment]

FOCUS_ID = "current_focus"
_STATS_PATH = DB_PATH / "session_stats.json"
_SESSION_START_FILE = DB_PATH / "session_start.txt"
_PREV_SESSION_MAX_AGE_DAYS = 7
_FOCUS_PREVIEW_CHARS = 150


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _session_delta(db_path: Path, current_count: int) -> int | None:
    """Return entry delta since the last session, or None if no valid prior record exists."""
    prev_file = db_path / "prev_session.json"
    try:
        data = json.loads(prev_file.read_text(encoding="utf-8"))
        age_days = (time.time() - data["ts"]) / 86400
        if age_days > _PREV_SESSION_MAX_AGE_DAYS:
            return None
        delta = current_count - int(data["count"])
        return delta if delta > 0 else None
    except Exception:
        return None


def _write_prev_session(db_path: Path, current_count: int) -> None:
    prev_file = db_path / "prev_session.json"
    try:
        prev_file.write_text(
            json.dumps({"ts": time.time(), "count": current_count}),
            encoding="utf-8",
        )
    except Exception:
        pass


def _focus_text(get_memory, collection: str) -> str | None:
    try:
        entries = get_memory.execute(collection, [FOCUS_ID])
        return entries[0].text if entries and entries[0].text else None
    except Exception:
        return None


_RANKER_MIN_LABELED = 10  # mirrors MIN_LABELED_EXAMPLES in userprompt_hook.py


def _ranker_signal(collection: str) -> str | None:
    """Return a one-line calibration status for the active collection, or None."""
    try:
        from ai_mem.infrastructure.ranker_storage import RankerStorage
        storage = RankerStorage(DB_PATH / "rankers")
        if not storage.load_buffer(collection):
            return None
        n_labeled = storage.labeled_count(collection)
        if n_labeled >= _RANKER_MIN_LABELED:
            return f"Ranker ({collection}): {n_labeled} labeled — calibrated, context hook active"
        return (
            f"Ranker ({collection}): {n_labeled} labeled — cold start "
            f"(< {_RANKER_MIN_LABELED}, context hook inactive — query mem manually)"
        )
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
        stdin_json: dict = json.load(sys.stdin)
    except Exception:
        stdin_json = {}

    if (detect_for_session_start is None or write_to_env_file is None
            or ChromaMemoryRepository is None or GetMemoryUseCase is None
            or ListCollectionsUseCase is None
            or detect_repo_context is None or record_injection is None):
        return

    agent_type: str | None = None
    try:
        agent_ctx = detect_for_session_start(stdin_json)
        agent_type = agent_ctx.agent_type
        write_to_env_file(agent_ctx)
        if not agent_ctx.should_inject:
            return
    except Exception:
        pass

    if not DB_PATH.exists():
        return

    try:
        _SESSION_START_FILE.write_text(str(time.time()))
    except Exception:
        pass

    try:
        ctx = detect_repo_context()
        repo = ChromaMemoryRepository(DB_PATH)
        get_memory = GetMemoryUseCase(repo)

        collections = ListCollectionsUseCase(repo).execute()
        current_count = sum(c.count for c in collections)

        repo_focus = _focus_text(get_memory, ctx.collection) if ctx.collection != WORKSPACE_COLLECTION else None
        global_focus = _focus_text(get_memory, GLOBAL_COLLECTION)

        expert_focus: str | None = None
        expert_collection: str | None = None
        if agent_type:
            expert_collection = f"subagent.{agent_type}"
            expert_focus = _focus_text(get_memory, expert_collection)

        try:
            record_injection(_STATS_PATH, GLOBAL_COLLECTION, injected=global_focus is not None)
        except Exception:
            pass

        _try_seed(ctx, repo, _STATS_PATH)

        parts = []
        if repo_focus:
            parts.append(f"[{ctx.scope_name} focus]\n{_truncate(repo_focus, _FOCUS_PREVIEW_CHARS)}")
        if global_focus:
            parts.append(f"[global focus]\n{_truncate(global_focus, _FOCUS_PREVIEW_CHARS)}")
        if expert_focus:
            parts.append(f"[{agent_type} expertise]\n{_truncate(expert_focus, _FOCUS_PREVIEW_CHARS)}")
        if expert_collection:
            parts.append(
                f'Expert collection: "subagent.{agent_type}". '
                f'Store cross-project learnings there with collection="subagent.{agent_type}".'
            )
        if ctx.has_claude_md:
            parts.append(
                f'Active collection: "{ctx.collection}". '
                f'Pass collection="{ctx.collection}" to mem_add and mem_query in this session.'
            )
            if not repo_focus:
                parts.append("Run /mem-init to set the initial focus for this scope.")

        delta = _session_delta(DB_PATH, current_count)
        if delta is not None:
            label = "entry" if delta == 1 else "entries"
            parts.append(f"Since last session: {delta} new {label} added")

        signal = _ranker_signal(ctx.collection)
        if signal:
            parts.append(signal)

        _write_prev_session(DB_PATH, current_count)

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
