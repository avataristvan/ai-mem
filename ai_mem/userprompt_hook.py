#!/usr/bin/env python3
"""UserPromptSubmit hook — injects relevant memories into Claude's context."""
import json
import os
import sys
from pathlib import Path
from typing import Any

from ai_mem.repo_context import GLOBAL_COLLECTION, WORKSPACE_COLLECTION, detect_repo_context

DB_PATH = Path(os.environ.get("AI_MEM_PATH", Path.home() / ".local" / "share" / "ai-mem"))
_STATS_PATH = DB_PATH / "session_stats.json"
TOP_K = 3
MAX_CHARS_PER_HIT = 300
MAX_TOTAL_CHARS = 1500
CONTEXT_MIN_SCORE = 0.3
SESSION_TTL_HOURS = 4
ANTIPATTERN_TOP_K = 2
MAX_CHARS_PER_ANTIPATTERN = 200
ANTIPATTERN_MIN_SCORE = 0.4
DILEMMA_TOP_K = 2
MAX_CHARS_PER_DILEMMA = 250
DILEMMA_MIN_SCORE = 0.4
ANTICIPATION_QUESTION = "  → Anticipation: Who holds the same role now? What would happen to them?"


def _load_session_injected(db_path: Path) -> set[str]:
    """Return the set of entry IDs already injected this session.

    Returns an empty set if the file is absent, unreadable, or older than SESSION_TTL_HOURS.
    """
    import time

    path = db_path / "session_injected.json"
    try:
        data = json.loads(path.read_text())
        age_hours = (time.time() - data["session_ts"]) / 3600
        if age_hours > SESSION_TTL_HOURS:
            return set()
        return set(data.get("ids", []))
    except Exception:
        return set()


def _save_session_injected(db_path: Path, ids: set[str]) -> None:
    """Persist the current session's injected IDs; silently ignores I/O errors."""
    import time

    path = db_path / "session_injected.json"
    try:
        existing = _load_session_injected(db_path)
        merged = existing | ids
        ts = time.time()
        # Preserve the original session_ts if the file is still valid.
        try:
            data = json.loads(path.read_text())
            age_hours = (time.time() - data["session_ts"]) / 3600
            if age_hours <= SESSION_TTL_HOURS:
                ts = data["session_ts"]
        except Exception:
            pass
        path.write_text(json.dumps({"session_ts": ts, "ids": list(merged)}))
    except Exception:
        pass


def _build_deps():
    from ai_mem._hook_deps import _build_query_uc

    query_uc = _build_query_uc(DB_PATH, with_bm25=True)
    return query_uc


def _hits(query_uc, collection: str, query: str):
    try:
        return query_uc.execute(collection=collection, query=query, n_results=TOP_K)
    except Exception:
        return []


def _antipattern_hits(query_uc, collection: str, query: str):
    try:
        results = query_uc.execute(
            collection=collection,
            query=query,
            n_results=ANTIPATTERN_TOP_K,
            type_filter="anti-pattern",
        )
        return [r for r in results if (r.score or 0.0) >= ANTIPATTERN_MIN_SCORE]
    except Exception:
        return []


def _dilemma_hits(query_uc, collection: str, query: str):
    try:
        results = query_uc.execute(
            collection=collection,
            query=query,
            n_results=DILEMMA_TOP_K,
            type_filter="dilemma",
        )
        return [r for r in results if (r.score or 0.0) >= DILEMMA_MIN_SCORE]
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

    query = payload.get("prompt", "").strip()
    if not query or not DB_PATH.exists():
        return

    try:
        query_uc = _build_deps()
    except Exception:
        return

    global_results = _hits(query_uc, GLOBAL_COLLECTION, query)
    global_hits = [r for r in global_results if (r.score or 0.0) >= CONTEXT_MIN_SCORE]

    repo_hits = []
    repo_collection = None
    antipattern_results = []
    dilemma_results = []
    try:
        ctx = detect_repo_context()
        if ctx.collection not in (GLOBAL_COLLECTION, WORKSPACE_COLLECTION):
            repo_collection = ctx.collection
            repo_results = _hits(query_uc, repo_collection, query)
            repo_hits = [r for r in repo_results if (r.score or 0.0) >= CONTEXT_MIN_SCORE]
            antipattern_results = _antipattern_hits(query_uc, repo_collection, query)
            dilemma_results = _dilemma_hits(query_uc, repo_collection, query)
    except Exception:
        pass

    if not global_hits and not repo_hits and not antipattern_results and not dilemma_results:
        return

    collected: list[tuple[str, Any]] = []
    collected.extend((GLOBAL_COLLECTION, r) for r in global_hits)
    if repo_collection:
        collected.extend((repo_collection, r) for r in repo_hits)

    try:
        from ai_mem.session_stats import record_injection
        record_injection(_STATS_PATH, GLOBAL_COLLECTION, injected=bool(collected))
    except Exception:
        pass

    if not collected and not antipattern_results and not dilemma_results:
        return

    # Per-session dedup: skip entries already injected in this session.
    already_injected = _load_session_injected(DB_PATH)
    collected = [(coll, r) for coll, r in collected if getattr(r, "id", None) not in already_injected or getattr(r, "id", None) is None]
    antipattern_results = [r for r in antipattern_results if getattr(r, "id", None) not in already_injected]
    dilemma_results = [r for r in dilemma_results if getattr(r, "id", None) not in already_injected]

    # Combined budget cap: include entries until MAX_TOTAL_CHARS is reached.
    budget_collected: list[tuple[str, Any]] = []
    chars_used = 0
    for coll, r in collected:
        entry_len = min(len(r.text), MAX_CHARS_PER_HIT)
        if chars_used + entry_len > MAX_TOTAL_CHARS:
            break
        budget_collected.append((coll, r))
        chars_used += entry_len
    collected = budget_collected

    if not collected and not antipattern_results and not dilemma_results:
        return

    lines: list[str] = []
    injected_ids: set[str] = set()

    if dilemma_results:
        lines.append("[ai-mem dilemmas]")
        for r in dilemma_results:
            text = r.text[:MAX_CHARS_PER_DILEMMA]
            if len(r.text) > MAX_CHARS_PER_DILEMMA:
                text += "..."
            lines.append(f"⚖ {text}")
            entry_id = getattr(r, "id", None)
            if entry_id is not None:
                injected_ids.add(entry_id)

    if antipattern_results:
        if lines:
            lines.append("")
        lines.append("[ai-mem warnings]")
        for r in antipattern_results:
            text = r.text[:MAX_CHARS_PER_ANTIPATTERN]
            if len(r.text) > MAX_CHARS_PER_ANTIPATTERN:
                text += "..."
            lines.append(f"⚠ {text}")
            if "Affected:" in r.text:
                lines.append(ANTICIPATION_QUESTION)
            entry_id = getattr(r, "id", None)
            if entry_id is not None:
                injected_ids.add(entry_id)

    if collected:
        if lines:
            lines.append("")
        lines.append("[ai-mem] Relevant context for your prompt:")
        for coll, r in collected:
            text = r.text[:MAX_CHARS_PER_HIT]
            if len(r.text) > MAX_CHARS_PER_HIT:
                text += "..."
            score = f"{r.score:.2f}" if r.score is not None else "n/a"
            lines.append(f"- [{coll} score={score}] {text}")
            entry_id = getattr(r, "id", None)
            if entry_id is not None:
                injected_ids.add(entry_id)

    _save_session_injected(DB_PATH, injected_ids)

    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": "\n".join(lines),
                }
            }
        )
    )


if __name__ == "__main__":
    main()
