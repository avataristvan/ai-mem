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
_GIT_COMMITS_MAX = 5
_HIGH_CONFIDENCE_THRESHOLD = 0.9
_HIGH_CONFIDENCE_MIN_ACCESS = 3
_HIGH_CONFIDENCE_MIN_BOOSTS = 1
_HIGH_CONFIDENCE_MAX = 3
_HIGH_CONFIDENCE_CHARS = 300
_EXPIRED_MAX = 3
_EXPIRED_PREVIEW_CHARS = 120


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


def _git_commits_since(db_path: Path) -> list[str]:
    """Return one-line git log entries since the previous session. Silent on failure."""
    import subprocess
    from datetime import datetime
    prev_file = db_path / "prev_session.json"
    try:
        data = json.loads(prev_file.read_text(encoding="utf-8"))
        age_days = (time.time() - data["ts"]) / 86400
        if age_days > _PREV_SESSION_MAX_AGE_DAYS:
            return []
        since = datetime.fromtimestamp(data["ts"]).strftime("%Y-%m-%dT%H:%M:%S")
        result = subprocess.run(
            ["git", "log", "--oneline", f"--since={since}", f"-{_GIT_COMMITS_MAX}"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        return []


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


def _high_confidence_entries(repo, collection: str, exclude_ids: set[str]) -> list:
    """Return entries with confidence > threshold and enough access history, sorted by confidence desc."""
    try:
        all_entries = repo.get_all(collection)
        candidates = []
        for e in all_entries:
            if e.id in exclude_ids:
                continue
            try:
                conf = float(e.metadata.get("confidence", 0.0))
            except (ValueError, TypeError):
                continue
            ac = int(e.metadata.get("access_count", 0))
            bc = int(e.metadata.get("boost_count", 0))
            if conf > _HIGH_CONFIDENCE_THRESHOLD and ac >= _HIGH_CONFIDENCE_MIN_ACCESS and bc >= _HIGH_CONFIDENCE_MIN_BOOSTS:
                candidates.append((conf, e))
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in candidates[:_HIGH_CONFIDENCE_MAX]]
    except Exception:
        return []



def _expired_entries(repo, collection: str, now_ts: float) -> list:
    """Return entries with expires_at in the past, sorted oldest-first."""
    try:
        result = []
        for e in repo.get_all(collection):
            raw = e.metadata.get("expires_at")
            if raw is None:
                continue
            try:
                exp = float(raw)
            except (ValueError, TypeError):
                continue
            if exp < now_ts:
                result.append((exp, e))
        result.sort(key=lambda x: x[0])
        return [e for _, e in result]
    except Exception:
        return []


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
        expert_entries: list = []
        if agent_type:
            expert_collection = f"subagent.{agent_type}"
            expert_focus = _focus_text(get_memory, expert_collection)
            _query = repo_focus or f"{ctx.collection} patterns and best practices"
            try:
                _results = repo.query(expert_collection, _query, n_results=3, max_age_days=None)
                expert_entries = [r for r in _results if r.id != FOCUS_ID][:2]
            except Exception:
                pass

        try:
            record_injection(_STATS_PATH, GLOBAL_COLLECTION, injected=global_focus is not None)
        except Exception:
            pass

        parts = []
        if repo_focus:
            parts.append(f"[{ctx.scope_name} focus]\n{_truncate(repo_focus, _FOCUS_PREVIEW_CHARS)}")
        if global_focus:
            parts.append(f"[global focus]\n{_truncate(global_focus, _FOCUS_PREVIEW_CHARS)}")
        if expert_focus or expert_entries:
            block = f"[{agent_type} expertise]"
            if expert_focus:
                block += f"\n{_truncate(expert_focus, _FOCUS_PREVIEW_CHARS)}"
            if expert_entries:
                block += "\nRelevant past learnings:\n" + "\n".join(
                    f"- {_truncate(e.text, 200)}" for e in expert_entries
                )
            parts.append(block)
        if expert_collection:
            parts.append(
                f'Expert collection: "subagent.{agent_type}". '
                f'Store cross-project learnings there with collection="subagent.{agent_type}".'
            )
        if ctx.collection != WORKSPACE_COLLECTION:
            _hc_exclude = {FOCUS_ID}
            _hc_entries = _high_confidence_entries(repo, ctx.collection, _hc_exclude)
            if _hc_entries:
                parts.append(
                    "[always-present]\n"
                    + "\n".join(f"- {_truncate(e.text, _HIGH_CONFIDENCE_CHARS)}" for e in _hc_entries)
                )
        _now_ts = time.time()
        _expired: list = []
        if ctx.collection != WORKSPACE_COLLECTION:
            _expired.extend(_expired_entries(repo, ctx.collection, _now_ts))
        _expired.extend(_expired_entries(repo, GLOBAL_COLLECTION, _now_ts))
        if _expired:
            from datetime import datetime
            _exp_lines = []
            for e in _expired[:_EXPIRED_MAX]:
                exp_ts = float(e.metadata.get("expires_at", 0))
                exp_date = datetime.fromtimestamp(exp_ts).strftime("%Y-%m-%d")
                preview = _truncate(e.text, _EXPIRED_PREVIEW_CHARS)
                _exp_lines.append(f"⏰ {e.id}: \"{preview}\" (expired {exp_date})")
            parts.append("[ai-mem expired]\n" + "\n".join(_exp_lines))

        if ctx.has_claude_md:
            parts.append(
                f'Active collection: "{ctx.collection}". '
                f'Pass collection="{ctx.collection}" to mem_add and mem_query in this session.'
            )
            if not repo_focus:
                parts.append("No focus entry found. Add one via mem_add with id=\"current_focus\".")

        delta = _session_delta(DB_PATH, current_count)
        if delta is not None:
            label = "entry" if delta == 1 else "entries"
            parts.append(f"Since last session: {delta} new {label} added")

        git_commits = _git_commits_since(DB_PATH)
        if git_commits:
            parts.append("Git commits since last session:\n" + "\n".join(f"  {c}" for c in git_commits))

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
