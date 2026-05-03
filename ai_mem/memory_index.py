"""Auto-demotion of MEMORY.md entries to ai-mem when the index approaches its line limit."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

DEMOTION_THRESHOLD = 180  # demote when MEMORY.md reaches this many lines
_TARGET_LINES = 160       # bring index down to this after demotion

# Lower value = demoted first (project is time-sensitive; feedback is set-and-forget)
_TYPE_PRIORITY = {"project": 0, "reference": 1, "user": 2, "feedback": 3}


@dataclass
class IndexEntry:
    line: str       # raw MEMORY.md line
    title: str
    filename: str   # relative to memory_dir
    mem_type: str   # "feedback" | "project" | "user" | "reference" | ""
    mtime: float    # file mtime (seconds since epoch)


def find_memory_dir(cwd: Path | None = None, home: Path | None = None) -> Path | None:
    """Return the Claude Code auto-memory directory for the given project, or None."""
    cwd = cwd or Path.cwd()
    home = home or Path.home()
    encoded = str(cwd).replace("/", "-")
    candidate = home / ".claude" / "projects" / encoded / "memory"
    return candidate if candidate.exists() else None


def _parse_type(content: str) -> str:
    m = re.search(r'^type:\s*(\S+)', content, re.MULTILINE)
    return m.group(1).strip("\"'") if m else ""


def read_index(memory_dir: Path) -> tuple[list[IndexEntry], int]:
    """Return (parsed entry list, total MEMORY.md line count)."""
    memory_md = memory_dir / "MEMORY.md"
    if not memory_md.exists():
        return [], 0
    lines = memory_md.read_text(encoding="utf-8").splitlines()
    entries: list[IndexEntry] = []
    for line in lines:
        m = re.match(r'-\s+\[([^\]]+)\]\(([^)]+\.md)\)', line)
        if not m:
            continue
        title, filename = m.group(1), m.group(2)
        fpath = memory_dir / filename
        mem_type = _parse_type(fpath.read_text(encoding="utf-8")) if fpath.exists() else ""
        mtime = fpath.stat().st_mtime if fpath.exists() else 0.0
        entries.append(IndexEntry(line=line, title=title, filename=filename, mem_type=mem_type, mtime=mtime))
    return entries, len(lines)


def _sort_key(entry: IndexEntry) -> tuple[int, float]:
    return (_TYPE_PRIORITY.get(entry.mem_type, 1), entry.mtime)


def auto_demote(
    memory_dir: Path,
    add_uc,
    global_collection: str = "global",
    project_collection: str | None = None,
    _threshold: int | None = None,
    _target_lines: int | None = None,
    stale_days: int | None = 30,
) -> list[str]:
    """
    Migrate entries to ai-mem when either:
    - MEMORY.md >= threshold lines (priority-ordered demotion down to target), or
    - An entry's backing file is older than stale_days (proactive, regardless of threshold).

    Returns the titles of demoted entries.
    """
    threshold = _threshold if _threshold is not None else DEMOTION_THRESHOLD
    target = _target_lines if _target_lines is not None else _TARGET_LINES
    entries, line_count = read_index(memory_dir)
    if not entries:
        return []

    stale_cutoff = time.time() - stale_days * 86400 if stale_days is not None else None
    is_over_threshold = line_count >= threshold
    has_stale = stale_cutoff is not None and any(
        e.mtime > 0 and e.mtime < stale_cutoff for e in entries
    )
    if not is_over_threshold and not has_stale:
        return []

    to_demote: list[IndexEntry] = []
    scheduled: set[str] = set()  # line strings already queued

    # Phase 1: threshold-based demotion (priority order)
    if is_over_threshold:
        to_free = max(line_count - target, 1)
        freed = 0
        for entry in sorted(entries, key=_sort_key):
            if freed >= to_free:
                break
            to_demote.append(entry)
            scheduled.add(entry.line)
            freed += 1

    # Phase 2: stale demotion (entries not touched in stale_days, skipping already queued)
    if stale_cutoff is not None:
        for entry in entries:
            if entry.line not in scheduled and entry.mtime > 0 and entry.mtime < stale_cutoff:
                to_demote.append(entry)
                scheduled.add(entry.line)

    if not to_demote:
        return []

    memory_md = memory_dir / "MEMORY.md"
    lines_to_remove: set[str] = set()
    demoted_titles: list[str] = []

    for entry in to_demote:
        fpath = memory_dir / entry.filename
        if not fpath.exists():
            lines_to_remove.add(entry.line)
            continue

        collection = (
            project_collection if entry.mem_type == "project" and project_collection
            else global_collection
        )
        text = fpath.read_text(encoding="utf-8")
        entry_id = Path(entry.filename).stem
        try:
            add_uc.execute(
                collection=collection,
                documents=[text],
                ids=[entry_id],
                metadatas=[{
                    "demoted_from": "MEMORY.md",
                    "mem_type": entry.mem_type,
                    "demoted_at": time.time(),
                }],
            )
        except Exception:
            continue  # keep in index on failure

        lines_to_remove.add(entry.line)
        demoted_titles.append(entry.title)

    if lines_to_remove:
        original = memory_md.read_text(encoding="utf-8")
        kept = [
            l for l in original.splitlines(keepends=True)
            if l.rstrip("\n") not in lines_to_remove
        ]
        memory_md.write_text("".join(kept), encoding="utf-8")

    return demoted_titles
