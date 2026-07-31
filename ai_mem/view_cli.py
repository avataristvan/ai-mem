"""mem-view CLI — interactive terminal browser for ai-mem collections/entries."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from ai_mem.domain.memory import CollectionInfo, MemoryEntry
from ai_mem.infrastructure.chroma_repository import ChromaMemoryRepository

DB_PATH = Path(os.environ.get("AI_MEM_PATH", Path.home() / ".local/share/ai-mem"))


def _prompt(msg: str) -> str:
    try:
        return input(msg).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)


def _pick(items: list[str], prompt_label: str) -> str | None:
    """Numbered list with live substring filtering. Returns the chosen raw item, or None to go back/quit."""
    filt = ""
    while True:
        shown = [i for i in items if filt.lower() in i.lower()] if filt else items
        print()
        for idx, item in enumerate(shown, 1):
            print(f"  {idx:>3}) {item}")
        if not shown:
            print("  (no matches)")
        suffix = f" [filter: {filt!r}]" if filt else ""
        raw = _prompt(f"\n{prompt_label}{suffix} — number, text to filter, '-' to clear filter, b=back, q=quit: ")
        if raw in ("q", "quit"):
            sys.exit(0)
        if raw in ("b", "back"):
            return None
        if raw == "-":
            filt = ""
            continue
        if raw.isdigit() and 1 <= int(raw) <= len(shown):
            return shown[int(raw) - 1]
        filt = raw


def _format_entry(entry: MemoryEntry) -> str:
    meta_str = ", ".join(f"{k}={v}" for k, v in sorted(entry.metadata.items())) if entry.metadata else "(no metadata)"
    return f"--- {entry.id} ---\n{meta_str}\n\n{entry.text}\n"


def main() -> None:
    repo = ChromaMemoryRepository(DB_PATH)
    collections: list[CollectionInfo] = sorted(repo.list_collections(), key=lambda c: c.name)
    if not collections:
        print(f"No collections found at {DB_PATH}")
        return

    col_labels = [f"{c.name}  ({c.count} entries)" for c in collections]
    name_by_label = {label: c.name for label, c in zip(col_labels, collections)}

    while True:
        chosen_label = _pick(col_labels, "Collection")
        if chosen_label is None:
            return
        col_name = name_by_label[chosen_label]
        entries = repo.get_all(col_name)
        by_id = {entry.id: entry for entry in entries}

        entry_labels = []
        label_to_id = {}
        for entry in entries:
            title = (entry.text.splitlines()[0] if entry.text else "")[:70]
            label = f"{entry.id}  —  {title}"
            entry_labels.append(label)
            label_to_id[label] = entry.id
        entry_labels.sort()

        while True:
            chosen_entry_label = _pick(entry_labels, f"[{col_name}] Entry")
            if chosen_entry_label is None:
                break
            entry = by_id[label_to_id[chosen_entry_label]]
            print()
            print(_format_entry(entry))
            _prompt("Enter to continue: ")


if __name__ == "__main__":
    main()
