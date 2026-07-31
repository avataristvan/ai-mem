"""Logs which memory ids userprompt_hook pushed to context, so mem-dream can measure whether
pushed entries were later deliberately used (pull-through) -- see arch_decision_push_vs_pull_2026_07_31,
point 6 (userprompt_hook's fate is decided after real usage data, not guessed)."""
from __future__ import annotations

import json
import time
from pathlib import Path

_MAX_EVENTS_PER_COLLECTION = 500


def record_pushed(log_path: Path, collection: str, ids: list[str]) -> None:
    """Append one push event per id, timestamped now. Trims each collection's event
    list to the most recent _MAX_EVENTS_PER_COLLECTION entries."""
    if not ids:
        return
    data = _load(log_path)
    events = data.get(collection, [])
    now = time.time()
    events.extend({"id": entry_id, "ts": now} for entry_id in ids)
    data[collection] = events[-_MAX_EVENTS_PER_COLLECTION:]
    _save(log_path, data)


def load_events(log_path: Path, collection: str) -> list[dict]:
    """Return the raw push events {"id", "ts"} logged for this collection."""
    return _load(log_path).get(collection, [])


def _load(log_path: Path) -> dict[str, list[dict]]:
    if not log_path.exists():
        return {}
    try:
        return json.loads(log_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save(log_path: Path, data: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(data), encoding="utf-8")
