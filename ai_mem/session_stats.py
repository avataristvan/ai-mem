"""Rolling injection stats — tracks per-scope whether hooks successfully injected context."""
from __future__ import annotations

import json
import time
from pathlib import Path

WINDOW = 20


def record_injection(stats_path: Path, scope: str, injected: bool) -> None:
    """Append a session record and trim to the last WINDOW entries."""
    data: dict[str, list[dict]] = {}
    if stats_path.exists():
        try:
            data = json.loads(stats_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}

    records = data.get(scope, [])
    records.append({"ts": time.time(), "injected": injected})
    data[scope] = records[-WINDOW:]

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(data), encoding="utf-8")


def injection_rate(stats_path: Path, scope: str) -> float:
    """Return injected/total for this scope (0.0 if no data)."""
    if not stats_path.exists():
        return 0.0
    try:
        data = json.loads(stats_path.read_text(encoding="utf-8"))
    except Exception:
        return 0.0

    records = data.get(scope, [])
    if not records:
        return 0.0
    return sum(1 for r in records if r.get("injected")) / len(records)
