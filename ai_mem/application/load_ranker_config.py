"""Loads optional hybrid-ranker group config from a JSON file."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from ai_mem.domain.learning import RankerScope


class LoadRankerConfigUseCase:
    def __init__(self, config_path: Path) -> None:
        self._path = config_path

    def execute(self) -> dict[str, RankerScope]:
        if not self._path.exists():
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[ai-mem] ranker_config: cannot read {self._path}: {exc!r}", file=sys.stderr)
            return {}

        result: dict[str, RankerScope] = {}
        groups = data.get("groups", [])
        if not isinstance(groups, list):
            print(f"[ai-mem] ranker_config: 'groups' must be a list — ignoring file", file=sys.stderr)
            return {}

        for group in groups:
            name = group.get("name", "") if isinstance(group, dict) else ""
            collections = group.get("collections", []) if isinstance(group, dict) else []
            if not isinstance(name, str) or not name:
                print(f"[ai-mem] ranker_config: skipping malformed group (missing name): {group!r}", file=sys.stderr)
                continue
            if not isinstance(collections, list) or not collections:
                print(f"[ai-mem] ranker_config: skipping group {name!r}: collections must be a non-empty list", file=sys.stderr)
                continue
            if not all(isinstance(c, str) and c for c in collections):
                print(f"[ai-mem] ranker_config: skipping group {name!r}: all collection entries must be non-empty strings", file=sys.stderr)
                continue

            scope = RankerScope(name=name, mode="hybrid", group=name, member_collections=list(collections))
            for col in collections:
                if col in result:
                    print(f"[ai-mem] ranker_config: collection {col!r} appears in multiple groups — last group wins", file=sys.stderr)
                result[col] = scope

        return result
