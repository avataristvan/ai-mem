"""Group a collection's entries by type — the missing rung between mem_list's flat
{id, title} dump and mem_get's full text, so a large collection stays browsable.

Mirrors the list_tool_categories -> list_tools_in_category -> get_tool_signature
pattern (see arch_redesign_sketch_pull_based_2026_07_31 in repo.ai-mem): mem_topics is
the "list_tools_in_category" rung, grouping by the 'type' metadata field that already
exists on every entry (pattern/anti-pattern/project/fact/feedback/reference/dilemma).
"""
from __future__ import annotations

from ai_mem.application.list_entries import title_of
from ai_mem.domain.memory import MemoryRepository

_UNTYPED = "untyped"


class ListTopicsUseCase:
    def __init__(self, repo: MemoryRepository) -> None:
        self._repo = repo

    def execute(self, collection: str) -> list[dict]:
        entries = self._repo.get_all(collection)
        topics: dict[str, list[dict]] = {}
        for entry in entries:
            topic = entry.metadata.get("type") or _UNTYPED
            topics.setdefault(topic, []).append({"id": entry.id, "title": title_of(entry.text)})

        return [
            {"type": topic, "count": len(items), "entries": items}
            for topic, items in sorted(topics.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        ]
