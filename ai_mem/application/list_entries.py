"""List all entries in a collection as id + title pairs."""
from __future__ import annotations

from ai_mem.domain.memory import MemoryRepository

_MAX_TITLE_LEN = 80


def _title(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:_MAX_TITLE_LEN]
    return ""


class ListEntriesUseCase:
    def __init__(self, repo: MemoryRepository) -> None:
        self._repo = repo

    def execute(self, collection: str) -> list[dict]:
        entries = self._repo.get_all(collection)
        return [{"id": e.id, "title": _title(e.text)} for e in entries]
