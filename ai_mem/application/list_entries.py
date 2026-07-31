"""List all entries in a collection as id + title pairs."""
from __future__ import annotations

from ai_mem.domain.memory import MemoryRepository

_MAX_TITLE_LEN = 80


def title_of(text: str, max_len: int = _MAX_TITLE_LEN) -> str:
    """First non-empty line of an entry's text, truncated — shared by mem_list, mem_topics,
    and the [available] injection format in hook.py / userprompt_hook.py."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:max_len]
    return ""


class ListEntriesUseCase:
    def __init__(self, repo: MemoryRepository) -> None:
        self._repo = repo

    def execute(self, collection: str) -> list[dict]:
        entries = self._repo.get_all(collection)
        return [{"id": entry.id, "title": title_of(entry.text)} for entry in entries]
