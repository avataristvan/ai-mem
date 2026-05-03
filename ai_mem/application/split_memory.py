"""SplitMemoryUseCase — break coarse entries into focused sub-entries via Claude."""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field

from ai_mem.application.add_memory import AddMemoryUseCase
from ai_mem.application.detect_split_hints import SPLIT_MIN_TEXT_CHARS, SPLIT_THRESHOLD_ACCESSES
from ai_mem.domain.memory import MemoryEntry, MemoryRepository

_MODEL = "claude-haiku-4-5-20251001"

_SPLIT_PROMPT = """\
Split the following memory entry into 2-3 focused sub-entries, each covering a distinct sub-topic.
Keep each sub-entry self-contained and concise. Derive sub-IDs from the original ID.

Return valid JSON only — no prose, no markdown fences.
Format:
[
  {{"id": "<original_id>_1", "text": "..."}},
  {{"id": "<original_id>_2", "text": "..."}}
]

Entry ID: {id}
Entry text:
{text}"""


@dataclass
class SplitResult:
    original_id: str
    new_ids: list[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""


class SplitMemoryUseCase:
    def __init__(
        self,
        repo: MemoryRepository,
        add_uc: AddMemoryUseCase,
        model: str = _MODEL,
    ) -> None:
        self._repo = repo
        self._add_uc = add_uc
        self._model = model

    def execute(self, collection: str, entry_id: str | None = None) -> list[SplitResult]:
        """Split one entry by ID, or all hinted entries in the collection when entry_id is None."""
        if entry_id:
            entries = self._repo.get_by_ids(collection, [entry_id])
        else:
            entries = [
                e for e in self._repo.get_all(collection)
                if int(e.metadata.get("access_count", 0)) >= SPLIT_THRESHOLD_ACCESSES
                and len(e.text) >= SPLIT_MIN_TEXT_CHARS
            ]

        return [self._split_one(collection, entry) for entry in entries]

    def _split_one(self, collection: str, entry: MemoryEntry) -> SplitResult:
        prompt = _SPLIT_PROMPT.format(id=entry.id, text=entry.text)
        try:
            raw = subprocess.run(
                ["claude", "--print", "--model", self._model],
                input=prompt,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except subprocess.CalledProcessError as exc:
            return SplitResult(
                original_id=entry.id,
                skipped=True,
                skip_reason=f"claude CLI error (exit {exc.returncode}): {exc.stderr.strip()[:200]}",
            )
        except FileNotFoundError:
            return SplitResult(
                original_id=entry.id,
                skipped=True,
                skip_reason="claude CLI not found — ensure claude is on PATH",
            )

        try:
            sub_entries = _parse_json(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            return SplitResult(
                original_id=entry.id,
                skipped=True,
                skip_reason=f"JSON parse error: {exc}",
            )

        self._repo.delete(collection, [entry.id])
        new_ids = [e["id"] for e in sub_entries]
        self._add_uc.execute(
            collection=collection,
            documents=[e["text"] for e in sub_entries],
            ids=new_ids,
        )
        return SplitResult(original_id=entry.id, new_ids=new_ids)


def _parse_json(raw: str) -> list[dict]:
    """Parse a JSON array from raw text; falls back to extracting between [ and ]."""
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("["), raw.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise
        result = json.loads(raw[start : end + 1])
    if not isinstance(result, list):
        raise ValueError(f"expected a JSON array, got {type(result).__name__}")
    for item in result:
        if not isinstance(item, dict) or "id" not in item or "text" not in item:
            raise ValueError(f"each item must have 'id' and 'text' keys, got {item!r}")
    return result
