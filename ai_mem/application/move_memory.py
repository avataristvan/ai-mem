"""Move entries between collections, preserving id/text/metadata verbatim."""
from __future__ import annotations

from dataclasses import dataclass, field

from ai_mem.domain.memory import MemoryRepository

_PREVIEW_LEN = 120


@dataclass
class MoveConflict:
    id: str
    existing_text_preview: str
    source_text_preview: str


@dataclass
class MoveResult:
    moved_ids: list[str] = field(default_factory=list)
    not_found_ids: list[str] = field(default_factory=list)
    conflicts: list[MoveConflict] = field(default_factory=list)


class MoveMemoryUseCase:
    def __init__(self, repo: MemoryRepository) -> None:
        self._repo = repo

    def execute(self, from_collection: str, to_collection: str, ids: list[str]) -> MoveResult:
        if from_collection == to_collection:
            raise ValueError("from_collection and to_collection must differ")

        ids = list(dict.fromkeys(ids))  # dedupe while preserving order
        if not ids:
            return MoveResult()

        source_entries = {entry.id: entry for entry in self._repo.get_by_ids(from_collection, ids)}
        not_found_ids = [id_ for id_ in ids if id_ not in source_entries]
        candidate_ids = [id_ for id_ in ids if id_ in source_entries]

        existing_target = {
            entry.id: entry for entry in self._repo.get_by_ids(to_collection, candidate_ids)
        }

        conflicts: list[MoveConflict] = []
        queued_entries = []
        for id_ in candidate_ids:
            source_entry = source_entries[id_]
            target_entry = existing_target.get(id_)
            if target_entry is not None and target_entry.text != source_entry.text:
                conflicts.append(
                    MoveConflict(
                        id=id_,
                        existing_text_preview=target_entry.text[:_PREVIEW_LEN],
                        source_text_preview=source_entry.text[:_PREVIEW_LEN],
                    )
                )
                continue
            queued_entries.append(source_entry)

        confirmed_ids: list[str] = []
        if queued_entries:
            self._repo.upsert(to_collection, queued_entries)
            queued_ids = [entry.id for entry in queued_entries]
            written_ids = {entry.id for entry in self._repo.get_by_ids(to_collection, queued_ids)}
            confirmed_ids = [id_ for id_ in queued_ids if id_ in written_ids]
            # An empty ids list to repo.delete drops the entire collection, so only
            # call it once we know there's at least one confirmed id to remove.
            if confirmed_ids:
                self._repo.delete(from_collection, confirmed_ids)

        return MoveResult(moved_ids=confirmed_ids, not_found_ids=not_found_ids, conflicts=conflicts)
