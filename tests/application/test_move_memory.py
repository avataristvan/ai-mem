"""Tests for MoveMemoryUseCase."""
from __future__ import annotations

import pytest

from ai_mem.application.add_memory import AddMemoryUseCase
from ai_mem.application.move_memory import MoveMemoryUseCase


def _seed(repo, collection: str, docs: dict[str, str], metadatas: list[dict] | None = None) -> None:
    """Add entries: {id: text}."""
    AddMemoryUseCase(repo).execute(
        collection=collection,
        documents=list(docs.values()),
        ids=list(docs.keys()),
        metadatas=metadatas,
    )


def test_move_basic_preserves_metadata(tmp_repo):
    _seed(tmp_repo, "source", {"a": "some entry text"}, metadatas=[{"confidence": 0.9, "type": "pattern"}])

    move = MoveMemoryUseCase(tmp_repo)
    result = move.execute("source", "target", ["a"])

    assert result.moved_ids == ["a"]
    assert result.not_found_ids == []
    assert result.conflicts == []
    assert tmp_repo.get_by_ids("source", ["a"]) == []

    moved = tmp_repo.get_by_ids("target", ["a"])
    assert len(moved) == 1
    assert moved[0].text == "some entry text"
    assert moved[0].metadata["confidence"] == 0.9
    assert moved[0].metadata["type"] == "pattern"


def test_move_missing_id_in_source(tmp_repo):
    _seed(tmp_repo, "source", {"a": "present entry"})

    move = MoveMemoryUseCase(tmp_repo)
    result = move.execute("source", "target", ["a", "ghost"])

    assert result.moved_ids == ["a"]
    assert result.not_found_ids == ["ghost"]
    assert result.conflicts == []


def test_move_missing_id_only_writes_nothing(tmp_repo):
    move = MoveMemoryUseCase(tmp_repo)
    result = move.execute("source", "target", ["ghost"])

    assert result.moved_ids == []
    assert result.not_found_ids == ["ghost"]
    assert result.conflicts == []
    assert tmp_repo.get_by_ids("target", ["ghost"]) == []


def test_move_conflict_when_target_has_different_text(tmp_repo):
    _seed(tmp_repo, "source", {"a": "source version of the text"})
    _seed(tmp_repo, "target", {"a": "different target version"})

    move = MoveMemoryUseCase(tmp_repo)
    result = move.execute("source", "target", ["a"])

    assert result.moved_ids == []
    assert len(result.conflicts) == 1
    conflict = result.conflicts[0]
    assert conflict.id == "a"
    assert conflict.existing_text_preview == "different target version"
    assert conflict.source_text_preview == "source version of the text"

    # untouched on both sides
    assert tmp_repo.get_by_ids("source", ["a"])[0].text == "source version of the text"
    assert tmp_repo.get_by_ids("target", ["a"])[0].text == "different target version"


def test_move_no_conflict_when_target_has_identical_text(tmp_repo):
    _seed(tmp_repo, "source", {"a": "identical text"})
    _seed(tmp_repo, "target", {"a": "identical text"})

    move = MoveMemoryUseCase(tmp_repo)
    result = move.execute("source", "target", ["a"])

    assert result.moved_ids == ["a"]
    assert result.conflicts == []
    assert tmp_repo.get_by_ids("source", ["a"]) == []
    assert tmp_repo.get_by_ids("target", ["a"])[0].text == "identical text"


def test_move_creates_target_collection_implicitly(tmp_repo):
    _seed(tmp_repo, "source", {"a": "entry for a brand new collection"})

    move = MoveMemoryUseCase(tmp_repo)
    result = move.execute("source", "brand_new_target", ["a"])

    assert result.moved_ids == ["a"]
    assert tmp_repo.get_by_ids("brand_new_target", ["a"])[0].text == "entry for a brand new collection"


def test_move_mixed_batch(tmp_repo):
    _seed(tmp_repo, "source", {
        "clean": "clean move entry",
        "conflicted": "source side of conflict",
        "same_text": "shared text unchanged",
    })
    _seed(tmp_repo, "target", {
        "conflicted": "target side of conflict",
        "same_text": "shared text unchanged",
    })

    move = MoveMemoryUseCase(tmp_repo)
    result = move.execute("source", "target", ["clean", "conflicted", "same_text", "missing"])

    assert set(result.moved_ids) == {"clean", "same_text"}
    assert result.not_found_ids == ["missing"]
    assert len(result.conflicts) == 1
    assert result.conflicts[0].id == "conflicted"

    assert tmp_repo.get_by_ids("source", ["clean"]) == []
    assert tmp_repo.get_by_ids("source", ["same_text"]) == []
    assert tmp_repo.get_by_ids("source", ["conflicted"])[0].text == "source side of conflict"
    assert tmp_repo.get_by_ids("target", ["clean"])[0].text == "clean move entry"


def test_move_same_collection_raises_and_does_not_delete(tmp_repo):
    _seed(tmp_repo, "source", {"a": "entry text"})

    move = MoveMemoryUseCase(tmp_repo)
    with pytest.raises(ValueError):
        move.execute("source", "source", ["a"])

    assert tmp_repo.get_by_ids("source", ["a"])[0].text == "entry text"


def test_move_duplicate_ids_are_deduped(tmp_repo):
    _seed(tmp_repo, "source", {"a": "entry text"})

    move = MoveMemoryUseCase(tmp_repo)
    result = move.execute("source", "target", ["a", "a"])

    assert result.moved_ids == ["a"]
    assert tmp_repo.get_by_ids("target", ["a"])[0].text == "entry text"


def test_move_all_conflicts_does_not_touch_source(tmp_repo):
    _seed(tmp_repo, "source", {"a": "source text", "b": "source text b"})
    _seed(tmp_repo, "target", {"a": "different target text", "b": "different target text b"})

    move = MoveMemoryUseCase(tmp_repo)
    result = move.execute("source", "target", ["a", "b"])

    assert result.moved_ids == []
    assert len(result.conflicts) == 2
    assert tmp_repo.get_by_ids("source", ["a", "b"]) != []
    assert len(tmp_repo.get_by_ids("source", ["a", "b"])) == 2


def test_move_empty_ids_does_not_drop_source_collection(tmp_repo):
    _seed(tmp_repo, "source", {"a": "entry text"})

    move = MoveMemoryUseCase(tmp_repo)
    result = move.execute("source", "target", [])

    assert result.moved_ids == []
    assert result.not_found_ids == []
    assert result.conflicts == []
    assert tmp_repo.get_by_ids("source", ["a"])[0].text == "entry text"
