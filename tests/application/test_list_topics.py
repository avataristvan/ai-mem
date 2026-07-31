"""ListTopicsUseCase: grouping entries by type for browsing at scale."""
from __future__ import annotations

from ai_mem.application.add_memory import AddMemoryUseCase
from ai_mem.application.list_topics import ListTopicsUseCase


def test_empty_collection_returns_empty(tmp_repo):
    result = ListTopicsUseCase(tmp_repo).execute("empty")
    assert result == []


def test_groups_entries_by_type(tmp_repo):
    AddMemoryUseCase(tmp_repo).execute(
        collection="col",
        documents=["A pattern entry", "An anti-pattern entry", "Another pattern entry"],
        ids=["p1", "ap1", "p2"],
        metadatas=[{"type": "pattern"}, {"type": "anti-pattern"}, {"type": "pattern"}],
    )
    result = ListTopicsUseCase(tmp_repo).execute("col")
    by_type = {topic["type"]: topic for topic in result}

    assert by_type["pattern"]["count"] == 2
    assert {e["id"] for e in by_type["pattern"]["entries"]} == {"p1", "p2"}
    assert by_type["anti-pattern"]["count"] == 1
    assert by_type["anti-pattern"]["entries"][0]["id"] == "ap1"


def test_entries_without_type_grouped_as_untyped(tmp_repo):
    AddMemoryUseCase(tmp_repo).execute(collection="col", documents=["No type set"], ids=["u1"])
    result = ListTopicsUseCase(tmp_repo).execute("col")
    assert len(result) == 1
    assert result[0]["type"] == "untyped"
    assert result[0]["entries"][0]["id"] == "u1"


def test_topics_sorted_by_count_desc_then_alphabetically(tmp_repo):
    AddMemoryUseCase(tmp_repo).execute(
        collection="col",
        documents=["f1", "p1", "p2", "p3"],
        ids=["f1", "p1", "p2", "p3"],
        metadatas=[{"type": "fact"}, {"type": "pattern"}, {"type": "pattern"}, {"type": "pattern"}],
    )
    result = ListTopicsUseCase(tmp_repo).execute("col")
    assert [topic["type"] for topic in result] == ["pattern", "fact"]


def test_entry_titles_use_title_of_truncation(tmp_repo):
    long_line = "x" * 100
    AddMemoryUseCase(tmp_repo).execute(
        collection="col", documents=[long_line], ids=["long"], metadatas=[{"type": "fact"}],
    )
    result = ListTopicsUseCase(tmp_repo).execute("col")
    assert len(result[0]["entries"][0]["title"]) == 80
