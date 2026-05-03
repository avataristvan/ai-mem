"""ListEntriesUseCase: basic listing and title extraction."""
from __future__ import annotations

from ai_mem.application.add_memory import AddMemoryUseCase
from ai_mem.application.list_entries import ListEntriesUseCase


def test_empty_collection_returns_empty(tmp_repo):
    result = ListEntriesUseCase(tmp_repo).execute("empty")
    assert result == []


def test_returns_id_and_title_for_each_entry(tmp_repo):
    AddMemoryUseCase(tmp_repo).execute(
        collection="col",
        documents=["First line\nSecond line", "Another entry"],
        ids=["a", "b"],
    )
    result = ListEntriesUseCase(tmp_repo).execute("col")
    by_id = {r["id"]: r["title"] for r in result}
    assert by_id["a"] == "First line"
    assert by_id["b"] == "Another entry"


def test_title_truncated_to_80_chars(tmp_repo):
    long_line = "x" * 100
    AddMemoryUseCase(tmp_repo).execute(
        collection="col",
        documents=[long_line],
        ids=["long"],
    )
    result = ListEntriesUseCase(tmp_repo).execute("col")
    assert len(result[0]["title"]) == 80


def test_skips_blank_leading_lines(tmp_repo):
    AddMemoryUseCase(tmp_repo).execute(
        collection="col",
        documents=["\n\n  \nActual content"],
        ids=["blank"],
    )
    result = ListEntriesUseCase(tmp_repo).execute("col")
    assert result[0]["title"] == "Actual content"


def test_returns_all_entries(tmp_repo):
    AddMemoryUseCase(tmp_repo).execute(
        collection="col",
        documents=[f"Entry {i}" for i in range(5)],
        ids=[str(i) for i in range(5)],
    )
    result = ListEntriesUseCase(tmp_repo).execute("col")
    assert len(result) == 5
