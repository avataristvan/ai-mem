"""Tests for AddEdgeUseCase, GetEdgesUseCase, and QueryMemoryUseCase edge-follow."""
from __future__ import annotations

from pathlib import Path

import pytest

from ai_mem.application.add_edge import AddEdgeUseCase
from ai_mem.application.add_memory import AddMemoryUseCase
from ai_mem.application.build_features import BuildFeaturesUseCase
from ai_mem.application.get_edges import GetEdgesUseCase
from ai_mem.application.query_memory import QueryMemoryUseCase
from ai_mem.application.ranker_registry import RankerRegistry
from ai_mem.application.track_access import TrackAccessUseCase
from ai_mem.application.train_ranker import TrainRankerUseCase
from ai_mem.domain.learning import RankerScope
from ai_mem.infrastructure.null_ranker import NullRanker
from ai_mem.infrastructure.ranker_storage import RankerStorage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed(repo, collection: str, docs: dict[str, str]) -> None:
    """Add entries: {id: text}."""
    add = AddMemoryUseCase(repo)
    add.execute(
        collection=collection,
        documents=list(docs.values()),
        ids=list(docs.keys()),
    )


def _make_query_uc(tmp_repo, tmp_path: Path) -> QueryMemoryUseCase:
    storage = RankerStorage(tmp_path / "rankers")
    train_ranker = TrainRankerUseCase(tmp_repo, storage, NullRanker)
    registry = RankerRegistry(
        scope_resolver=lambda c: RankerScope(name=c, mode="isolated"),
        ranker_factory=NullRanker,
        storage=storage,
    )
    return QueryMemoryUseCase(
        repo=tmp_repo,
        track_access=TrackAccessUseCase(tmp_repo),
        build_features=BuildFeaturesUseCase(),
        train_ranker=train_ranker,
        ranker_provider=registry,
    )


# ---------------------------------------------------------------------------
# AddEdgeUseCase
# ---------------------------------------------------------------------------

def test_add_edge_happy_path(tmp_repo, tmp_path):
    _seed(tmp_repo, "col", {"a": "pattern: always do X", "b": "anti-pattern: never do Y"})
    add_edge = AddEdgeUseCase(tmp_repo)
    add_edge.execute("col", "b", "a", "contradicts")

    get_edges = GetEdgesUseCase(tmp_repo)
    edges = get_edges.execute("col", "b")
    assert len(edges) == 1
    assert edges[0].target_id == "a"
    assert edges[0].edge_type == "contradicts"


def test_add_edge_deduplication(tmp_repo, tmp_path):
    _seed(tmp_repo, "col", {"a": "pattern text", "b": "anti text"})
    add_edge = AddEdgeUseCase(tmp_repo)
    add_edge.execute("col", "b", "a", "contradicts")
    add_edge.execute("col", "b", "a", "contradicts")  # duplicate

    edges = GetEdgesUseCase(tmp_repo).execute("col", "b")
    assert len(edges) == 1


def test_add_edge_different_types_not_deduped(tmp_repo, tmp_path):
    _seed(tmp_repo, "col", {"a": "target", "b": "source"})
    add_edge = AddEdgeUseCase(tmp_repo)
    add_edge.execute("col", "b", "a", "contradicts")
    add_edge.execute("col", "b", "a", "related")

    edges = GetEdgesUseCase(tmp_repo).execute("col", "b")
    assert len(edges) == 2
    types = {e.edge_type for e in edges}
    assert types == {"contradicts", "related"}


def test_add_edge_raises_when_source_missing(tmp_repo, tmp_path):
    _seed(tmp_repo, "col", {"a": "target"})
    add_edge = AddEdgeUseCase(tmp_repo)
    with pytest.raises(ValueError, match="not found"):
        add_edge.execute("col", "nonexistent", "a", "related")


def test_add_edge_raises_when_target_missing(tmp_repo, tmp_path):
    _seed(tmp_repo, "col", {"b": "source"})
    add_edge = AddEdgeUseCase(tmp_repo)
    with pytest.raises(ValueError, match="not found"):
        add_edge.execute("col", "b", "nonexistent", "related")


# ---------------------------------------------------------------------------
# GetEdgesUseCase
# ---------------------------------------------------------------------------

def test_get_edges_empty_when_no_edges(tmp_repo, tmp_path):
    _seed(tmp_repo, "col", {"a": "some text"})
    edges = GetEdgesUseCase(tmp_repo).execute("col", "a")
    assert edges == []


def test_get_edges_empty_for_missing_entry(tmp_repo, tmp_path):
    edges = GetEdgesUseCase(tmp_repo).execute("col", "ghost")
    assert edges == []


# ---------------------------------------------------------------------------
# QueryMemoryUseCase — edge follow
# ---------------------------------------------------------------------------

def test_query_appends_linked_entry(tmp_repo, tmp_path):
    # "antipat1" has strong semantic overlap with the query "skip validation".
    # "linked_target" has no semantic overlap and would not appear in top-k on its own.
    # After adding an edge antipat1 → linked_target, it should appear as a via_edge result.
    _seed(tmp_repo, "col", {
        "antipat1": "skip validation because it slows things down",
        "linked_target": "ZZZQQQXXX completely unrelated boilerplate content here",
    })
    AddEdgeUseCase(tmp_repo).execute("col", "antipat1", "linked_target", "contradicts")

    query_uc = _make_query_uc(tmp_repo, tmp_path)
    # n_results=1 so only "antipat1" is in the primary result set
    results = query_uc.execute("col", "skip validation", n_results=1)

    ids = [r.id for r in results]
    assert "antipat1" in ids

    # The linked entry should be appended via edge traversal
    linked = [r for r in results if r.metadata.get("via_edge") == "contradicts"]
    assert len(linked) == 1
    assert linked[0].id == "linked_target"
    assert linked[0].metadata["via_source"] == "antipat1"


def test_query_linked_entry_not_duplicated_when_already_in_results(tmp_repo, tmp_path):
    """If the linked target is already in the result set, it should not appear twice."""
    _seed(tmp_repo, "col", {
        "a": "validate user input",
        "b": "skip validation is an anti-pattern",
    })
    AddEdgeUseCase(tmp_repo).execute("col", "b", "a", "contradicts")

    query_uc = _make_query_uc(tmp_repo, tmp_path)
    # Query broad enough to retrieve both entries directly
    results = query_uc.execute("col", "validate input", n_results=5)

    ids = [r.id for r in results]
    # 'a' should appear exactly once, whether as direct result or via edge
    assert ids.count("a") == 1


def test_query_edge_budget_caps_at_two(tmp_repo, tmp_path):
    """At most _EDGE_BUDGET=2 linked entries are appended per query."""
    docs = {"src": "source entry about X"}
    for i in range(5):
        docs[f"target{i}"] = f"linked target number {i}"
    _seed(tmp_repo, "col", docs)

    add_edge = AddEdgeUseCase(tmp_repo)
    for i in range(5):
        add_edge.execute("col", "src", f"target{i}", "related")

    query_uc = _make_query_uc(tmp_repo, tmp_path)
    results = query_uc.execute("col", "source entry", n_results=1)

    linked = [r for r in results if r.metadata.get("via_edge")]
    assert len(linked) <= 2


def test_query_does_not_crash_on_corrupt_edges(tmp_repo, tmp_path):
    """A corrupt 'edges' field should be silently ignored, not crash the query."""
    from ai_mem.application.add_memory import AddMemoryUseCase
    AddMemoryUseCase(tmp_repo).execute(
        collection="col",
        documents=["some text"],
        ids=["broken"],
        metadatas=[{"edges": "not valid json {{{{"}],
    )
    query_uc = _make_query_uc(tmp_repo, tmp_path)
    results = query_uc.execute("col", "some text", n_results=5)
    assert any(r.id == "broken" for r in results)
