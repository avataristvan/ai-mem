"""Infrastructure-level tests for add_edge / get_edges in ChromaMemoryRepository."""
from __future__ import annotations

from ai_mem.application.add_memory import AddMemoryUseCase
from ai_mem.domain.memory import MemoryEdge


def _seed(repo, collection: str, docs: dict[str, str]) -> None:
    AddMemoryUseCase(repo).execute(
        collection=collection,
        documents=list(docs.values()),
        ids=list(docs.keys()),
    )


def test_add_edge_and_get_edges_roundtrip(tmp_repo):
    _seed(tmp_repo, "col", {"src": "source text", "tgt": "target text"})

    edge = MemoryEdge(target_id="tgt", edge_type="fixes")
    tmp_repo.add_edge("col", "src", edge)

    edges = tmp_repo.get_edges("col", "src")
    assert len(edges) == 1
    assert edges[0].target_id == "tgt"
    assert edges[0].edge_type == "fixes"


def test_add_edge_idempotent(tmp_repo):
    _seed(tmp_repo, "col", {"src": "source", "tgt": "target"})
    edge = MemoryEdge(target_id="tgt", edge_type="related")
    tmp_repo.add_edge("col", "src", edge)
    tmp_repo.add_edge("col", "src", edge)

    edges = tmp_repo.get_edges("col", "src")
    assert len(edges) == 1


def test_multiple_edges_on_same_entry(tmp_repo):
    _seed(tmp_repo, "col", {"src": "source", "t1": "target 1", "t2": "target 2"})
    tmp_repo.add_edge("col", "src", MemoryEdge(target_id="t1", edge_type="causes"))
    tmp_repo.add_edge("col", "src", MemoryEdge(target_id="t2", edge_type="related"))

    edges = tmp_repo.get_edges("col", "src")
    assert len(edges) == 2
    ids = {e.target_id for e in edges}
    assert ids == {"t1", "t2"}


def test_get_edges_returns_empty_for_missing_entry(tmp_repo):
    edges = tmp_repo.get_edges("col", "ghost")
    assert edges == []


def test_get_edges_returns_empty_for_no_edges(tmp_repo):
    _seed(tmp_repo, "col", {"a": "no edges"})
    edges = tmp_repo.get_edges("col", "a")
    assert edges == []


def test_add_edge_silently_ignores_missing_source(tmp_repo):
    """add_edge on a nonexistent source should be silent (no crash, no side effect)."""
    _seed(tmp_repo, "col", {"tgt": "target"})
    tmp_repo.add_edge("col", "ghost", MemoryEdge(target_id="tgt", edge_type="related"))
    # target should have no edges pointing from it
    edges = tmp_repo.get_edges("col", "ghost")
    assert edges == []
