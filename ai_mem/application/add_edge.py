"""Add a typed causal edge between two memory entries."""
from __future__ import annotations

from ai_mem.domain.memory import EdgeType, MemoryEdge, MemoryRepository


class AddEdgeUseCase:
    def __init__(self, repo: MemoryRepository) -> None:
        self._repo = repo

    def execute(
        self,
        collection: str,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
    ) -> None:
        """Link source_id → target_id with the given edge_type.

        Raises ValueError if either entry does not exist in the collection.
        Silently deduplicates if the same (target_id, edge_type) already exists.
        """
        missing = []
        for id_ in (source_id, target_id):
            if not self._repo.get_by_ids(collection, [id_]):
                missing.append(id_)
        if missing:
            raise ValueError(f"Entry ID(s) not found in '{collection}': {missing}")

        self._repo.add_edge(collection, source_id, MemoryEdge(target_id=target_id, edge_type=edge_type))
