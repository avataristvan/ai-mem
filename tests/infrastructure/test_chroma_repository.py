"""Unit tests for ChromaMemoryRepository.query() score computation.

Collections are created via get_or_create_collection() with no explicit space
metadata, which defaults to ChromaDB's `l2` (squared L2) space, not cosine.
For normalized embeddings, dist = 2 - 2*cos_sim, so cos_sim = 1 - dist/2.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _mock_query_result(distances: list[float]) -> dict:
    n = len(distances)
    return {
        "documents": [[f"doc{i}" for i in range(n)]],
        "ids": [[f"id{i}" for i in range(n)]],
        "metadatas": [[{} for _ in range(n)]],
        "distances": [distances],
    }


def _mock_collection(distances: list[float]) -> MagicMock:
    col = MagicMock()
    col.count.return_value = len(distances)
    col.query.return_value = _mock_query_result(distances)
    return col


def test_query_score_identical_vectors_dist_zero(tmp_repo):
    tmp_repo._col = MagicMock(return_value=_mock_collection([0.0]))

    results = tmp_repo.query("col", "text", n_results=1, max_age_days=None)

    assert results[0].score == 1.0


def test_query_score_orthogonal_vectors_dist_two(tmp_repo):
    tmp_repo._col = MagicMock(return_value=_mock_collection([2.0]))

    results = tmp_repo.query("col", "text", n_results=1, max_age_days=None)

    assert results[0].score == 0.0


def test_query_score_matches_verified_real_world_distance(tmp_repo):
    tmp_repo._col = MagicMock(return_value=_mock_collection([1.758348822593689]))

    results = tmp_repo.query("col", "text", n_results=1, max_age_days=None)

    assert results[0].score == pytest.approx(0.1208, abs=1e-4)
