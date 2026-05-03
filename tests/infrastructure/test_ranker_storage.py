"""RankerStorage: JSONL buffer append, load, prune, clear."""
from __future__ import annotations

from pathlib import Path

import pytest

from ai_mem.domain.learning import RankingFeatures, TrainingExample
from ai_mem.infrastructure.ranker_storage import RankerStorage


@pytest.fixture
def storage(tmp_path: Path) -> RankerStorage:
    return RankerStorage(tmp_path / "rankers")


def _example(memory_id: str, target: float | None = None, retrieved_at: float = 1000.0) -> TrainingExample:
    feats = RankingFeatures(
        cosine_similarity=0.1,
        age_days=1.0,
        last_access_days=1.0,
        access_count=2,
        has_ttl=False,
        expires_in_days=0.0,
    )
    return TrainingExample(
        memory_id=memory_id,
        features=feats,
        retrieved_at=retrieved_at,
        co_activated_ids=["other"],
        target_future_access=target,
    )


def test_append_and_load_roundtrip(storage: RankerStorage):
    ex1 = _example("a")
    ex2 = _example("b", target=1.0)
    storage.append_examples("col1", [ex1, ex2])

    loaded = storage.load_buffer("col1")
    assert len(loaded) == 2
    assert loaded[0].memory_id == "a"
    assert loaded[0].target_future_access is None
    assert loaded[1].memory_id == "b"
    assert loaded[1].target_future_access == 1.0


def test_load_missing_file_returns_empty(storage: RankerStorage):
    assert storage.load_buffer("nonexistent") == []


def test_prune_keeps_newest_unlabeled(storage: RankerStorage):
    examples = [_example(str(i), retrieved_at=float(i)) for i in range(10)]
    storage.append_examples("col", examples)
    storage.prune_buffer("col", keep_last=3)
    loaded = storage.load_buffer("col")
    assert len(loaded) == 3
    assert {e.memory_id for e in loaded} == {"7", "8", "9"}


def test_prune_keeps_labeled_over_older_unlabeled(storage: RankerStorage):
    """Labeled examples survive pruning even when they are older than newer unlabeled ones."""
    labeled = _example("labeled", target=1.0, retrieved_at=1.0)
    unlabeled = [_example(str(i), retrieved_at=float(i + 2)) for i in range(9)]
    storage.append_examples("col", [labeled] + unlabeled)
    storage.prune_buffer("col", keep_last=3)
    loaded = storage.load_buffer("col")
    assert len(loaded) == 3
    assert any(e.memory_id == "labeled" for e in loaded)


def test_clear_empties_buffer(storage: RankerStorage):
    storage.append_examples("col", [_example("x")])
    storage.clear_buffer("col")
    assert storage.load_buffer("col") == []


def test_weights_path_and_buffer_path(storage: RankerStorage, tmp_path: Path):
    base = tmp_path / "rankers"
    assert storage.weights_path("mycol") == base / "mycol.pt"
    assert storage.buffer_path("mycol") == base / "mycol.examples.jsonl"
