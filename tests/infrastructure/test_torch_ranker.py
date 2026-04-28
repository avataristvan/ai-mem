"""TorchMicroRanker: rank shape, save/load roundtrip, train_step behavior."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ai_mem.domain.learning import RankingFeatures, TrainingExample, TrainingMetrics
from ai_mem.infrastructure.torch_ranker import TorchMicroRanker


def _features(cosine_similarity: float = 0.2) -> RankingFeatures:
    return RankingFeatures(
        cosine_similarity=cosine_similarity,
        age_days=1.0,
        last_access_days=1.0,
        access_count=1,
        has_ttl=False,
        expires_in_days=0.0,
    )


def _example(memory_id: str, target: float | None) -> TrainingExample:
    return TrainingExample(
        memory_id=memory_id,
        features=_features(),
        retrieved_at=1000.0,
        co_activated_ids=[],
        target_future_access=target,
    )


def test_rank_returns_correct_length():
    ranker = TorchMicroRanker()
    feats = [_features(0.1), _features(0.5), _features(0.9)]
    scores = ranker.rank(feats)
    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_rank_empty_input():
    ranker = TorchMicroRanker()
    assert ranker.rank([]) == []


def test_train_step_empty_returns_skipped():
    ranker = TorchMicroRanker()
    metrics = ranker.train_step([])
    assert metrics.skipped is True
    assert metrics.n == 0


def test_train_step_unlabeled_only_returns_skipped():
    ranker = TorchMicroRanker()
    examples = [_example("a", None), _example("b", None)]
    metrics = ranker.train_step(examples)
    assert metrics.skipped is True


def test_train_step_labeled_returns_finite_loss():
    ranker = TorchMicroRanker()
    examples = [_example("a", 1.0), _example("b", 0.0)]
    metrics = ranker.train_step(examples)
    assert metrics.skipped is False
    assert metrics.n == 2
    assert metrics.loss is not None
    assert math.isfinite(metrics.loss)


def test_save_and_load_preserves_weights(tmp_path: Path):
    ranker = TorchMicroRanker()
    feats = [_features()]
    scores_before = ranker.rank(feats)

    path = tmp_path / "weights.pt"
    ranker.save(path)
    assert path.exists()

    ranker2 = TorchMicroRanker()
    ranker2.load(path)
    scores_after = ranker2.rank(feats)
    assert abs(scores_before[0] - scores_after[0]) < 1e-6


def test_load_missing_file_is_noop(tmp_path: Path):
    ranker = TorchMicroRanker()
    ranker.load(tmp_path / "nonexistent.pt")
    assert ranker.rank([_features()]) is not None
