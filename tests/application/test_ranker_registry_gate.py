"""RankerRegistry: fallback gate — returns NullRanker when labeled examples are below threshold."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from ai_mem.application.ranker_registry import MIN_LABELED_EXAMPLES, RankerRegistry
from ai_mem.domain.learning import RankerScope
from ai_mem.infrastructure.null_ranker import NullRanker


def _make_registry(
    tmp_path: Path,
    labeled: int,
    with_fallback: bool = True,
    min_labeled: int = MIN_LABELED_EXAMPLES,
) -> tuple[RankerRegistry, MagicMock]:
    """Build a registry with a mock primary ranker and controlled labeled_count."""
    storage = MagicMock()
    storage.labeled_count.return_value = labeled
    storage.weights_path.return_value = tmp_path / "weights.pt"

    primary_ranker = MagicMock()
    primary_factory = MagicMock(return_value=primary_ranker)

    registry = RankerRegistry(
        scope_resolver=lambda c: RankerScope(name=c, mode="isolated"),
        ranker_factory=primary_factory,
        storage=storage,
        fallback_factory=NullRanker if with_fallback else None,
        min_labeled=min_labeled,
    )
    return registry, primary_ranker


def test_below_threshold_returns_null_ranker(tmp_path: Path):
    registry, _ = _make_registry(tmp_path, labeled=MIN_LABELED_EXAMPLES - 1)
    result = registry.get("col")
    assert isinstance(result, NullRanker)


def test_at_threshold_returns_primary(tmp_path: Path):
    registry, primary = _make_registry(tmp_path, labeled=MIN_LABELED_EXAMPLES)
    result = registry.get("col")
    assert result is primary


def test_above_threshold_returns_primary(tmp_path: Path):
    registry, primary = _make_registry(tmp_path, labeled=MIN_LABELED_EXAMPLES + 5)
    result = registry.get("col")
    assert result is primary


def test_no_fallback_always_returns_primary(tmp_path: Path):
    registry, primary = _make_registry(tmp_path, labeled=0, with_fallback=False)
    result = registry.get("col")
    assert result is primary


def test_fallback_not_cached_rechecks_each_call(tmp_path: Path):
    """Below threshold: each call gets a fresh NullRanker (not cached)."""
    registry, _ = _make_registry(tmp_path, labeled=0)
    r1 = registry.get("col")
    r2 = registry.get("col")
    assert isinstance(r1, NullRanker)
    assert isinstance(r2, NullRanker)
    assert r1 is not r2  # fresh instance each time


def test_primary_cached_after_threshold(tmp_path: Path):
    """At/above threshold: primary ranker is loaded once and cached."""
    registry, primary = _make_registry(tmp_path, labeled=MIN_LABELED_EXAMPLES)
    r1 = registry.get("col")
    r2 = registry.get("col")
    assert r1 is r2 is primary
