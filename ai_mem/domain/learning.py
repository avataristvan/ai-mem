"""Domain layer for adaptive ranking — contracts only, no I/O, no torch."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol


_NO_TTL_HORIZON_DAYS = 365.0  # placeholder horizon for entries without TTL


@dataclass(frozen=True)
class RankingFeatures:
    """Raw features for a candidate memory at query time.

    Frozen + raw-only: log transforms and the never-accessed flag are derived
    in `as_vector()` so the four-way consistency between raw and derived values
    is structurally impossible to violate.
    """
    cosine_similarity: float        # 0..1 (1 = identical, populated from ChromaDB score = 1 - distance)
    age_days: float                 # now - created_at
    last_access_days: float         # falls back to age_days if never accessed
    access_count: int
    has_ttl: bool
    expires_in_days: float          # only meaningful when has_ttl is True

    @property
    def is_never_accessed(self) -> bool:
        return self.access_count == 0

    def as_vector(self) -> list[float]:
        ttl_horizon = self.expires_in_days if self.has_ttl else _NO_TTL_HORIZON_DAYS
        return [
            self.cosine_similarity,
            self.age_days,
            self.last_access_days,
            float(self.access_count),
            1.0 if self.is_never_accessed else 0.0,
            1.0 if self.has_ttl else 0.0,
            ttl_horizon,
            math.log1p(max(0.0, self.age_days)),
            math.log1p(max(0.0, self.last_access_days)),
            math.log1p(max(0.0, float(self.access_count))),
        ]


@dataclass
class TrainingExample:
    """One labeled (features, target) pair.

    `target_future_access` is None at retrieval time; the trainer fills it
    in (0.0 or 1.0) once the label window has elapsed.
    `source_collection` records which collection this example came from;
    used in hybrid mode to look up labels from the correct collection.
    """
    memory_id: str
    features: RankingFeatures
    retrieved_at: float                                     # Unix timestamp
    co_activated_ids: list[str] = field(default_factory=list)
    source_collection: str | None = None
    target_future_access: float | None = None

    def __post_init__(self) -> None:
        t = self.target_future_access
        if t is not None and t not in (0.0, 1.0):
            raise ValueError(
                f"target_future_access must be 0.0, 1.0, or None — got {t!r}"
            )


@dataclass
class TrainingMetrics:
    """Result of one `train_step` call."""
    n: int
    loss: float | None = None
    skipped: bool = False


@dataclass
class RankerScope:
    """Routing config: does a collection have its own NN, or share with a group?"""
    name: str
    mode: Literal["isolated", "hybrid"]
    group: str | None = None
    member_collections: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mode == "hybrid" and not self.group:
            raise ValueError(
                f"RankerScope mode='hybrid' requires a non-empty group (collection {self.name!r})"
            )
        if self.mode == "isolated" and (self.group or self.member_collections):
            raise ValueError(
                f"RankerScope mode='isolated' must not specify group or member_collections "
                f"(collection {self.name!r})"
            )


class LearnedRanker(Protocol):
    """A small re-ranker that learns from access patterns.

    Implementations may be torch-based, numpy-based, or future NPU-backed.
    """

    def rank(self, features: list[RankingFeatures]) -> list[float]:
        """Score candidates. Higher = more relevant. Length matches input."""
        ...

    def train_step(self, examples: list[TrainingExample]) -> TrainingMetrics:
        """One gradient step on a labeled batch.

        Empty / unlabeled batches return TrainingMetrics(n=0, skipped=True).
        """
        ...

    def save(self, path: Path) -> None:
        """Persist weights to disk. Creates parent directories as needed."""
        ...

    def load(self, path: Path) -> None:
        """Load weights from disk. No-op (keep current init) if file missing."""
        ...


class TrainingBufferRepository(Protocol):
    """Persistence contract for the training buffer and ranker weights.

    Used by TrainRankerUseCase. Keeps the application layer free of a
    concrete infrastructure dependency on RankerStorage.
    """

    def append_examples(self, scope: str, examples: list[TrainingExample]) -> None: ...
    def load_buffer(self, scope: str) -> list[TrainingExample]: ...
    def clear_buffer(self, scope: str) -> None: ...
    def prune_buffer(self, scope: str, keep_last: int) -> None: ...
    def weights_path(self, scope: str) -> Path: ...


class RankerProvider(Protocol):
    """Resolves a collection name to its loaded LearnedRanker instance."""

    def get(self, collection: str) -> LearnedRanker: ...
