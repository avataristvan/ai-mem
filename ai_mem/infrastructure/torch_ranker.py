"""PyTorch MLP re-ranker. torch is an optional dependency — import guard at class init."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from ai_mem.domain.learning import RankingFeatures, TrainingExample, TrainingMetrics

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

try:
    import torch as _torch
    import torch.nn as _nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class TorchMicroRanker:
    """[11 -> 32 -> 16 -> 1] MLP with sigmoid output. AdamW lr=1e-3."""

    def __init__(self, seed: int | None = None) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "torch is required for TorchMicroRanker. "
                "Install it with: pip install 'ai-mem[ml]'"
            )
        if seed is not None:
            _torch.manual_seed(seed)
        self._model = _nn.Sequential(
            _nn.Linear(11, 32),
            _nn.ReLU(),
            _nn.Linear(32, 16),
            _nn.ReLU(),
            _nn.Linear(16, 1),
            _nn.Sigmoid(),
        )
        self._optimizer = _torch.optim.AdamW(self._model.parameters(), lr=1e-3)

    def rank(self, features: list[RankingFeatures]) -> list[float]:
        if not features:
            return []
        x = _torch.tensor([f.as_vector() for f in features], dtype=_torch.float32)
        with _torch.no_grad():
            scores = self._model(x).squeeze(-1)
        return scores.cpu().tolist()

    def train_step(self, examples: list[TrainingExample]) -> TrainingMetrics:
        labeled = [e for e in examples if e.target_future_access is not None]
        if not labeled:
            return TrainingMetrics(n=0, skipped=True)

        self._model.train()

        x = _torch.tensor([e.features.as_vector() for e in labeled], dtype=_torch.float32)
        targets = _torch.tensor([e.target_future_access for e in labeled], dtype=_torch.float32)
        scores = self._model(x).squeeze(-1)

        bce_loss = _nn.functional.binary_cross_entropy(scores, targets)
        contrastive_loss = _compute_contrastive(labeled, scores)
        total_loss = bce_loss + 0.3 * contrastive_loss

        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()

        return TrainingMetrics(n=len(labeled), loss=float(total_loss.detach()))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        _torch.save(self._model.state_dict(), path)

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            state = _torch.load(path, map_location="cpu", weights_only=True)
            self._model.load_state_dict(state)
            self._model.eval()
        except Exception as exc:
            # Shape mismatch (e.g. MLP topology changed across versions) or
            # corrupt file. Stay with the fresh init and surface the issue.
            print(
                f"[ai-mem] torch_ranker: ignoring incompatible weights at {path}: {exc!r}",
                file=sys.stderr,
            )


def _compute_contrastive(labeled: list[TrainingExample], scores: "torch.Tensor") -> "torch.Tensor":
    margin = 0.1
    id_to_idx = {e.memory_id: i for i, e in enumerate(labeled)}
    accumulator = _torch.tensor(0.0)
    pairs_found = False

    for i, ex in enumerate(labeled):
        co_indices = [id_to_idx[id_] for id_ in ex.co_activated_ids if id_ in id_to_idx]
        non_co_indices = [
            j for j, e2 in enumerate(labeled)
            if e2.memory_id != ex.memory_id and e2.memory_id not in ex.co_activated_ids
        ]
        if not co_indices or not non_co_indices:
            continue

        pairs_found = True
        k = non_co_indices[0]
        for j in co_indices:
            pair_loss = (
                _torch.clamp(scores[k] - scores[j] + margin, min=0.0) +
                _torch.clamp(scores[k] - scores[i] + margin, min=0.0)
            )
            accumulator = accumulator + pair_loss

    if not pairs_found:
        return _torch.tensor(0.0)
    return accumulator
