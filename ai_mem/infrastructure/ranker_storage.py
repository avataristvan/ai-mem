"""Owns filesystem paths and JSONL training-buffer I/O for learned rankers."""
from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path

from ai_mem.domain.learning import RankingFeatures, TrainingExample

_DEFAULT_BASE = Path(os.environ.get("AI_MEM_PATH", Path.home() / ".local" / "share" / "ai-mem")) / "rankers"


class RankerStorage:
    def __init__(self, base_path: Path = _DEFAULT_BASE) -> None:
        self._base = base_path

    def weights_path(self, scope: str) -> Path:
        return self._base / f"{scope}.pt"

    def buffer_path(self, scope: str) -> Path:
        return self._base / f"{scope}.examples.jsonl"

    def append_examples(self, scope: str, examples: list[TrainingExample]) -> None:
        path = self.buffer_path(scope)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(_serialize(ex)) + "\n")

    def load_buffer(self, scope: str) -> list[TrainingExample]:
        path = self.buffer_path(scope)
        if not path.exists():
            return []
        examples: list[TrainingExample] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    examples.append(_deserialize(json.loads(line)))
                except Exception as exc:
                    print(
                        f"[ai-mem] ranker_storage: skipping corrupt buffer line in {path}: {exc!r}",
                        file=sys.stderr,
                    )
        return examples

    def prune_buffer(self, scope: str, keep_last: int) -> None:
        path = self.buffer_path(scope)
        if not path.exists():
            return
        examples = self.load_buffer(scope)
        if len(examples) <= keep_last:
            return
        # Keep labeled examples first (they carry training signal not yet used),
        # then unlabeled by recency. This prevents evicting fresh labeled examples
        # in favour of old uncertain ones (the old FIFO strategy's failure mode).
        labeled = sorted(
            [e for e in examples if e.target_future_access is not None],
            key=lambda e: e.retrieved_at,
            reverse=True,
        )
        unlabeled = sorted(
            [e for e in examples if e.target_future_access is None],
            key=lambda e: e.retrieved_at,
            reverse=True,
        )
        kept = (labeled + unlabeled)[:keep_last]
        kept.sort(key=lambda e: e.retrieved_at)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for ex in kept:
                f.write(json.dumps(_serialize(ex)) + "\n")

    def clear_buffer(self, scope: str) -> None:
        path = self.buffer_path(scope)
        if path.exists():
            path.write_text("", encoding="utf-8")


def _serialize(ex: TrainingExample) -> dict:
    return {
        "memory_id": ex.memory_id,
        "features": dataclasses.asdict(ex.features),
        "retrieved_at": ex.retrieved_at,
        "co_activated_ids": ex.co_activated_ids,
        "source_collection": ex.source_collection,
        "target_future_access": ex.target_future_access,
    }


def _deserialize(d: dict) -> TrainingExample:
    return TrainingExample(
        memory_id=d["memory_id"],
        features=RankingFeatures(**d["features"]),
        retrieved_at=d["retrieved_at"],
        co_activated_ids=d.get("co_activated_ids", []),
        source_collection=d.get("source_collection"),
        target_future_access=d.get("target_future_access"),
    )
