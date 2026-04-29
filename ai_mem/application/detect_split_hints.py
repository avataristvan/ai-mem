"""Detect memory entries that have grown too coarse and should be split."""
from __future__ import annotations

from ai_mem.domain.memory import QueryResult, SplitHint

SPLIT_THRESHOLD_ACCESSES = 5
SPLIT_MIN_TEXT_CHARS = 150


class DetectSplitHintsUseCase:
    def execute(self, results: list[QueryResult]) -> list[SplitHint]:
        hints = []
        for r in results:
            access_count = int(r.metadata.get("access_count", 0))
            if access_count >= SPLIT_THRESHOLD_ACCESSES and len(r.text) >= SPLIT_MIN_TEXT_CHARS:
                hints.append(SplitHint(
                    id=r.id,
                    text_preview=r.text[:80],
                    access_count=access_count,
                ))
        return hints
