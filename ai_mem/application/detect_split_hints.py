"""Detect memory entries that have grown too coarse and should be split."""
from __future__ import annotations

import re

from ai_mem.domain.memory import QueryResult, SplitHint

SPLIT_THRESHOLD_ACCESSES = 5
SPLIT_MIN_TEXT_CHARS = 500  # plain length fallback (was 150 — too noisy)

_MULTI_HEADER_RE = re.compile(r"^#{1,3}\s+\S", re.MULTILINE)
_TRIED_RE = re.compile(r"^Tried:", re.MULTILINE)
_NUMBERED_RE = re.compile(r"^\d+[.)]\s+\S", re.MULTILINE)


def _is_multi_section(text: str) -> bool:
    """True when text carries structural markers suggesting multiple independent concepts."""
    if len(_MULTI_HEADER_RE.findall(text)) >= 2:
        return True
    if len(_TRIED_RE.findall(text)) >= 2:
        return True
    if len(_NUMBERED_RE.findall(text)) >= 4:
        return True
    return False


class DetectSplitHintsUseCase:
    def execute(self, results: list[QueryResult]) -> list[SplitHint]:
        hints = []
        for r in results:
            access_count = int(r.metadata.get("access_count", 0))
            if access_count < SPLIT_THRESHOLD_ACCESSES:
                continue
            if _is_multi_section(r.text) or len(r.text) >= SPLIT_MIN_TEXT_CHARS:
                hints.append(SplitHint(
                    id=r.id,
                    text_preview=r.text[:80],
                    access_count=access_count,
                ))
        return hints
