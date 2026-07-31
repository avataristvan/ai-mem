#!/usr/bin/env python3
"""PostToolUse hook for Write|Edit — updates access_count in ai-mem.

After a file is written or edited, the file path is used as a query signal.
Matching memory entries get their last_accessed_at updated (via TrackAccessUseCase
inside QueryMemoryUseCase), which feeds the confidence lifecycle.

No output is produced — the hook is fully silent.
"""
import json
import os
import sys
from pathlib import Path

from ai_mem.agent_context import _CONFIG_PATH, _load_settings
from ai_mem.repo_context import GLOBAL_COLLECTION, detect_repo_context

DB_PATH = Path(os.environ.get("AI_MEM_PATH", Path.home() / ".local" / "share" / "ai-mem"))
_QUERY_K = 5  # enough candidates to surface relevant entries; no injection needed
# Only propagate access credit when the file-path query hits a semantically strong
# match. Weak matches stay unlabeled — they rely on the 7-day access window instead.
_MIN_LABEL_SCORE = 0.50


def _build_deps():
    from ai_mem._hook_deps import _build_query_uc
    return _build_query_uc(DB_PATH)


def _signal_collection(
    query_uc, collection: str, query: str,
    min_label_score: float, query_k: int = _QUERY_K,
) -> None:
    """Query a collection, updating last_accessed_at for strong matches."""
    try:
        query_uc.execute(
            collection=collection,
            query=query,
            n_results=query_k,
            min_score_for_tracking=min_label_score,
        )
    except Exception:
        pass


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return

    if payload.get("tool_name") not in ("Write", "Edit"):
        return

    file_path = payload.get("tool_input", {}).get("file_path", "")
    if not file_path or not DB_PATH.exists():
        return

    settings = _load_settings(_CONFIG_PATH)
    min_label_score = float(settings.get("min_label_score", _MIN_LABEL_SCORE))
    query_k = int(settings.get("query_k", _QUERY_K))

    try:
        query_uc = _build_deps()
    except Exception:
        return

    file_name = Path(file_path).name
    query = f"{file_name} {file_path}"

    _signal_collection(query_uc, GLOBAL_COLLECTION, query, min_label_score, query_k)

    try:
        ctx = detect_repo_context()
        if ctx.has_claude_md:
            _signal_collection(query_uc, ctx.collection, query, min_label_score, query_k)
    except Exception:
        pass


if __name__ == "__main__":
    main()
