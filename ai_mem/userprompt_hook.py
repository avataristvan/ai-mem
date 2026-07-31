#!/usr/bin/env python3
"""UserPromptSubmit hook — injects relevant memories into Claude's context."""
import json
import os
import sys
from pathlib import Path
from typing import Any

from ai_mem.application.list_entries import title_of
from ai_mem.repo_context import GLOBAL_COLLECTION, detect_repo_context

DB_PATH = Path(os.environ.get("AI_MEM_PATH", Path.home() / ".local" / "share" / "ai-mem"))
TOP_K = 3
MAX_TOTAL_CHARS = 1500
# STALE as of 2026-07-31 (later same day): BM25's residual min-max flooring (described below)
# was fixed by replacing it with a saturating transform (bm25_repository.py::_saturate) — the
# measurements and thresholds below predate that fix and reflect the OLD distribution. The
# thresholds have NOT yet been re-measured against the new fused-score distribution; do not
# trust them as calibrated. See ai-mem entry bm25_saturating_transform_research_decision_2026_07_31
# and CLAUDE.md's Critical Invariants for the pending recalibration follow-up.
#
# Thresholds recalibrated 2026-07-31 against the live ~/.local/share/ai-mem DB after the
# Chunk 1/2 scoring fixes (real cosine similarity + non-double-normalized BM25 fusion), using
# _build_query_uc(with_bm25=True) — the same hybrid path this hook uses. Measured top-hit
# scores, unfiltered CONTEXT query:
#   - Clearly irrelevant prompts ("weather in Paris", "pizza dough hydration", "quantum
#     entanglement", ...) against `global` and `repo.ai-mem`: ceiling ~0.60 (worst case 0.6023).
#     This ceiling is not noise-free: BM25's min-max normalization still gives the best-in-pool
#     candidate a bm25_norm of 1.0 regardless of absolute relevance, so alpha*cosine + (1-alpha)*1.0
#     puts a floor under even off-topic top hits.
#   - Verifiably relevant prompts (matching real entries like the ChromaDB scoring fix, BM25
#     fusion fix, confidence lifecycle, hook system docs) against the same collections: floor
#     ~0.58 in the noisier multi-project `global` collection, ~0.71 in the more homogeneous
#     `repo.ai-mem` collection.
# CONTEXT_MIN_SCORE is set just above the measured irrelevant ceiling (0.62), accepting that a
# few marginal true positives in `global` (0.58-0.62) are filtered out — the session that
# produced this recalibration found the old threshold (0.3, tuned for min-max-normalized scores
# that always inflated the top hit toward 1.0) injected near-unconditionally, so precision is
# prioritized over recall here.
CONTEXT_MIN_SCORE = 0.62
MIN_QUERY_CHARS = 15
SESSION_TTL_HOURS = 4
ANTIPATTERN_TOP_K = 2
MAX_CHARS_PER_ANTIPATTERN = 200
# Anti-pattern/dilemma queries use a type_filter, so the BM25 candidate pool is much smaller
# (11 anti-pattern entries in `global`, 2 in `repo.ai-mem` at measurement time; 0 dilemma entries
# exist anywhere, so DILEMMA_MIN_SCORE is set by structural analogy rather than direct
# measurement). Measured irrelevant top-hit ceiling for anti-pattern queries: ~0.555 (0.5537
# global, 0.5553 repo.ai-mem). Measured relevant floor: ~0.60 (global) / ~0.75 (repo.ai-mem).
# Set just above the measured irrelevant ceiling (0.58) to keep the real match through while
# rejecting the noise ceiling.
ANTIPATTERN_MIN_SCORE = 0.58
DILEMMA_TOP_K = 2
MAX_CHARS_PER_DILEMMA = 250
DILEMMA_MIN_SCORE = 0.58
ANTICIPATION_QUESTION = "  → Anticipation: Who holds the same role now? What would happen to them?"


def _load_session_injected(db_path: Path) -> set[str]:
    """Return the set of entry IDs already injected this session.

    Returns an empty set if the file is absent, unreadable, or older than SESSION_TTL_HOURS.
    """
    import time

    path = db_path / "session_injected.json"
    try:
        data = json.loads(path.read_text())
        age_hours = (time.time() - data["session_ts"]) / 3600
        if age_hours > SESSION_TTL_HOURS:
            return set()
        return set(data.get("ids", []))
    except Exception:
        return set()


def _save_session_injected(db_path: Path, ids: set[str]) -> None:
    """Persist the current session's injected IDs; silently ignores I/O errors."""
    import time

    path = db_path / "session_injected.json"
    try:
        existing = _load_session_injected(db_path)
        merged = existing | ids
        ts = time.time()
        # Preserve the original session_ts if the file is still valid.
        try:
            data = json.loads(path.read_text())
            age_hours = (time.time() - data["session_ts"]) / 3600
            if age_hours <= SESSION_TTL_HOURS:
                ts = data["session_ts"]
        except Exception:
            pass
        path.write_text(json.dumps({"session_ts": ts, "ids": list(merged)}))
    except Exception:
        pass


def _build_deps():
    from ai_mem._hook_deps import _build_query_uc

    query_uc = _build_query_uc(DB_PATH, with_bm25=True)
    return query_uc


def _hits(query_uc, collection: str, query: str):
    try:
        return query_uc.execute(collection=collection, query=query, n_results=TOP_K)
    except Exception:
        return []


def _antipattern_hits(query_uc, collection: str, query: str):
    try:
        results = query_uc.execute(
            collection=collection,
            query=query,
            n_results=ANTIPATTERN_TOP_K,
            type_filter="anti-pattern",
        )
        return [result for result in results if (result.score or 0.0) >= ANTIPATTERN_MIN_SCORE]
    except Exception:
        return []


def _dilemma_hits(query_uc, collection: str, query: str):
    try:
        results = query_uc.execute(
            collection=collection,
            query=query,
            n_results=DILEMMA_TOP_K,
            type_filter="dilemma",
        )
        return [result for result in results if (result.score or 0.0) >= DILEMMA_MIN_SCORE]
    except Exception:
        return []


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return

    try:
        from ai_mem.agent_context import detect_for_hook
        if not detect_for_hook(payload).should_inject:
            return
    except Exception:
        pass

    query = payload.get("prompt", "").strip()
    if len(query) < MIN_QUERY_CHARS or not DB_PATH.exists():
        return

    try:
        query_uc = _build_deps()
    except Exception:
        return

    global_results = _hits(query_uc, GLOBAL_COLLECTION, query)
    global_hits = [result for result in global_results if (result.score or 0.0) >= CONTEXT_MIN_SCORE]

    repo_hits = []
    repo_collection = None
    antipattern_results = []
    dilemma_results = []
    try:
        ctx = detect_repo_context()
        if ctx.has_claude_md:
            repo_collection = ctx.collection
            repo_results = _hits(query_uc, repo_collection, query)
            repo_hits = [result for result in repo_results if (result.score or 0.0) >= CONTEXT_MIN_SCORE]
            antipattern_results = _antipattern_hits(query_uc, repo_collection, query)
            dilemma_results = _dilemma_hits(query_uc, repo_collection, query)
    except Exception:
        pass

    if not global_hits and not repo_hits and not antipattern_results and not dilemma_results:
        return

    collected: list[tuple[str, Any]] = []
    collected.extend((GLOBAL_COLLECTION, result) for result in global_hits)
    if repo_collection:
        collected.extend((repo_collection, result) for result in repo_hits)

    try:
        from ai_mem.session_stats import record_injection
        record_injection(DB_PATH / "session_stats.json", GLOBAL_COLLECTION, injected=bool(collected))
    except Exception:
        pass

    if not collected and not antipattern_results and not dilemma_results:
        return

    # Per-session dedup: skip entries already injected in this session.
    already_injected = _load_session_injected(DB_PATH)
    collected = [(coll, r) for coll, r in collected if getattr(r, "id", None) not in already_injected or getattr(r, "id", None) is None]
    antipattern_results = [result for result in antipattern_results if getattr(result, "id", None) not in already_injected]
    dilemma_results = [result for result in dilemma_results if getattr(result, "id", None) not in already_injected]

    # Combined budget cap: include entries until MAX_TOTAL_CHARS is reached. Entries render as
    # id + title (see the [ai-mem available] block below), not full text, so the budget is
    # sized against what's actually displayed.
    budget_collected: list[tuple[str, Any]] = []
    chars_used = 0
    for coll, r in collected:
        entry_len = len(title_of(r.text)) + len(getattr(r, "id", "") or "")
        if chars_used + entry_len > MAX_TOTAL_CHARS:
            break
        budget_collected.append((coll, r))
        chars_used += entry_len
    collected = budget_collected

    if not collected and not antipattern_results and not dilemma_results:
        return

    lines: list[str] = []
    injected_ids: set[str] = set()
    pushed_by_collection: dict[str, set[str]] = {}

    def _mark_pushed(coll: str | None, entry_id: str | None) -> None:
        if entry_id is None:
            return
        injected_ids.add(entry_id)
        if coll is not None:
            pushed_by_collection.setdefault(coll, set()).add(entry_id)

    if dilemma_results:
        lines.append("[ai-mem dilemmas]")
        for result in dilemma_results:
            text = result.text[:MAX_CHARS_PER_DILEMMA]
            if len(result.text) > MAX_CHARS_PER_DILEMMA:
                text += "..."
            lines.append(f"⚖ {text}")
            _mark_pushed(repo_collection, getattr(result, "id", None))

    if antipattern_results:
        if lines:
            lines.append("")
        lines.append("[ai-mem warnings]")
        for result in antipattern_results:
            text = result.text[:MAX_CHARS_PER_ANTIPATTERN]
            if len(result.text) > MAX_CHARS_PER_ANTIPATTERN:
                text += "..."
            lines.append(f"⚠ {text}")
            if "Affected:" in result.text:
                lines.append(ANTICIPATION_QUESTION)
            _mark_pushed(repo_collection, getattr(result, "id", None))

    if collected:
        if lines:
            lines.append("")
        lines.append("[ai-mem available] (mem_get(ids=[...], collection=<name>) for full text)")
        for coll, r in collected:
            score = f"{r.score:.2f}" if r.score is not None else "n/a"
            entry_id = getattr(r, "id", None)
            lines.append(f"- [{coll} score={score}] {entry_id}: {title_of(r.text)}")
            _mark_pushed(coll, entry_id)

    _save_session_injected(DB_PATH, injected_ids)

    try:
        from ai_mem.injection_log import record_pushed
        log_path = DB_PATH / "injection_log.json"
        for coll, ids in pushed_by_collection.items():
            record_pushed(log_path, coll, list(ids))
    except Exception:
        pass

    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": "\n".join(lines),
                }
            }
        )
    )


if __name__ == "__main__":
    main()
