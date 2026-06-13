# Confidence Lifecycle

> The ML re-ranker was removed in 2026-06. This document describes the replacement model.

ai-mem tracks the epistemic status of each entry via a `confidence` field (float 0.0–1.0) stored in ChromaDB metadata.

## Lifecycle

| Stage | Trigger | Effect |
|-------|---------|--------|
| Write | New entry created | `confidence = 0.7` |
| Boost | `/reflect` confirms entry was decisive | `mem_boost(delta=+0.1)` |
| Decay | Dream cycle flags it as stale | `mem_boost(delta=-0.1)` |
| Always-present | `confidence > 0.9` AND `access_count ≥ 3` AND `boost_count ≥ 1` | Injected at every `SessionStart` |
| Decay candidate | `confidence < 0.3` | Flagged in `mem_dream` report |
| Promotion candidate | `confidence > 0.9` AND `access_count ≥ 3` | Flagged in `mem_dream` report for CLAUDE.md |

Explicit override: pass `confidence=<value>` in `mem_add` metadata to set a precise start value. Pass `confidence_delta=<±value>` to apply an additive adjustment relative to the stored value.

## Retrieval

`QueryMemoryUseCase` returns results in ChromaDB's native cosine order. No re-ranking step. `TrackAccessUseCase` increments `access_count` and updates `last_accessed_at` on each retrieval — these drive the confidence gate for `[always-present]`.

## mem_boost

```python
mem_boost(ids=["entry_id"], collection="repo.my-project", delta=0.1)   # boost
mem_boost(ids=["entry_id"], collection="repo.my-project", delta=-0.1)  # decay
```

Called by `/reflect` Step 3.1 for entries that were decisive in the session.

## [always-present] injection

`hook.py` scans the active collection at session start. Entries meeting the gate (`confidence > 0.9`, `access_count ≥ 3`, `boost_count ≥ 1`) are embedded directly in the `additionalContext` block, independent of any query.

Constants (in `hook.py`):
- `_HIGH_CONFIDENCE_THRESHOLD = 0.9`
- `_HIGH_CONFIDENCE_MIN_ACCESS = 3`
- `_HIGH_CONFIDENCE_MIN_BOOSTS = 1` (must have been explicitly boosted at least once via /reflect)
- `_HIGH_CONFIDENCE_MAX = 3` (max entries injected)
- `_HIGH_CONFIDENCE_CHARS = 300` (truncation limit per entry)
