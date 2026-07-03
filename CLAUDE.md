# CLAUDE.md

Agent-facing supplement to README.md. Dev commands and critical contributor gotchas.

## Dev Commands

```bash
pip install -e .            # editable install (no ML, no BM25)
pip install -e ".[hybrid]"  # with BM25 hybrid search
pip install -e ".[dream]"   # with Anthropic SDK for mem-dream
pip install -e ".[config]"  # with pyyaml for agents.yaml support
python3 -m ai_mem.server    # run MCP server directly
python3 install.py          # register with Claude Code / Gemini CLI / Cursor
python3 uninstall.py        # remove all integration artifacts (preserves data)
python3 -m ai_mem.hook          # run SessionStart hook manually
python3 -m ai_mem.posttool_hook # run PostToolUse hook manually (pipe JSON payload on stdin)
python -m pytest tests/ -v  # run tests
mem-dream --dry-run         # preview entries without API calls
mem-dream --mode hier       # consolidate all collections (hier = default)
mem-dream --expert          # consolidate all subagent.* collections with expert focus hint
```

## Critical Invariants

Things that will bite you in the first hour:

- Tests that call `upsert` directly must include non-empty metadata (ChromaDB rejects empty dicts). Use `AddMemoryUseCase` in tests — it always injects timestamps.
- `posttool_hook.py` imports `GLOBAL_COLLECTION`, `WORKSPACE_COLLECTION`, and `detect_repo_context` at **module level** (not lazily inside `main()`) so tests can patch them via `patch.object`. Hooks that use lazy imports inside `main()` are not patchable at module scope.
- `mem_delete` with no `ids` drops the **entire collection**; repo signals this by returning `-1`.
- `QueryMemoryUseCase` takes only `(repo, track_access)` — no ranker, no build_features. ChromaDB returns results in cosine order; no re-ranking needed.
- `userprompt_hook.py` context injection fires whenever a result score ≥ `CONTEXT_MIN_SCORE = 0.3`. No ranker calibration gate — removed in 2026-06.
- `_hook_deps.py` exposes `_build_query_uc(db_path, with_bm25)` — the single assembly point for hooks. `posttool_hook` passes `with_bm25=False`; `userprompt_hook` passes `True`.
- The ML ranker (`TorchMicroRanker`, `RankerRegistry`) was removed in 2026-06. No `[ml]` extra or `torch` dependency remains.
- `[always-present]` gate requires `confidence > 0.9` AND `access_count ≥ 3` AND `boost_count ≥ 1`. Entries without an explicit `/reflect` boost never qualify — regardless of age or access count. Existing high-value entries from before the confidence system need a one-time `mem_boost` to re-qualify.
- `mem_move` (`MoveMemoryUseCase` in `application/move_memory.py`) moves entries between collections by reading via `get_by_ids`, writing via `upsert`, then `delete`-ing the confirmed subset from the source — it deliberately bypasses `AddMemoryUseCase` (whose "preserve prior metadata" merge only looks within the same collection it's writing to, which would wrongly reset confidence/access_count on a cross-collection move). Same pattern as `split_memory.py`. If an id already exists at the target with different text, it's skipped and reported as a `MoveConflict` rather than overwritten; identical text at the target is a no-op merge (still moved). `from_collection == to_collection` raises `ValueError` (caught in `server.py`, returned as `Error: ...`) instead of silently deleting entries via a no-op upsert+delete round trip. Duplicate ids in the request are deduped up front. An empty `ids` list short-circuits to an empty `MoveResult` before any repo call — `get_by_ids([])` itself raises inside ChromaDB, and even if it didn't, `repo.delete(collection, [])` would drop the entire collection. Known v1 limitation: edges only reference `target_id`, no collection field, so other entries in the source collection that referenced a moved entry become dangling — not fixed, no reverse-edge index exists.
