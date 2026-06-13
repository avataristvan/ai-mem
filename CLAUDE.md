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
