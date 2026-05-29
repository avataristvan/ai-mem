# CLAUDE.md

Agent-facing supplement to README.md. Dev commands and critical contributor gotchas.

## Dev Commands

```bash
pip install -e .            # editable install (no ML, no BM25)
pip install -e ".[hybrid]"  # with BM25 hybrid search
pip install -e ".[ml]"      # with PyTorch re-ranker
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

- `TrainingExample` has **no `.label` attribute** — use `.target_future_access is not None` to check for a label. `.label` raises `AttributeError` silently caught by surrounding try/except, making labeled counts always 0.
- Tests that call `upsert` directly must include non-empty metadata (ChromaDB rejects empty dicts). Use `AddMemoryUseCase` in tests — it always injects timestamps.
- `posttool_hook.py` imports `GLOBAL_COLLECTION`, `WORKSPACE_COLLECTION`, and `detect_repo_context` at **module level** (not lazily inside `main()`) so tests can patch them via `patch.object`. Hooks that use lazy imports inside `main()` are not patchable at module scope.
- `MIN_LABELED_EXAMPLES = 10` is defined in `ranker_registry.py` — that is the canonical source. `userprompt_hook.py` and `hook.py` import/mirror from there.
- `mem_delete` with no `ids` drops the **entire collection**; repo signals this by returning `-1`.
- `RankingFeatures.cosine_similarity` = `1 - chromadb_distance` (higher = more relevant). Never invert.
