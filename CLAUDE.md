# CLAUDE.md

Agent-facing supplement to README.md. Contains dev commands, architecture navigation, and non-obvious conventions.

## Dev Commands

```bash
pip install -e .            # editable install (no ML, no BM25)
pip install -e ".[hybrid]"  # with BM25 hybrid search
pip install -e ".[ml]"      # with PyTorch re-ranker
pip install -e ".[dream]"   # with Anthropic SDK for mem-dream
python3 -m ai_mem.server    # run MCP server directly
python3 install.py          # register with Claude Code / Gemini CLI / Cursor
python3 -m ai_mem.hook          # run SessionStart hook manually
python3 -m ai_mem.stop_hook     # run Stop hook manually (from a git repo dir)
python3 -m ai_mem.posttool_hook # run PostToolUse hook manually (pipe JSON payload on stdin)
python -m pytest tests/ -v  # run tests
mem-dream --dry-run         # preview entries without API calls
mem-dream --mode hier       # consolidate all collections (hier = default)
mem-dream --mode team --collection repo.ExoDeck  # team exchange, one collection
```

## Architecture

Capability-centric DDD — three layers, no upward imports.

**Domain** (`ai_mem/domain/`) — pure contracts, no I/O, no torch.
- `memory.py` — `MemoryEntry`, `QueryResult`, `CollectionInfo`, `MemoryRepository` Protocol, `MemoryEdge`, `EdgeType`
- `learning.py` — `RankingFeatures` (frozen, 10-element `as_vector()`), `TrainingExample`, `TrainingMetrics`, `RankerScope`, `LearnedRanker` Protocol, `TrainingBufferRepository` Protocol, `RankerProvider` Protocol

**Application** (`ai_mem/application/`) — one use case per file, single `execute()`, deps injected via `__init__`.
- `QueryMemoryUseCase` — top-50 fetch → build features → `RankerProvider.get()` → re-rank → truncate → track access → record training signal → 1-hop edge follow (budget: 2 linked entries appended)
- `AddEdgeUseCase` — validates both endpoints exist, deduplicates on (target_id, edge_type), writes via `repo.add_edge()`
- `GetEdgesUseCase` — thin wrapper over `repo.get_edges()`
- `TrainRankerUseCase` — buffer write, label assignment (7-day window), gradient step, NaN-loss guard, labeled-example eviction
- `RankerRegistry` — lazy-loads and caches one ranker per scope key; implements `RankerProvider`; lives in application layer because it coordinates infrastructure artifacts
- `CleanupMemoryUseCase` — returns `CleanupResult(collections: dict[str, CollectionCleanupStats])`
- `ListEntriesUseCase` — returns all entries in a collection as `[{id, title}]`; title = first non-empty line, max 80 chars

**Infrastructure** (`ai_mem/infrastructure/`)
- `ChromaMemoryRepository` — timestamps as Unix float metadata: `created_at`, `expires_at`, `last_accessed_at`, `access_count`
- `BM25MemoryRepository` — optional wrapper; fetches 50 candidates from inner repo, fuses BM25+cosine scores, returns hybrid-ranked results. Requires `rank_bm25` (`.[hybrid]`). Wired in `server.py` and `userprompt_hook.py` with `try/except ImportError` fallback.
- `TorchMicroRanker` — `[10→32→16→1]` MLP, AdamW lr=1e-3, BCE + 0.3×contrastive loss. `seed` param for deterministic init in tests only.
- `NullRanker` — returns `cosine_similarity` scores unchanged; active when torch is absent
- `RankerStorage` — implements `TrainingBufferRepository`; JSONL buffer + `.pt` weights per scope key

**Adapter** (`ai_mem/server.py`) — wires use cases at module load, exposes MCP tools. Claude Code lifecycle hooks:
- `hook.py` — SessionStart: injects current_focus + collection routing
- `userprompt_hook.py` — UserPromptSubmit: injects relevant memories when ranker is trained enough
- `pretool_hook.py` — PreToolUse(Write|Edit): injects relevant past experiences before a file is touched
- `posttool_hook.py` — PostToolUse(Write|Edit): silent passive training signal; queries global + repo collections with the edited file path, updating `last_accessed_at` on matching entries so `train_step` labels them positive in the 7-day window. No output to Claude.

**Utility modules** (not use-case classes — standalone functions, no DI):
- `session_stats.py` — `record_injection` / `injection_rate`; rolling 20-session JSONL at `{DB_PATH}/session_stats.json`
- `repo_seeder.py` — `seed_collection`; reads CLAUDE.md H2 sections → `AddMemoryUseCase`; gated by `injection_rate >= SEED_THRESHOLD (0.60)`

## Key Conventions

- `mem_delete` with no `ids` drops the entire collection; repo signals this by returning `-1`.
- `current_focus` (id `"current_focus"`) is the primary context entry per collection. The Stop hook reminds the agent to update it when files changed.
- `RankingFeatures.cosine_similarity` = `1 - chromadb_distance` (higher = more relevant). Never invert.
- `_FETCH_K = 20` in `QueryMemoryUseCase` — always over-fetches 20 candidates before re-ranking; `n_results` only controls final truncation.
- Hybrid mode: buffer and weights files are keyed by **group name**, not collection name. `RankerRegistry.scope_key()` resolves this.
- `source_collection: str | None` on `TrainingExample` — `None` means the field was absent in older buffer files (backwards-compat on deserialize).
- `_try_seed` in `hook.py` checks `collection count == 0` (or not listed) before seeding; wrapped in `try/except` so it is always silent.
- `hook.py` records global injection stats (`record_injection`) before `_try_seed` so the first-ever session counts toward the threshold.
- Section splitting in `repo_seeder.py` only includes H2 sections (`## `); the H1 intro block is intentionally excluded.
- `BM25MemoryRepository` is transparent to the application layer — `QueryResult.score` holds the fused hybrid score, which becomes `RankingFeatures.cosine_similarity` in `BuildFeaturesUseCase`. The app layer never detects the wrapper.
- Tests that call `upsert` directly must include non-empty metadata (ChromaDB rejects empty dicts). Use `AddMemoryUseCase` in tests to avoid this — it always injects timestamps.
- `type` metadata field: `mem_add` accepts an optional `type` param (e.g. `"feedback"`, `"reference"`, `"project"`, `"user"`). `mem_query` accepts a matching `type` filter. ChromaDB `$and` is used when both `max_age_days` and `type_filter` are set.
- `mem_list` with a `collection` param returns entry titles instead of collection counts; backed by `ListEntriesUseCase` and `get_all()`.
- `BM25MemoryRepository.query()` forwards `type_filter` to the inner repo — filter is applied at the ChromaDB level before BM25 re-ranking.
- `posttool_hook.py` imports `GLOBAL_COLLECTION`, `WORKSPACE_COLLECTION`, and `detect_repo_context` at module level (not lazily inside `main()`) so tests can patch them via `patch.object(hook, ...)`. This is the same pattern as `userprompt_hook.py`. Hooks that use lazy `from ... import` inside `main()` are not patchable at module scope.
- PostToolUse hook does NOT wrap the inner repo with `BM25MemoryRepository` — BM25 adds latency and the hook only needs semantic proximity for label propagation, not high-precision retrieval.

## Typed Causal Edges

Entries can be linked with directional typed edges (`contradicts`, `fixes`, `causes`, `related`).

**Storage**: edges are stored as a JSON-encoded string in the `edges` metadata field of the **source** entry. ChromaDB only accepts scalar metadata values, so the list is `json.dumps`/`json.loads` as a string. `_parse_edges()` in `chroma_repository.py` always returns `[]` on parse error (silent).

**Retrieval**: `QueryMemoryUseCase._append_linked()` runs after the primary ranked results are assembled. It follows all outgoing edges of each result entry (1-hop only, no recursion), fetches the target via `get_by_ids`, and appends any target not already in the result set. Budget: `_EDGE_BUDGET = 2` linked entries per query total. Appended entries have `via_edge` and `via_source` in their metadata. Edge traversal is wrapped in a bare `except Exception: pass` so it cannot crash a query.

**Protocol**: `MemoryRepository` has two new methods — `add_edge(collection, source_id, edge)` and `get_edges(collection, entry_id) → list[MemoryEdge]`. Both `ChromaMemoryRepository` and `BM25MemoryRepository` implement them (`BM25` delegates to inner).

**MCP tools**: `mem_link(source_id, target_id, edge_type, collection)` and `mem_edges(entry_id, collection)`.

**Test for edge-follow**: the `linked_target` entry must NOT be semantically similar to the query — otherwise ChromaDB returns it directly and `via_edge` is never set (the dedup guard skips it). Use semantically unrelated text (e.g. `ZZZQQQXXX boilerplate`) for the linked target in tests.
