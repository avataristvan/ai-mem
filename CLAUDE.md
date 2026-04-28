# CLAUDE.md

Agent-facing supplement to README.md. Contains dev commands, architecture navigation, and non-obvious conventions.

## Dev Commands

```bash
pip install -e .            # editable install (no ML)
pip install -e ".[ml]"      # with PyTorch re-ranker
python3 -m ai_mem.server    # run MCP server directly
python3 install.py          # register with Claude Code / Gemini CLI / Cursor
python3 -m ai_mem.hook      # run SessionStart hook manually
python3 -m ai_mem.stop_hook # run Stop hook manually (from a git repo dir)
python -m pytest tests/ -v  # run tests
```

## Architecture

Capability-centric DDD — three layers, no upward imports.

**Domain** (`ai_mem/domain/`) — pure contracts, no I/O, no torch.
- `memory.py` — `MemoryEntry`, `QueryResult`, `CollectionInfo`, `MemoryRepository` Protocol
- `learning.py` — `RankingFeatures` (frozen, 10-element `as_vector()`), `TrainingExample`, `TrainingMetrics`, `RankerScope`, `LearnedRanker` Protocol, `TrainingBufferRepository` Protocol, `RankerProvider` Protocol

**Application** (`ai_mem/application/`) — one use case per file, single `execute()`, deps injected via `__init__`.
- `QueryMemoryUseCase` — top-20 fetch → build features → `RankerProvider.get()` → re-rank → truncate → track access → record training signal
- `TrainRankerUseCase` — buffer write, label assignment (7-day window), gradient step, NaN-loss guard, labeled-example eviction
- `RankerRegistry` — lazy-loads and caches one ranker per scope key; implements `RankerProvider`; lives in application layer because it coordinates infrastructure artifacts
- `CleanupMemoryUseCase` — returns `CleanupResult(collections: dict[str, CollectionCleanupStats])`

**Infrastructure** (`ai_mem/infrastructure/`)
- `ChromaMemoryRepository` — timestamps as Unix float metadata: `created_at`, `expires_at`, `last_accessed_at`, `access_count`
- `TorchMicroRanker` — `[10→32→16→1]` MLP, AdamW lr=1e-3, BCE + 0.3×contrastive loss. `seed` param for deterministic init in tests only.
- `NullRanker` — returns `cosine_similarity` scores unchanged; active when torch is absent
- `RankerStorage` — implements `TrainingBufferRepository`; JSONL buffer + `.pt` weights per scope key

**Adapter** (`ai_mem/server.py`) — wires use cases at module load, exposes MCP tools. `hook.py` / `stop_hook.py` are Claude Code lifecycle hooks.

## Key Conventions

- `mem_delete` with no `ids` drops the entire collection; repo signals this by returning `-1`.
- `current_focus` (id `"current_focus"`) is the primary context entry per collection. The Stop hook reminds the agent to update it when files changed.
- `RankingFeatures.cosine_similarity` = `1 - chromadb_distance` (higher = more relevant). Never invert.
- `_FETCH_K = 20` in `QueryMemoryUseCase` — always over-fetches 20 candidates before re-ranking; `n_results` only controls final truncation.
- Hybrid mode: buffer and weights files are keyed by **group name**, not collection name. `RankerRegistry.scope_key()` resolves this.
- `source_collection: str | None` on `TrainingExample` — `None` means the field was absent in older buffer files (backwards-compat on deserialize).
