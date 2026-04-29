# Architecture

ai-mem uses capability-centric DDD with three strict layers. Imports only flow downward: adapter → application → domain.

```
ai_mem/
  domain/          pure contracts — no I/O, no torch
  application/     one use case per file, single execute(), deps injected
  infrastructure/  ChromaDB, torch, file I/O
  server.py        MCP adapter — wires everything at module load
  hook.py          SessionStart hook
  stop_hook.py     SessionStop hook
  userprompt_hook.py  UserPromptSubmit hook
  pretool_hook.py  PreToolUse hook (context injection)
  memory_index.py  MEMORY.md auto-demotion utility
  repo_seeder.py   CLAUDE.md → seed entries on first session
  session_stats.py rolling injection stats (20-session window)
  repo_context.py  git-root detection + collection name derivation
```

## Domain (`ai_mem/domain/`)

### `memory.py`
- `MemoryEntry` — id, text, metadata + datetime fields (created_at, expires_at, last_accessed_at, access_count)
- `QueryResult` — rank, id, score, text, metadata
- `CollectionInfo` — name, count
- `MemoryRepository` Protocol — upsert, query, get_by_ids, list_collections, delete, delete_expired, record_access, delete_stale

### `learning.py`
- `RankingFeatures` (frozen dataclass) — 6 raw fields; `as_vector()` returns 10-element list (raw + log transforms + never-accessed flag)
  - **`cosine_similarity` = 1 − ChromaDB distance** (higher = more relevant, never invert)
- `TrainingExample` — memory_id, features, retrieved_at, co_activated_ids, source_collection, target_future_access (None → 0.0/1.0 after label window)
- `TrainingMetrics` — n, loss, skipped
- `RankerScope` — name, mode (`"isolated"` | `"hybrid"`), group, member_collections; `__post_init__` validates invariants
- `LearnedRanker` Protocol — rank(), train_step(), save(), load()
- `TrainingBufferRepository` Protocol — append_examples, load_buffer, clear_buffer, prune_buffer, weights_path
- `RankerProvider` Protocol — get(collection) → LearnedRanker

## Application (`ai_mem/application/`)

| File | Use case | Responsibility |
|------|----------|----------------|
| `add_memory.py` | `AddMemoryUseCase` | upsert with timestamps; preserves existing history on re-add |
| `query_memory.py` | `QueryMemoryUseCase` | fetch 20 → build features → re-rank → truncate → track access → append training signal |
| `get_memory.py` | `GetMemoryUseCase` | fetch by exact IDs |
| `delete_memory.py` | `DeleteMemoryUseCase` | delete by IDs or drop collection |
| `list_collections.py` | `ListCollectionsUseCase` | list with counts |
| `cleanup_memory.py` | `CleanupMemoryUseCase` | TTL expiry + optional stale delete |
| `track_access.py` | `TrackAccessUseCase` | bump last_accessed_at + access_count |
| `build_features.py` | `BuildFeaturesUseCase` | QueryResult + MemoryEntry → RankingFeatures |
| `train_ranker.py` | `TrainRankerUseCase` | buffer write, label assignment (7-day window), gradient step, NaN-loss guard, labeled eviction |
| `load_ranker_config.py` | `LoadRankerConfigUseCase` | parse optional ranker_config.json → scope map |
| `ranker_registry.py` | `RankerRegistry` | lazy-load + cache one ranker per scope key; implements RankerProvider |

`_FETCH_K = 20` in `QueryMemoryUseCase` — always over-fetches 20 before re-ranking; `n_results` controls final slice only.

## Infrastructure (`ai_mem/infrastructure/`)

| File | Class | Notes |
|------|-------|-------|
| `chroma_repository.py` | `ChromaMemoryRepository` | timestamps as Unix float metadata |
| `bm25_repository.py` | `BM25MemoryRepository` | Wraps any MemoryRepository; fetches 50 candidates, fuses BM25+cosine scores, returns hybrid-ranked results |
| `torch_ranker.py` | `TorchMicroRanker` | [10→32→16→1] MLP, AdamW lr=1e-3, BCE + 0.3×contrastive loss; `seed` param for deterministic test init |
| `null_ranker.py` | `NullRanker` | returns cosine_similarity scores unchanged; active when torch absent |
| `ranker_storage.py` | `RankerStorage` | JSONL buffer + `.pt` weights per scope key; stderr-logs corrupt buffer lines |

## Key Invariants

- `mem_delete` with no `ids` drops the entire collection; repository signals this with return value `-1`
- `current_focus` (id `"current_focus"`) is the primary context entry per collection
- Hybrid mode: buffer and weights files are keyed by **group name**, not collection name
- `source_collection: str | None` on `TrainingExample` — `None` = absent in older buffer files (backwards-compat)
- `BM25MemoryRepository` is transparent to the application layer — `QueryResult.score` holds the fused hybrid score, which becomes `RankingFeatures.cosine_similarity` in `BuildFeaturesUseCase`
