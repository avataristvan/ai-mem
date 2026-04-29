# Adaptive Re-ranking

ai-mem trains a small neural network per collection (or group) that learns to predict which memories will be accessed again, boosting their rank at query time.

## Feature Vector (10 elements)

`RankingFeatures.as_vector()` returns:

| Index | Value | Source |
|-------|-------|--------|
| 0 | `cosine_similarity` | 1 − ChromaDB distance |
| 1 | `age_days` | now − created_at |
| 2 | `last_access_days` | now − last_accessed_at (fallback: age_days) |
| 3 | `access_count` | raw count |
| 4 | `is_never_accessed` | 1.0 if count == 0 |
| 5 | `has_ttl` | 1.0 if TTL is set |
| 6 | `ttl_horizon` | expires_in_days (or 365.0 if no TTL) |
| 7 | `log1p(age_days)` | log transform |
| 8 | `log1p(last_access_days)` | log transform |
| 9 | `log1p(access_count)` | log transform |

**Critical:** `cosine_similarity = 1 − distance`. Higher = more relevant. Never invert this value.

## TorchMicroRanker

Architecture: `[10 → 32 → 16 → 1]` MLP with BatchNorm and ReLU activations.

**Loss:** BCE loss + 0.3 × contrastive loss (co-activated entries pulled closer).

**Training:**
- Labels assigned from `last_accessed_at` history after a 7-day window
- If an entry was accessed within 7 days of retrieval: label = 1.0, else 0.0
- NaN-loss guard: training step is skipped if loss is NaN (prevents weight corruption)
- Labeled examples are evicted from buffer after training to prevent re-labeling

**Optimizer:** AdamW, lr=1e-3

**Fallback:** When PyTorch is not installed, `NullRanker` returns raw `cosine_similarity` scores unchanged.

## Training Pipeline

```
QueryMemoryUseCase.execute()
  ├── ChromaDB.query(top_20)          # always over-fetches 20
  ├── BuildFeaturesUseCase            # QueryResult + MemoryEntry → RankingFeatures
  ├── RankerRegistry.get(collection)  # loads/caches LearnedRanker
  ├── ranker.rank(features)           # re-score
  ├── sort by new score, slice n_results
  ├── TrackAccessUseCase              # bump last_accessed_at + access_count
  └── TrainRankerUseCase.append()    # buffer the TrainingExample
```

`TrainRankerUseCase.train_step()` is called explicitly via `mem_train` tool (or from `userprompt_hook.py`).

## RankerStorage

- Buffer: `{DB_PATH}/rankers/{scope_key}.jsonl`
- Weights: `{DB_PATH}/rankers/{scope_key}.pt`
- Corrupt lines are logged to stderr and skipped (never crash)

## Hybrid Mode

Multiple collections can share one trained ranker via `ranker_config.json`:

```json
{
  "groups": [
    {
      "name": "work-services",
      "collections": ["repo.payment-svc", "repo.order-svc"],
      "mode": "hybrid"
    }
  ]
}
```

- Scope key for isolated collection = collection name
- Scope key for hybrid member = group name
- `RankerRegistry.scope_key(collection)` resolves this
- `TrainRankerUseCase` deduplicates by scope key when training all collections to avoid double-training shared weights

## RankerRegistry

Lives in the application layer (coordinates infrastructure artifacts). Lazy-loads and caches one `LearnedRanker` per scope key. Implements `RankerProvider` Protocol.
