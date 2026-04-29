# Hybrid Search (BM25 + Cosine)

## Design Decision

`BM25MemoryRepository` is a transparent infrastructure wrapper, not a use case. The application layer (`QueryMemoryUseCase`) never knows whether it is talking to a plain `ChromaMemoryRepository` or the hybrid wrapper — both satisfy the `MemoryRepository` protocol. This means:

- No application-layer changes are needed to enable or disable hybrid search.
- The wrapper can be inserted or removed at the wiring site (`server.py`, `userprompt_hook.py`) without touching business logic.
- `rank_bm25` stays an optional dependency; the system degrades gracefully when it is absent.

## Pipeline

```
ChromaDB (fetch 50 candidates via cosine)
    └─► BM25Okapi (score same 50 documents)
        └─► min-max normalise both score sets to [0, 1]
            └─► fuse: hybrid = alpha × cosine_norm + (1-alpha) × bm25_norm
                └─► sort descending, truncate to n_results
                    └─► QueryResult.score = fused hybrid score
                        └─► TorchMicroRanker re-ranks top-20 with learned features
```

`_BM25_FETCH = 50` is the internal over-fetch count. The caller's `n_results` only controls the final truncation.

## Score Fusion Formula

```
hybrid = alpha × cosine_norm + (1 - alpha) × bm25_norm
```

Both `cosine_norm` and `bm25_norm` are min-max normalised from the same 50-candidate pool before fusion.

## Normalisation

```python
def _normalize(scores):
    lo, hi = min(scores), max(scores)
    if hi == lo:          # degenerate: all scores equal
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]
```

Degenerate case (all scores identical) returns `1.0` for every candidate so no information is lost and the other signal determines final order.

## Alpha Configuration

`alpha` is a constructor parameter with a default of `0.5` (equal weight).

```python
# Equal weight (default)
repo = BM25MemoryRepository(inner_repo)

# Prefer cosine (semantic-heavy workload)
repo = BM25MemoryRepository(inner_repo, alpha=0.7)

# Prefer BM25 (exact-term-heavy workload)
repo = BM25MemoryRepository(inner_repo, alpha=0.3)
```

## Installation

```bash
# hybrid only
pip install -e ".[hybrid]"

# hybrid + ML re-ranker
pip install -e ".[ml,hybrid]"

# dev environment (includes rank_bm25)
pip install -e ".[dev]"
```

## Fallback Behaviour

When `rank_bm25` is not installed, `server.py` and `userprompt_hook.py` fall back to the plain `ChromaMemoryRepository` silently:

```python
try:
    from ai_mem.infrastructure.bm25_repository import BM25MemoryRepository
    _repo = BM25MemoryRepository(_inner_repo)
except ImportError:
    _repo = _inner_repo
```

No configuration or code change is needed — install `rank_bm25` and restart the server to activate hybrid search.

## Tokenisation

BM25 tokenises with `.lower().split()` — simple whitespace splitting, no NLTK or spaCy dependency.
