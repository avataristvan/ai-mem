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
        └─► cosine used as-is; BM25 raw scores pass through a saturating transform
            └─► fuse: hybrid = alpha × cosine_raw + (1-alpha) × bm25_sat
                └─► sort descending, truncate to n_results
                    └─► QueryResult.score = fused hybrid score
```

`_BM25_FETCH = 50` is the internal over-fetch count. The caller's `n_results` only controls the final truncation.

## Score Fusion Formula

```
hybrid = alpha × cosine_raw + (1 - alpha) × bm25_sat
```

`cosine_raw` is used as-is — it's already an absolute, fixed-scale similarity (post score-scaling fix in `chroma_repository.py`), so no normalization is applied. `bm25_sat` is BM25's raw score passed through a saturating transform.

## Normalisation

BM25's raw score is unbounded and corpus/query-length dependent, so it's mapped into `[0, 1)` with a saturating transform instead of a pool-relative min-max rescale. Raw scores are clamped to 0 first: `rank_bm25` can return negative scores (it floors negative IDFs at `0.25 * average_idf`, which is itself negative for small/near-duplicate corpora), and `raw/(raw+k)` is only monotonic on one side of its pole at `raw = -k` — clamping avoids crossing that pole and inverting rank order:

```python
_BM25_SATURATION_K = 2.0

def _saturate(scores):
    return [max(s, 0.0) / (max(s, 0.0) + _BM25_SATURATION_K) for s in scores]
```

`k = 2.0` is empirically derived from the live-DB median nonzero raw BM25 score (~1.6), not a guess. Unlike min-max normalisation, this transform depends only on each candidate's own raw score, not on what else is in the fetched pool — so the best-in-pool candidate is no longer unconditionally mapped to `1.0` regardless of its absolute relevance.

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
