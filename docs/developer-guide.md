# Developer Guide

## Adding a Use Case

1. Create `ai_mem/application/<domain>.py` with a class named `<Action>UseCase`
2. `__init__` accepts only injected dependencies (repositories, other use cases)
3. Single public method: `execute(self, ...)` — no side effects via globals
4. Wire it up in `server.py` at module load (not inside request handlers)
5. Add MCP tool declaration in `list_tools()` and handler branch in `call_tool()`

Example skeleton:
```python
from ai_mem.domain.memory import MemoryRepository

class MyUseCase:
    def __init__(self, repo: MemoryRepository) -> None:
        self._repo = repo

    def execute(self, collection: str, ...) -> ...:
        ...
```

## Adding Infrastructure

Add a concrete implementation of a domain Protocol in `ai_mem/infrastructure/`. Never import from `application/` or other infrastructure modules (no cross-infrastructure deps).

## Running Tests

```bash
python -m pytest tests/ -v           # all tests
python -m pytest tests/test_session_stats.py -v  # single file
```

Test fixtures are in `tests/conftest.py`. Infrastructure tests use a real ChromaDB instance in a temp directory.

## Test Counts (as of 2026-04-29)

- Total: 76 tests, all green
- Breakdown by module: memory_index (15), repo_seeder (part of 76), session_stats, userprompt_hook

## Conventions

- All timestamps stored as Unix float in ChromaDB metadata: `created_at`, `expires_at`, `last_accessed_at`, `access_count`
- `mem_delete` with no `ids` returns `-1` (signals full collection drop, not count)
- `_FETCH_K = 20` in `QueryMemoryUseCase` — hardcoded over-fetch before re-ranking
- `RankingFeatures` is frozen: all derived values computed in `as_vector()`, raw fields are immutable
- `RankerScope` validates its own invariants in `__post_init__`
- `NullRanker` is the zero-dependency fallback — must not import torch

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_MEM_PATH` | `~/.local/share/ai-mem` | Database + ranker weights location |

Optional `{AI_MEM_PATH}/ranker_config.json` configures hybrid ranker groups. Restart the MCP server after editing.

## Install / Registration

```bash
python3 install.py   # registers MCP server + hooks in ~/.claude/settings.json
pip install -e .     # editable install (no ML)
pip install -e ".[ml]"  # with PyTorch re-ranker
```

## Module Entry Points

```bash
python3 -m ai_mem.server          # MCP server (stdio)
python3 -m ai_mem.hook            # SessionStart hook
python3 -m ai_mem.stop_hook       # Stop hook
python3 -m ai_mem.userprompt_hook # UserPromptSubmit hook
```
