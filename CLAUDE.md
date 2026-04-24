# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install package in editable mode
pip install -e .

# Run the MCP server directly (for testing)
python3 -m ai_mem.server

# Run the installer (registers with Claude Code / Gemini CLI / Cursor)
python3 install.py

# Run the SessionStart hook manually
python3 -m ai_mem.hook
```

There is no test suite yet. The package is installed editable (`pip install -e .`) so changes take effect immediately without reinstall.

## Architecture

Capability-centric DDD in three layers:

**`ai_mem/domain/memory.py`** — pure entities (`MemoryEntry`, `QueryResult`, `CollectionInfo`) and the `MemoryRepository` Protocol. No I/O, no ChromaDB imports.

**`ai_mem/application/`** — one use case per file (`AddMemoryUseCase`, `QueryMemoryUseCase`, `DeleteMemoryUseCase`, `ListCollectionsUseCase`, `CleanupMemoryUseCase`). Each takes a `MemoryRepository` in `__init__` and exposes a single `execute()` method. Timestamp injection and TTL calculation live here (in `AddMemoryUseCase`), not in the infra layer.

**`ai_mem/infrastructure/chroma_repository.py`** — `ChromaMemoryRepository` implements `MemoryRepository` using ChromaDB's `PersistentClient`. Timestamps are stored as Unix float metadata fields (`created_at`, `expires_at`). The `max_age_days` filter in `query()` and the expired-entry sweep in `delete_expired()` both operate on these fields via ChromaDB `where` clauses.

**`ai_mem/server.py`** — thin MCP adapter. Instantiates the repo and all use cases at module load, wires them to `mem_add` / `mem_query` / `mem_list` / `mem_delete` / `mem_cleanup` tools. Default collection is `"workspace"`.

**`ai_mem/hook.py`** — Claude Code `SessionStart` hook. Reads the `current_focus` entry from the `workspace` collection and emits it as `hookSpecificOutput` so Claude sees it at session start. Silent if no entry exists.

**`install.py`** — patches `~/.claude.json` (MCP server registration), `~/.claude/settings.json` (SessionStart + Stop hooks), `~/.gemini/settings.json`, and `~/.cursor/mcp.json`. Idempotent — checks for duplicates before inserting.

## Key conventions

- `mem_add` always injects `created_at` (UTC timestamp). `expires_at` is only added when `ttl_days` is set.
- `mem_delete` with no `ids` drops the entire collection (returns `-1` from the repo to signal this).
- The `current_focus` entry (id `"current_focus"` in `workspace`) is a convention used by the Stop hook to remind the agent to update its focus when files change.
- Data path defaults to `~/.local/share/ai-mem`; override with `AI_MEM_PATH` env var.
