# ai-mem

Persistent semantic memory for AI agents. Stores, searches, and retrieves information across sessions using natural language. Backed by [ChromaDB](https://www.trychroma.com/) with an optional learned re-ranker that adapts to your access patterns over time.

Works with **Claude Code**, **Gemini CLI**, and **Cursor**.

## Install

```bash
git clone https://github.com/avataristvan/ai-mem.git
cd ai-mem
python3 install.py
```

Restart your AI tool after installation. For the adaptive re-ranker (optional):

```bash
pip install -e ".[ml]"   # requires PyTorch
```

## Tools

| Tool | Description |
|------|-------------|
| `mem_add` | Store or update entries. Set `ttl_days` for automatic expiry. |
| `mem_query` | Semantic search. Returns re-ranked results with scores. |
| `mem_list` | List all collections with entry counts. |
| `mem_delete` | Delete entries by ID, or drop an entire collection. |
| `mem_cleanup` | Remove expired (TTL) entries. Pass `stale_after_days` to also delete entries not accessed within that window (forgetting curve). |
| `mem_train` | Run a training step on the learned re-ranker for one or all collections. |

## Memory Scoping

Collections are auto-detected from the working directory:

| Scenario | Collection |
|---|---|
| `CLAUDE.md` at git root | `repo.<repo-name>` |
| `CLAUDE.md` in monorepo subdir | `repo.<repo-name>.<subdir>` |
| No `CLAUDE.md` found | `workspace` |
| Cross-session general knowledge | `global` |

The `SessionStart` hook injects the active collection on every session start.

## Adaptive Re-ranking

When PyTorch is installed, each collection trains a small MLP `[10 → 32 → 16 → 1]` from access patterns:

- **Future access prediction** — entries accessed again after retrieval score higher (BCE loss)
- **Co-activation** — entries retrieved together in the same query are pulled closer (contrastive loss)
- **Forgetting curve** — entries never accessed can be pruned with `mem_cleanup stale_after_days`

Training is self-supervised: labels are generated automatically from `last_accessed_at` history after a 7-day window. No explicit feedback needed. Without PyTorch the system falls back to ChromaDB's native cosine ranking.

## Hybrid Ranker Mode

Multiple collections can share one trained ranker — useful for related projects (e.g. a microservice cluster):

```json
// ~/.local/share/ai-mem/ranker_config.json
{
  "groups": [
    {
      "name": "work-services",
      "collections": ["repo.payment-svc", "repo.order-svc", "repo.gateway"],
      "mode": "hybrid"
    }
  ]
}
```

Collections not listed are isolated (default). Restart the MCP server after editing.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_MEM_PATH` | `~/.local/share/ai-mem` | Database and ranker weights location |

## Update

```bash
cd ai-mem && git pull
```

No reinstall needed — installed in editable mode.

## Requirements

- Python 3.10+
- PyTorch 2.0+ (optional, for `[ml]` extra)
