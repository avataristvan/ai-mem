# ai-mem

Persistent semantic memory for AI agents. Stores, searches, and retrieves information across sessions using natural language. Backed by [ChromaDB](https://www.trychroma.com/) with an optional learned re-ranker that adapts to your access patterns over time.

Works with **Claude Code**, **Gemini CLI**, and **Cursor**.

> **Beyond memory.** Paired with the `/reflect` ritual and the Dreaming Agent, ai-mem forms an epistemology framework — a system for how agents come to *know* things, not just store them. Agents accumulate structured experience across tasks and projects, enabling a genuine Junior → Senior progression without model changes. [Read the framework docs →](docs/epistemology-framework.md) · [Wire your own skills and agents →](docs/skill-agent-integration.md)

## Workflow

ai-mem is designed around a three-phase task loop:

```
/plan  →  [coding]  →  /reflect
  ↑                        |
  └──── memories updated ←─┘
```

| Phase | What happens | ai-mem role |
|-------|-------------|-------------|
| **Plan** | Review goals, prior decisions, current focus | `mem_query` surfaces relevant context |
| **Code** | Implement the task | — |
| **Reflect** | Capture what was learned, good and bad | `mem_add` stores learnings for the next session |

Run `/reflect` after completing a task. It walks through a lightweight standup: agent observations first, two questions, then next-todo aggregation. The next session starts where this one left off.

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
| `mem_add` | Store or update entries. Set `ttl_days` for automatic expiry. Set `type` for filtering (`feedback`, `project`, `reference`, `pattern`, `anti-pattern`). |
| `mem_query` | Semantic search. Returns re-ranked results with scores. Optionally filter by `type` or `max_age_days`. |
| `mem_list` | List collections with counts, or list all entries in a specific collection. |
| `mem_delete` | Delete entries by ID, or drop an entire collection. |
| `mem_cleanup` | Remove expired (TTL) entries. Pass `stale_after_days` to prune entries not accessed within that window. |
| `mem_train` | Run a training step on the learned re-ranker for one or all collections. |
| `mem_split` | Split a long entry into focused sub-entries for more precise retrieval. |
| `mem_dream` | Consolidate a collection using Claude — detects contradictions, redundancies, and stale entries. |
| `mem_get` | Fetch entries by ID directly — bypasses semantic search and ranking. Use when the exact ID is known. |
| `mem_link` | Create a typed causal edge between two entries (`contradicts`, `fixes`, `causes`, `related`). |
| `mem_edges` | List all outgoing edges for an entry. |

## Memory Scoping

Collections are auto-detected from the working directory:

| Scenario | Collection |
|---|---|
| `CLAUDE.md` at git root | `repo.<repo-name>` |
| `CLAUDE.md` in monorepo subdir | `repo.<repo-name>.<subdir>` |
| No `CLAUDE.md` found | `workspace` |
| Cross-session general knowledge | `global` |

The `SessionStart` hook injects the active collection on every session start.

## Lifecycle Hooks

ai-mem registers four Claude Code hooks automatically during install:

| Hook | Trigger | What it does |
|------|---------|--------------|
| `SessionStart` | Session opens | Injects `current_focus` + active collection routing |
| `UserPromptSubmit` | Before each prompt | Anti-pattern warnings (always) + relevant memories once ranker is trained (≥10 labeled examples, avg score ≥ 0.55) |
| `PreToolUse` | Before Write/Edit | Injects relevant past experiences for the file being touched |
| `PostToolUse` | After Write/Edit | Silent passive training signal — updates `last_accessed_at` on matched entries so the ranker labels them positive |

**Context stays lean by design.** The `UserPromptSubmit` hook queries the active collection against each incoming prompt and injects only the top-3 relevant entries — never the full collection. A film-shoot prompt retrieves brand-voice context; a Kotlin bug prompt retrieves build conventions — automatically, from the same collection. This is the primary answer to context-bloat: not a global dump, but per-prompt semantic selection.

The `PostToolUse` hook is what makes the ranker self-calibrating: every file edit is implicit evidence that the matched memory entries were relevant, with no manual `mem_train` calls required.

## Adaptive Re-ranking

When PyTorch is installed, each collection trains a small MLP `[10 → 32 → 16 → 1]` from access patterns:

- **Future access prediction** — entries accessed again after retrieval score higher (BCE loss)
- **Co-activation** — entries retrieved together in the same query are pulled closer (contrastive loss)
- **Forgetting curve** — entries never accessed can be pruned with `mem_cleanup stale_after_days`

Training is self-supervised: labels are generated automatically from `last_accessed_at` history after a 7-day window. No explicit feedback needed. Without PyTorch the system falls back to ChromaDB's native cosine ranking.

## Typed Causal Edges

Entries can be linked with directional typed edges to model relationships between knowledge:

```
mem_link(source_id="antipattern_xyz", target_id="pattern_abc", edge_type="contradicts", collection="repo.my-project")
```

Edge types: `contradicts` · `fixes` · `causes` · `related`

During `mem_query`, linked entries are automatically surfaced alongside their source (1-hop, budget: 2 entries per query). Appended entries are tagged with `via_edge` and `via_source` in their metadata.

**Primary use case:** link `type=anti-pattern` entries to the `type=pattern` they contradict. When you retrieve a best-practice, the matching anti-pattern surfaces automatically — and vice versa.

```
mem_edges(entry_id="pattern_abc", collection="repo.my-project")
# → [{"target_id": "antipattern_xyz", "edge_type": "contradicts"}]
```

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

### Custom agents (`~/.config/ai-mem/agents.yaml`)

Register your own agents and tune hook parameters:

```yaml
inject_agents:
  - my-coding-agent       # subagents that receive ai-mem context even when spawned

signatures:
  - name: my-coding-agent
    markers:
      - "unique phrase from the agent's system prompt"

settings:
  min_label_score: 0.50   # [0.0–1.0] posttool_hook label threshold
  query_k: 5              # posttool_hook candidate count
```

Requires `pyyaml`: `pip install ai-mem[config]`. Falls back to defaults if absent.

## Architecture

Three-layer capability-centric DDD — imports only flow downward.

| Layer | Path | Role |
|---|---|---|
| Domain | `ai_mem/domain/` | Pure contracts (`MemoryRepository`, `LearnedRanker`, `TrainingBufferRepository` protocols). No I/O, no torch. |
| Application | `ai_mem/application/` | One use case per file, single `execute()`, deps injected. `RankerRegistry` gates `TorchMicroRanker` behind `MIN_LABELED_EXAMPLES=10` — falls back to `NullRanker` until trained. |
| Infrastructure | `ai_mem/infrastructure/` | `ChromaMemoryRepository`, `BM25MemoryRepository` (optional wrapper), `TorchMicroRanker` `[9→32→16→1]` MLP, `NullRanker` UCB fallback, `RankerStorage`. |

`server.py` is the adapter — wires use cases at module load, exposes MCP tools.

## Uninstall

```bash
python3 uninstall.py          # interactive — remove hooks, MCP entry, commands
python3 uninstall.py --dry-run  # preview without changes
```

Memory data (`~/.local/share/ai-mem`) and user config (`~/.config/ai-mem`) are never touched.

## Update

```bash
cd ai-mem && git pull
```

No reinstall needed — installed in editable mode.

## Requirements

- Python 3.10+
- PyTorch 2.0+ (optional, for `[ml]` extra)
