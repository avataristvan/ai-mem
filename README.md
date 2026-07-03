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

> **First steps:** Run `/mem-init` in your project directory, then `/reflect` after every task.
> Storage alone doesn't make ai-mem useful — the reflect ritual is what closes the loop.

## Tools

| Tool | Description |
|------|-------------|
| `mem_add` | Store or update entries. Set `ttl_days` for automatic expiry. Set `type` for filtering (`feedback`, `project`, `reference`, `pattern`, `anti-pattern`). |
| `mem_query` | Semantic search. Returns re-ranked results with scores. Optionally filter by `type` or `max_age_days`. |
| `mem_list` | List collections with counts, or list all entries in a specific collection. |
| `mem_delete` | Delete entries by ID, or drop an entire collection. |
| `mem_cleanup` | Remove expired (TTL) entries. Pass `stale_after_days` to prune entries not accessed within that window. |
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
| `UserPromptSubmit` | Before each prompt | Anti-pattern warnings (always) + relevant memories when cosine score ≥ 0.3 |
| `PreToolUse` | Before Write/Edit | Injects relevant past experiences for the file being touched |
| `PostToolUse` | After Write/Edit | Updates `last_accessed_at` and `access_count` on matched entries |

**Context stays lean by design.** The `UserPromptSubmit` hook queries the active collection against each incoming prompt and injects only the top-3 relevant entries — never the full collection. A film-shoot prompt retrieves brand-voice context; a Kotlin bug prompt retrieves build conventions — automatically, from the same collection. This is the primary answer to context-bloat: not a global dump, but per-prompt semantic selection.

## Confidence Lifecycle

Each entry carries a `confidence` score (0.0–1.0) that tracks its epistemic status over time:

| Event | Effect |
|-------|--------|
| New entry | Starts at **0.7** — must earn its place |
| `/reflect` confirms it was decisive | `mem_boost(delta=+0.1)` |
| Dream cycle flags it as stale | `mem_boost(delta=-0.1)` |
| `confidence > 0.9` + `access_count ≥ 3` | Injected at every session start as **[always-present]** |
| `confidence < 0.3` | Flagged as decay candidate in `mem_dream` report |

Retrieval uses ChromaDB's native cosine similarity. Confidence governs *which entries are always available*, not result ordering.

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

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_MEM_PATH` | `~/.local/share/ai-mem` | Database and ranker weights location |
| `AI_MEM_WORKSPACE_ROOT` | unset (git-based detection) | Root directory used to name collections by folder structure instead of git remotes, e.g. `~/workspace` → `ExoDeck-project/filming` → `repo.ExoDeck-project.filming`. Set interactively during `install.py`, or manually in the `env` block of the `ai-mem` entry in `~/.claude.json`. |

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
| Domain | `ai_mem/domain/` | Pure contracts (`MemoryRepository` protocol). No I/O. |
| Application | `ai_mem/application/` | One use case per file, single `execute()`, deps injected. |
| Infrastructure | `ai_mem/infrastructure/` | `ChromaMemoryRepository`, `BM25MemoryRepository` (optional wrapper). |

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
