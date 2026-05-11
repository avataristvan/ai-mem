# ai-mem vs. the Field — Competitive Landscape

Last updated: 2026-05-11. Based on direct research of active Claude Code memory tools.

---

## The Tools

### MemPalace

Hierarchical verbatim memory. Wings (person/project) > Rooms (topics) > Drawers (content). Backend: ChromaDB (pluggable). **Nothing is summarized** — all content is preserved verbatim.

- Hybrid retrieval: vector + keyword + temporal proximity. Claims 98.4% on LongMemEval.
- Knowledge graph with temporal entities (SQLite).
- 29 MCP tools.
- Two hooks: periodic save + pre-compression save before context limit.
- Local embeddings (300 MB, no API key needed).
- Agent isolation: each specialist agent has its own Wing + Diary.

**Philosophy:** Verbatim preservation + high recall. The agent reconstructs sessions from stored snapshots.

### claude-mem (thedotmack)

The most active project in the field (v12.6.4, 109 contributors). Dual-DB: SQLite (FTS5, session metadata, processing queue) + ChromaDB (semantics). Five lifecycle hooks including PostToolUse — every tool use is queued for processing. AI compression before re-injection. 4 MCP query tools.

**Philosophy:** Capture everything, compress before injecting. Optimizes for breadth.

### mcp-memory-service (doobidoo)

Local knowledge graph. Backends: SQLite-vec / Milvus / Cloudflare. Local ONNX embeddings. Typed-edge knowledge graph (`causes`, `fixes`, `contradicts`). Hybrid retrieval via Reciprocal Rank Fusion. Plugin hooks for developers. Autonomous consolidation via local LLM (Ollama/LiteLLM).

**Philosophy:** Relationship modeling. The only other tool that models causal edges between entries.

### OB1 (NateBJones-Projects)

Universal memory infrastructure, not Claude-specific. Backend: Supabase (PostgreSQL + pgvector). Schema-aware routing distributes unstructured text across tables via LLM. Integrates Slack, Gmail, Discord. No lifecycle hooks, no session concept — pure data layer.

**Philosophy:** Unified heterogeneous source aggregation. Belongs to shared infrastructure, not agent memory.

---

## Feature Matrix

| Dimension | **ai-mem** | MemPalace | claude-mem | mcp-memory-service | OB1 |
|---|---|---|---|---|---|
| Vector store | ChromaDB | ChromaDB (pluggable) | ChromaDB + SQLite | SQLite-vec / Milvus | PostgreSQL + pgvector |
| Hybrid retrieval | BM25 + cosine | Vector + keyword + temporal | FTS5 + cosine | RRF (BM25 + vector) | Vector only |
| Adaptive ranker | TorchMicroRanker (MLP) | — | — | — | — |
| Lifecycle hooks | 5 (incl. PostToolUse) | 2 | 5 | Plugin hooks (dev) | — |
| Consolidation | mem_dream (Claude API) | — | AI compression | Local LLM | — |
| Memory hierarchy | global → workspace → repo | Wings > Rooms > Drawers | Session-centric | tags/entities | Table schema |
| Verbatim storage | No (abstracted) | Yes (core feature) | No (compressed) | No | No |
| Causal edge graph | Yes (contradicts/fixes/causes/related) | Temporal entities only | — | Typed-edge graph | — |
| Type tags + TTL | Yes (user/feedback/project/reference/pattern/anti-pattern) | — | — | stale_days | — |
| Anti-pattern support | Yes (with TTL exemption) | — | — | — | — |
| Local embeddings | No (ChromaDB model) | Yes (300 MB) | No | Yes (ONNX) | No |
| No API key needed | No (mem_dream uses Claude) | Yes | No | Yes | No |

---

## Honest Assessment

**Where ai-mem is stronger:**

- **Adaptive ranker** — the only tool that learns which entries *you* actually find useful, and reweights retrieval accordingly. Nobody else does this.
- **Type ontology + anti-pattern support** — `user/feedback/project/reference/pattern/anti-pattern` is the richest semantic categorization in the field. TTL-exempt permanent types (pattern, anti-pattern) prevent accidental cleanup of structural knowledge.
- **mem_dream** — active consolidation via Claude API produces genuine editorial decisions (merge, delete, propagate), not just compression. MemPalace has nothing; claude-mem compresses passively.
- **Plan-Code-Reflect cycle** — the `/reflect` ritual + `current_focus` + Stop hook formalize a learning loop that no other tool makes explicit.
- **Causal edges** — typed directed relationships between entries. Only mcp-memory-service has something comparable.

**Where others are stronger:**

- **Recall accuracy** — MemPalace's 98.4% on LongMemEval is a real measurement against a real benchmark. ai-mem has no published benchmark.
- **Community maturity** — claude-mem at v12.6.4 with 109 contributors has been broken in real-world edge cases and fixed. ai-mem has not.
- **Local embeddings** — MemPalace and mcp-memory-service work entirely offline. ai-mem requires ChromaDB's embedding model and mem_dream requires the Claude API.
- **Verbatim session reconstruction** — MemPalace can reconstruct prior sessions accurately. ai-mem's abstraction approach loses verbatim fidelity by design.

---

## Philosophical Divergence

The central trade-off in this field is **abstraction vs. verbatim**:

| Approach | Tools | Strength | Weakness |
|---|---|---|---|
| Verbatim | MemPalace | High recall, reconstructable | Context-hungry, no editorial layer |
| Abstracted | ai-mem, claude-mem | Compact, editable, rankable | Lossy — nuance can be lost |

ai-mem chooses abstraction deliberately: distilled knowledge over session transcripts. The bet is that a well-organized entry about *what was learned* is more useful than a verbatim record of *what happened* — especially over long project horizons.

This is a philosophical choice, not a technical limitation. Both approaches are valid for different use cases: MemPalace for session reconstruction, ai-mem for accumulating project expertise across months.

---

## Ideas Adopted from Competitors

- **PostToolUse passive training signal** — inspired by claude-mem's aggressive PostToolUse capture. ai-mem's version is lighter: instead of queuing all tool uses for compression, it simply queries memory with the edited file path to update `last_accessed_at`, letting the 7-day label window do the work.
- **Causal edge graph** — inspired by mcp-memory-service's typed-edge knowledge graph. ai-mem's implementation stores edges as JSON-string in ChromaDB metadata (no separate graph layer), keeping the infrastructure footprint minimal.
