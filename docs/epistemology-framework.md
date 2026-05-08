# The Epistemology Framework

ai-mem is a memory tool, but memory alone does not produce knowledge. What turns stored entries into genuine agent expertise is the methodology around it. Together they form an **epistemology framework** — a system that determines how an agent comes to know things, not just how it stores them.

> **Implementation note:** This document distinguishes between what is code-backed and what is convention. Sections marked *[convention]* describe workflow patterns enforced by agent prompts and Skill definitions, not by code constraints. Sections marked *[planned]* describe intended capabilities not yet implemented.

---

## Foundational Premise: Agents Do Not Know Where They Are

A Claude agent has no intrinsic awareness of its own position in a multi-agent hierarchy. It does not know whether it is the top-level Orchestrator, a subagent spawned for a subtask, or a temporarily embedded Expert. Without explicit context, it will not behave accordingly.

`agent_context.py` detects hierarchy position at the **hook level** — via depth counters and signature matching — and uses this to gate context injection. But this is external detection, not agent self-awareness. The agent reasoning inside Claude never receives "you are the Orchestrator" as a ground truth — it receives it only if the prompt says so.

This gap is the origin of the framework. Every convention in the system — the Orchestrator-as-central-writer pattern, `[retro]` blocks, collection routing, the `/reflect` ritual — exists to **provide hierarchy context explicitly** rather than assuming the agent will derive it on its own.

**Practical implication:** The convention layer is not optional workflow polish. It is structural scaffolding that gives agents a role, a scope, and a persistent identity they would otherwise lack. Removing or weakening the conventions does not simplify the system — it removes the only mechanism by which agents know who they are.

### Transparency as a Feature

The convention-based approach has a benefit that full automation cannot offer: **mutual visibility**.

When the `/reflect` ritual surfaces agent observations, subagent `[retro]` blocks, and user feedback in a shared exchange, both sides learn from the other in that moment. The agent sees what the user considers important; the user sees what the agent noticed that they missed. This bilateral exchange is not overhead — it is a second learning signal on top of what gets stored in ai-mem.

A fully automated system (hooks firing silently, labels assigned without confirmation, dream running in the background) would close the feedback loop technically but break this exchange. The agent would learn from patterns; the user would not learn from the agent.

The deliberate pace of the convention layer is therefore not a limitation of the current implementation — it is a design choice that keeps the human in the loop as an active participant, not just a source of ground-truth labels.

> **The core trade-off:** Automation would give the agent more signals. Convention gives it better ones. The framework optimizes for accuracy, not efficiency.

---

## The Three Pillars

### 1. Empirical Methodology — the Taskly *[convention]*

After every task, the agent runs `/reflect` (the Taskly ritual). It is structured self-observation modeled after Agile retrospectives:

- Subagents append a `[retro]` block (`learned:`, `blocked:`, `next:`) at the end of their response
- The orchestrating agent contributes its own observations first
- The user adds ground-truth feedback
- The agent writing the reflect output stores results via `mem_add`

This is the input side. Raw experience becomes structured entries.

**What the code does:** The `/reflect` Skill (`skills/reflect.md`) defines this ritual as a five-step prompt sequence. `[retro]` blocks are read and interpreted by the LLM at runtime — there is no code parser. The convention that the Orchestrator (and not individual subagents) is the sole writer to `mem_add` is a workflow rule, not a technical constraint.

### 2. Deterministic Persistence — ai-mem

The tool provides reliable storage and retrieval:

- **ChromaDB** for semantic vector search
- **BM25 hybrid layer** (optional) — fuses cosine + BM25 scores over 50 candidates
- **Adaptive MLP re-ranker** `[11→32→16→1]` that learns from access patterns over time
- **TTL + forgetting curve** — `mem_cleanup` with `stale_after_days` removes entries not accessed within a window
- **Typed entries** (`feedback`, `project`, `reference`, `user`) for filtered retrieval

The re-ranker is the implicit learning signal: entries that get retrieved and accessed again within a 7-day window receive a positive label. BCE loss + 0.3× contrastive loss. No explicit feedback labels needed. Without PyTorch, the system falls back to cosine ranking.

**Split-hint detection** is also code-backed: entries with `access_count ≥ 5` and `text ≥ 150 chars` are flagged in every `mem_query` response as candidates for splitting.

### 3. Dreaming Agent — consolidation

Periodically (currently triggered manually by the Navigator), the Dreaming Agent (`mem_dream`) runs. It calls a Claude model to produce a structured proposal:

- **UPDATE** — rewrite an entry for clarity
- **MERGE** — combine two related entries
- **DELETE** — remove redundant or stale entries (auto-applied when `auto_apply=true`)
- **ADD** — propose a new generalized entry

`mem_dream` operates within a single collection or across all collections (`mode=hier`). It also supports a `team` mode for cross-agent consolidation.

**What `mem_dream` does not do automatically:**

- It does not split large entries — splitting is a separate tool (`mem_split`) triggered independently

**Trigger placement:** `mem_dream` is invoked as part of the `/reflect` ritual (convention), not via a hook. Hook-based auto-triggering would fire for every subagent session, not just the orchestrator — the same structural problem that caused the Stop-hook to be retired. Keeping the trigger in `/reflect` ensures it runs exactly once per task cycle, always at the orchestrator level, always deliberately.

**Upward propagation** is handled during the same Step 6 of `/reflect`. When `mem_dream` runs in `hier` mode it sees all collections simultaneously. ADD proposals that target a higher-level collection than their source (e.g. a `repo.*` pattern proposed for `workspace`) are surfaced as propagation candidates. The Orchestrator confirms; `mem_add` copies the entry into the target collection. The source entry is kept — propagation copies upward, it does not move. This keeps the decision in the agent's judgment rather than in code, consistent with the rest of the convention layer.

**Splitting** uses `SplitMemoryUseCase`: deletes the original entry and replaces it with 2–3 focused sub-entries via a Claude call. Split-hints from `mem_query` indicate when an entry is a candidate.

The Dreaming Agent is the bridge between raw empiricism and structured knowledge — analogous to sleep-based memory consolidation. The automatic, threshold-triggered version of this loop is in active development.

---

## The Knowledge Hierarchy

Collections are organized by scope. Two levels are auto-detected from the working directory:

```
global                   ← cross-project, manually managed
  └── workspace          ← fallback when no git repo is found
        └── repo.<name>  ← auto-detected from CLAUDE.md + git remote
```

The full intended hierarchy (partially implemented):

```
global
  └── workspace          (team / department)
        ├── repo.<name>   (project — auto-detected)
        │     └── subagent  (individual agent — convention only, not auto-detected)
        └── [expert agents]  (cross-project — convention only, no schema enforcement)
```

The `subagent` level and `expert.*` collections can be created manually by naming them accordingly. The `SessionStart` hook (`hook.py`) injects the active repo collection and `current_focus`; it also detects whether it is running inside a subagent via `agent_context.py`, but does not automatically create a separate subagent-scoped collection.

A learning starts at the level where it was experienced. Upward propagation — from `repo.*` to `workspace` to `global` — is a design goal; the Dreaming Agent currently identifies candidates but does not execute the cross-collection move.

---

## The Junior → Senior Effect

This is the central claim of the framework:

> An agent that runs Tasklies accumulates structured experience faster than any individual can consciously track. Over time it becomes more competent in its domain — not because it runs a larger model, but because it has better-organized knowledge.

The progression:

| Stage | What the agent has |
|---|---|
| Junior | Large model, no domain memory |
| Mid-level | Memory of what was done |
| Senior | Memory of what worked, what failed, and in which context — organized by the Dreaming Agent |

The key distinction from simple retrieval systems: a senior does not just remember more — they remember *better organized* knowledge. The re-ranker + Dreaming Agent provide that organization over time.

---

## Agent Teams *[convention]*

The framework extends to multi-agent teams. These patterns are enforced by agent prompts and Orchestrator conventions, not by code.

### Team composition

Each agent has three knowledge layers:
- **Team knowledge** — shared expertise of the group (`workspace` collection)
- **Project knowledge** — context for the current engagement (`repo.<name>` collection)
- **Personal knowledge** — individual profile: role, personality, accumulated experience (named by convention, e.g. `subagent.<role>`)

### Expert agents

Expert agents are specialists with their own cross-project collections. They accumulate expertise across every team they join.

When called into a team:
1. They receive full context: team knowledge + project knowledge + task
2. They work as a full team member for the duration
3. Their `[retro]` feedback flows to the Orchestrator
4. The Orchestrator writes relevant learnings into both the team collection and the expert's own collection
5. The expert leaves with new experience; the team retains what the expert contributed

This is the **guest professor model**: the expert teaches while learning, and both sides are stronger after the engagement. Collection naming and routing are enforced by the Orchestrator prompt, not by code.

### The Orchestrator

By convention, the Orchestrator is the central writer to `mem_add`. Individual subagents surface learnings via `[retro]` blocks; the Orchestrator decides what gets stored where.

```
All agents → [retro] → Orchestrator
                           ↓
                     evaluates + routes
                           ↓
              mem_add (team / expert / project)
```

---

## The Full Loop

```
Task
  ├── subagents produce [retro] blocks     [convention]
  └── /reflect (Taskly)                   [convention]
          ↓
     Orchestrator aggregates               [convention]
          ↓
       mem_add
     (feedback / antipattern / project / current_focus)
          ↓
     Reranker learns from access patterns  [code, implicit]
          ↓
   /reflect Step 3.5: git-commit check     [convention]
     → mem_train (explicit outcome signal)
          ↓
   /reflect Step 6: mem_dream hier         [convention]
     → compresses, merges, deletes
     → ADD proposals with higher target    → propagation candidates
          ↓
     Orchestrator confirms propagation     [convention]
          ↓
     mem_add to workspace / global         [code]
          ↓
   global / workspace / repo.<name>
```

---

## Comparison: OB1 + OpenClaw vs. ai-mem framework

| Dimension | OB1 + OpenClaw | ai-mem framework |
|---|---|---|
| Who drives memory? | The user (weekly review, manual capture) | The agent (Taskly after every task) |
| Feedback loop | None — capture only | Reranker (implicit) + Taskly (explicit) |
| Consolidation | None | Dreaming Agent (manual trigger today, auto-trigger planned) |
| Knowledge hierarchy | Flat collections | global → workspace → repo (subagent: convention) |
| Expert agents | Not modeled | Convention-based pattern, no code schema |
| Junior → Senior effect | Not achievable structurally | Built into the loop (reranker + dream) |

OB1 is a **capture tool**. The ai-mem framework is a **learning system**. The difference is not the storage backend — it is whether there is a feedback loop that closes.

---

## What This Is Not

- Not a model fine-tuning approach — the model weights do not change
- Not a RAG system — there is no document corpus, only structured experience entries
- Not a rule engine — rules emerge from reflection, they are not hand-coded

The framework is epistemological: it concerns how knowledge is acquired and structured, not what the knowledge is about. It applies equally to coding agents, writing teams, marketing teams, and executive decision-making groups.

---

## Implementation Status Summary

| Feature | Status |
|---|---|
| ChromaDB storage, TTL, typed entries | Implemented |
| BM25 hybrid retrieval | Implemented |
| Adaptive MLP re-ranker (`[11→32→16→1]`) | Implemented |
| Split-hint detection | Implemented |
| `mem_split` (manual) | Implemented |
| `mem_dream` (manual trigger) | Implemented |
| Outcome signal via git-commit in `/reflect` Step 3.5 | Convention — `mem_train` triggered when session produced a commit |
| Explicit `label=1.0` per injected entry in `mem_train` | Planned code extension |
| `mem_dream` trigger via `/reflect` ritual | Convention (by design — hook placement causes subagent chain problem) |
| Upward propagation via `/reflect` Step 6 | Convention — `mem_dream hier` surfaces candidates, Orchestrator confirms |
| Auto-split via Stop-hook | Planned |
| `/reflect` Taskly ritual | Convention (Skill prompt) |
| `[retro]` block parsing | Convention (LLM reasoning) |
| Orchestrator-only write constraint | Convention |
| Expert-agent collection schema | Convention |
| `subagent.*` auto-detection | Not implemented |
