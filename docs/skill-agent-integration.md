# Integrating Skills and Agents with ai-mem

This guide explains how to wire Claude Code skills and agents into ai-mem's epistemological framework — so they accumulate cross-project experience instead of starting cold every session.

---

## Core Concepts

### Collection Routing

| Collection | Purpose | Who writes |
|---|---|---|
| `repo.<project>` | Project-specific knowledge — architecture decisions, gotchas, current focus | Hooks (auto-detected), orchestrator, /reflect |
| `subagent.<name>` | Agent's cross-project experience — patterns that apply across codebases | The agent itself, /reflect Step 4.5 |
| `global` | Universal principles — apply regardless of project or agent | /reflect propagation, orchestrator |
| `workspace` | Cross-project but not universal — todos, context that spans projects | Orchestrator |

### Type Tags

| Type | When to use | Mandatory structure |
|---|---|---|
| `pattern` | Transferable rule that held across contexts | `Rule: ...\nWhen: ...\nWhy: ...` |
| `anti-pattern` | Failed approach worth remembering | `Tried: ...\nFailed because: ...\nInstead: ...` |
| `feedback` | Validated approach or correction | Free text — lead with the rule |
| `project` | Time-sensitive fact (use `ttl_days`) | Free text |
| `dilemma` | Genuine value conflict with no single right answer | See mem_add tool description |

### The Learning Loop

```
Session Start → mem_query (load prior experience)
      ↓
Work
      ↓
Session Close → mem_add (store transferable learnings)
      ↓
/reflect → mem_train (label training examples from commits)
```

---

## The Skill Pattern

Skills run in the orchestrator context — they have access to SessionStart-injected data and the full conversation history. Use this pattern for directly user-invokable skills that benefit from session context.

**Minimal skill template:**

```markdown
## Before Every Session

Query relevant prior experience:

\`\`\`
mem_query(collection="<relevant-collection>", query="<task context>", n_results=3)
\`\`\`

Surface anti-patterns, validated approaches, or decisions from prior sessions.
If nothing relevant, proceed without it.

---

[… skill content …]

---

## Session Close

Evaluate: what from this session is worth keeping for next time?

\`\`\`
mem_add(
    documents=[<learning>],
    collection="<relevant-collection>",
    type="feedback",   # or "anti-pattern" / "pattern"
    ids=["<slug>"],
)
\`\`\`

Skip if nothing transferable happened.
```

**Which collection to query?**

- A project-specific skill (e.g. a content strategy tool) → `repo.<project>` + a dedicated session collection (e.g. `repo.<project>-<skill-name>`)
- A general coding or architecture skill → `subagent.<skill-name>` or `global`

**Reference implementation:** `skills/reflect.md` (the /reflect ritual itself)

---

## The Agent Pattern

Agents start cold — no SessionStart injection, no conversation history. Compensate by querying the expert collection explicitly at task start, and writing back at task end.

**Minimal agent template** (add to `~/.claude/agents/<name>.md`):

```markdown
## Session Start — Query Prior Experience

Before planning, query cross-project experience:

\`\`\`
mem_query(collection="subagent.<name>", query="<task summary>", n_results=3)
\`\`\`

If the collection is empty or nothing is relevant, proceed without it.

---

[… agent content …]

---

## Session Close — Store Transferable Learnings

After the task, evaluate: what would apply in any project, not just this one?

Signals worth capturing:
- A fix that corrected a wrong mental model
- An architectural rule that held across layers
- An anti-pattern encountered and resolved

\`\`\`
mem_add(
    documents=[<principle>],
    collection="subagent.<name>",
    type="feedback",   # or "anti-pattern" / "pattern"
    ids=["<name>_<slug>"],
)
\`\`\`

Skip if the session was project-specific with nothing transferable.
```

**The `[retro]` block convention**

When the orchestrator spawns an agent for a non-trivial task, it should request a `[retro]` block at the end of the agent's response:

```
Am Ende deiner Antwort füge einen [retro]-Block hinzu:
[retro]
learned: <was überraschend war — oder "nothing">
blocked: <was dich gehindert hat — oder "nothing">
next: <Vorschlag für nächstes Todo — oder "nothing">
```

The orchestrator reads this block during /reflect (Step 4.5A) and decides whether learnings belong in the agent's expert collection.

**Reference implementation:** `~/.claude/agents/the-coder.md`

---

## The /reflect Safety Net

/reflect (Step 4.5) acts as a fallback extraction step — it surfaces learnings that neither the skill nor the agent wrote themselves:

- **Step 4.5A**: Extracts from `[retro]` blocks of subagents → writes to `subagent.<agent-type>`
- **Step 4.5B**: Asks about transferable principles from direct orchestrator coding → writes to `global` or `subagent.*`

/reflect is the recovery path, not the primary path. Skills and agents should write their own learnings at session close.

---

## Naming Conventions

| Entity | Collection name | Entry ID prefix |
|---|---|---|
| Agent `the-coder` | `subagent.the-coder` | `the-coder_<slug>` |
| Agent `dr-rin` | `subagent.dr-rin` | `dr-rin_<slug>` |
| Skill for project X | `repo.X` or `repo.X-<skill>` | free |
| Universal pattern | `global` | free |

---

## What install.py Sets Up

`python3 install.py` registers:
- The MCP server (makes `mem_add`, `mem_query`, etc. available as tools)
- SessionStart, UserPromptSubmit, PreToolUse, PostToolUse hooks
- `/mem-init` and `/reflect` commands

It does **not** set up your skills or agents — those are local configuration that you own and maintain separately.
