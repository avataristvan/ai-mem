---
name: reflect
description: "Lightweight end-of-session standup. Run after completing a task or todo. Closes the Plan → Code → Reflect loop. Subagents: always append a [retro] block at the end of your response (learned: / blocked: / next:)."
---

Lightweight end-of-session standup. Run after completing a task or todo. Closes the Plan → Code → Reflect loop.

The subagent summary is already the "done" — this adds only what matters beyond the code.

Use the active collection from the session context (e.g. `repo.ai-mem`).

---

## Step 1 — Collect

Scan subagent return values for `[retro]` blocks (see convention below). Note any `learned:`, `blocked:`, and `next:` lines. No user input needed yet.

---

## Step 2 — My contributions first

Before asking the user, contribute your own observations:

- **Learned:** what was surprising, what assumption was wrong, what tool failed silently
- **Blocked:** what got in the way — missing tool, unclear requirement, wrong mental model
- **Next:** one or two todo suggestions based on what you saw

Flag each as **"Meine Beobachtung:"** so the user knows it's from the agent side.

---

## Step 3 — Two questions

Ask both questions together:

> **"Was hast du gelernt oder was hat dich überrascht?"**
> **"Was hat dich gehindert?"**

Wait for the user's answer. Store each learning or blocker immediately:

```
mem_add(
    documents=[<learning or blocker>],
    collection=<active>,
    type="feedback",
    ids=["feedback_<slug>"],
)
```

If a blocker describes a failed approach, store it as an anti-pattern instead:

```
mem_add(
    documents=["Tried: <approach>\nFailed because: <reason>\nInstead: <alternative>"],
    collection=<active>,
    type="anti-pattern",
    ids=["antipattern_<slug>"],
)
```

The `Tried:` field makes the entry self-retrieving — future queries about the same approach will surface it automatically without any special retrieval logic.

Skip if the user has nothing to add.

---

## Step 3.5 — Feedback signal

**Outdated or misleading entries (always, no commit needed)**

Review the conversation: were any retrieved memories outdated, misleading, or contradicted by reality?

For each: reform in place as anti-pattern — the content change is the signal, no ranker label needed:

```
mem_add(
    documents=["Tried: <original claim>\nOutdated/wrong because: <reason>\nInstead: <what is true now>"],
    ids=[<original_id>],
    collection=<active>,
    type="anti-pattern",
)
```

Completely wrong with no useful lesson → `mem_delete` instead.

Skip silently if nothing was outdated.

---

**Positive signal**

Silently check: `git log --oneline --since="session start" 2>/dev/null | head -5`

If commits found: `mem_train(collection=<active>)` — no user confirmation needed.

Skip silently if no commit was found.

Note in the Close summary if mem_train ran or if entries were reformed.

---

## Step 4 — Next todos

Aggregate next-todo suggestions:
- Your own (from Step 2)
- Any `next:` lines from `[retro]` blocks
- The user's ideas (if any from Step 3)

Present them as a short list. Do not store automatically — ask which ones to keep:

> **"Welche davon sind relevant?"**

Store confirmed todos:

```
mem_add(
    documents=[<todo>],
    collection=<active>,
    type="project",
    ttl_days=14,   # optional, adjust per todo
)
```

---

## Step 4.5 — Expert propagation (only if subagents produced [retro] blocks)

For each subagent that contributed a `learned:` line in Step 1, decide: is this learning about **how to approach a problem** (technique, process, pattern) — or about **what was done in this project** (specific decision, project fact)?

- Process/technique → candidate for the expert collection
- Project-specific → stays in the project collection only

Ask for each candidate:

> **"Soll dieses Learning für `<agent-type>` cross-projekt gespeichert werden?"**

If yes:

```
mem_add(
    documents=[<learning text>],
    collection="subagent.<agent-type>",   # e.g. "subagent.the-coder"
    type="feedback",
    ids=["<agent-type>_<slug>"],
)
```

The expert collection accumulates across all projects. Over time it gives the agent genuine cross-project intuition — the same way a senior engineer recognizes patterns they've seen before, regardless of the specific codebase.

Skip entirely if no subagents were used or no `[retro]` blocks were found.

---

## Step 5 — Update current_focus

Ask: **"Was ist dein nächster Fokus?"** (or derive from the confirmed todos if obvious).

```
mem_add(
    documents=[<focus>],
    ids=["current_focus"],
    collection=<active>,
)
```

---

## Step 6 — Consolidation and propagation

After storing, silently run `mem_list` on the active collection and count entries.

**Consolidation:** If entry count **> 30**, or if `mem_query` returned split-hint warnings in this session:

> **"Die Collection hat X Einträge — soll ich das Gedächtnis konsolidieren? (`mem_dream`)"**

If yes, run dream in `hier` mode to see across all collections at once:

```
mem_dream(mode="hier", auto_apply=false)
```

Review the proposal with the user:
- DELETE / UPDATE / MERGE proposals → apply to the source collection with `mem_add` / `mem_delete`
- **ADD proposals where `target_collection` is higher than the source** (e.g. `workspace` or `global`) → these are **propagation candidates**: a pattern that proved universal across projects

For each propagation candidate, ask:

> **"Dieses Learning taucht in mehreren Projekten auf — soll es nach `[target_collection]` hochgestuft werden?"**

If yes:

```
mem_add(
    documents=[<entry text>],
    collection=<target_collection>,   # e.g. "workspace" or "global"
    type=<same type as source>,
)
```

The entry stays in the source collection too — propagation copies upward, it does not move.

If count ≤ 30 and no split hints: skip silently.

---

## Close

One sentence: what was stored + whether dream ran + whether anything was propagated. Example: "2 Learnings, 1 Blocker, 2 Todos, current_focus aktualisiert." or "… + 1 Learning nach `global` hochgestuft." Nothing else.

---

## [retro] convention for subagents

When spawning subagents for non-trivial tasks, add this to the end of their prompt:

> Am Ende deiner Antwort füge einen `[retro]`-Block hinzu:
> ```
> [retro]
> learned: <was überraschend war oder nicht funktioniert hat — oder "nothing">
> blocked: <was dich gehindert hat — oder "nothing">
> next: <Vorschlag für ein nächstes Todo — oder "nothing">
> ```

The main agent reads this block in Step 1 and surfaces it in Step 4. This makes subagent observations (silent fallbacks, tool failures, dead ends) visible without requiring transcript review.
