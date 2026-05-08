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
    ids=["feedback_<slug>"],   # or "antipattern_<slug>" for blockers
)
```

Skip if the user has nothing to add.

---

## Step 3.5 — Outcome signal

Silently check: `git log --oneline --since="session start" 2>/dev/null | head -5`

If one or more commits happened during this session:

> **"Diese Session hat einen git-commit produziert — soll ich die abgerufenen Memories als hilfreich markieren?"**

If yes, run a training step on the active collection:

```
mem_train(collection=<active>)
```

This tells the re-ranker that entries retrieved this session contributed to a productive outcome. Over time the ranker learns which entries are associated with sessions that ship.

Skip silently if: no commit found, or user says no.

> **Note:** `mem_train` currently derives labels from the 7-day re-access window. Explicit outcome labels (`label=1.0` per injected entry) are a planned code extension — this step already closes the feedback loop as opt-in convention.

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
