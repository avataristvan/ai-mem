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

## Close

One sentence: what was stored. Example: "2 Learnings, 1 Blocker, 2 Todos, current_focus aktualisiert." Nothing else.

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
