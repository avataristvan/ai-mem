# Friction Points, Research Questions, and Improvement Ideas

*Collected 2026-05-12 — agent perspective on real friction in the current system.*

---

## Real Friction (Agent Perspective)

### 1. Cold-Start Problem
UserPromptSubmit requires ≥10 labeled examples before firing. In the first sessions of a new project, there is no context beyond `current_focus`. This is when the most avoidable mistakes happen.

**Resolution direction:** Expert agent collections (`subagent.<agent_type>`) accumulate cross-project experience over time. Cold start is epistemologically acceptable for a genuinely new agent — the problem is when an agent *could* have had prior experience but starts cold anyway. See `arch_expert_agent_collections`.

### 2. Anti-Pattern Retrieval Comes Too Late ← *priority*
PreToolUse helps right before a file edit. But design decisions happen earlier — in conversation, before any file is touched. That is exactly when "you tried this before and it failed because X" would be most valuable. Currently, the anti-pattern surfaces after the decision has already been made.

**Concrete idea:** Pre-Plan-Hook or `/plan` integration — before the first file edit, explicitly query anti-patterns for the planned approach. "You are planning X — this failed before because Y."

### 3. Retrieval Quality Is Opaque ← *priority*
When `mem_query` returns 3 results, there is no signal about whether those are the right 3 or whether important context is missing. A simple indicator — "ranker is well-calibrated for this collection" vs. "few training examples" — would help calibrate confidence in injected memories.

**Concrete idea:** Ranker-Confidence-Signal at SessionStart or UserPromptSubmit — number of labeled examples, avg_score. The agent would weight injected memories differently based on calibration quality.

---

## Research Questions

### Prospective vs. Retrospective Memory
Anti-pattern entries are *prospective* — "remind me NOT to do X when Y occurs." This is cognitively different from fact recall. Cognitive science research on prospective memory may suggest retrieval strategies optimized for this:

**Hypothesis:** Anti-patterns need higher recall (surface even borderline matches — better to warn unnecessarily), while patterns need higher precision (only surface when clearly relevant). The current system treats both identically.

### Edge Traversal Depth
The current system uses 1-hop traversal with budget=2. This was pragmatically chosen. Research question: is there literature from knowledge-graph completion or graph-based RAG on optimal traversal depth? At what point do 2-hop results introduce noise rather than relevant context?

### Personalized Retrieval Cold Start
A known problem in recommendation systems — the transition from cosine-only to learned ranking without poor early sessions poisoning the training signal. How do recommender systems handle this transition? Could their solutions apply to the MLP ranker?

---

## Concrete Improvement Ideas

| Idea | Impact | Effort | Status |
|------|--------|--------|--------|
| Pre-Plan-Hook / `/plan` integration | High | Medium | Todo |
| Ranker-Confidence-Signal at SessionStart | High | Low | Todo |
| Cross-Session-Delta at SessionStart | Medium | Low | Todo |
| Real-time contradiction detection on `mem_add` | Medium | Medium | Todo |

### Detail: Pre-Plan-Hook
Before the first file edit, query anti-patterns for the planned approach. Could be a `/plan` skill that does `mem_query(type="anti-pattern", query=<plan summary>)` and surfaces warnings before implementation starts.

### Detail: Ranker-Confidence-Signal
At SessionStart or UserPromptSubmit: show labeled example count and avg_score for the active collection. Agent calibrates trust in injected memories accordingly. Low-effort addition to `hook.py` or `userprompt_hook.py`.

### Detail: Cross-Session-Delta
At SessionStart: "since last session — 3 new entries stored, 2 files frequently edited." Accelerates context reconstruction without requiring `mem_query` calls.

### Detail: Real-time Contradiction Detection
On `mem_add`: lightweight check if a new entry contradicts an existing one. Currently only happens during `mem_dream`. A fast cosine similarity check against `type=pattern` entries on anti-pattern storage (auto-link suggestion already does a version of this) could be extended to flag contradictions inline.

---

## Framework Note

The goal — "we both benefit" — is the right frame. The biggest constraint is not recall quality or storage capacity. It is **timing**: the right information at the right moment, *before* a mistake happens, not after.
