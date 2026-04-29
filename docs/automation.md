# Automation Pipeline

These three subsystems automate memory hygiene so neither user nor agent needs to make manual judgments.

## 1. MEMORY.md Auto-Demotion (`memory_index.py`)

**Trigger:** Stop hook calls `auto_demote()` silently on every session end.

**Logic:**
- Threshold: 180 lines — demote when MEMORY.md reaches this
- Target: 160 lines after demotion
- Candidates sorted by: type priority (project first, then reference, user, feedback) × mtime (oldest first)
- Routing: `project` type entries → repo collection; all others → `global`
- Dangling entries (file missing) are removed from the index without adding to ai-mem
- On add failure, entry stays in the index (no data loss)

**Constants:**
```python
DEMOTION_THRESHOLD = 180  # demote when MEMORY.md reaches this many lines
_TARGET_LINES = 160       # bring index down to this after demotion
_TYPE_PRIORITY = {"project": 0, "reference": 1, "user": 2, "feedback": 3}
```

`find_memory_dir()` resolves `~/.claude/projects/<encoded-cwd>/memory/` — returns `None` if not found.

## 2. Session Injection Stats (`session_stats.py`)

Tracks whether the SessionStart hook successfully injected context, per scope.

**Rolling window:** 20 sessions per scope

**Functions:**
- `record_injection(stats_path, scope, injected: bool)` — append + trim to 20
- `injection_rate(stats_path, scope) → float` — `injected_count / total` (0.0 if no data)

**Storage:** `{DB_PATH}/session_stats.json` — `{scope: [{ts: float, injected: bool}]}`

**Who records:**
- `hook.py` (SessionStart): records for global scope before `_try_seed`
- `userprompt_hook.py`: records after each injection decision

## 3. Repo Seeder (`repo_seeder.py`)

Seeds a new `repo.*` collection from CLAUDE.md on the first session.

**Gate:** `injection_rate(stats_path, "global") >= SEED_THRESHOLD (0.60)`
- Ensures seeding only activates once memories are actively being used

**Section splitting:** Only H2 sections (`## ` prefix); H1 intro block intentionally excluded.

**Entry IDs:** `seed_<slug>` where slug = lowercase + underscores + alphanumerics only.

**Metadata:** `{seeded_from: "CLAUDE.md", seed_ts: <unix float>}`

**Guard:** `_try_seed` in `hook.py` checks `collection count == 0` (or not listed) before seeding; wraps everything in try/except → always silent.

## 4. UserPromptSubmit Auto-Injection (`userprompt_hook.py`)

Proactively injects relevant memories before each user prompt when the ranker has learned enough.

**Activation (per scope, OR-combined across scopes):**
- `labeled_count >= 10` examples in the training buffer
- `avg_top_3_score >= 0.55`

**Why threshold-gated:** Early in a project's life the ranker isn't trained yet; injecting low-confidence results would add noise. The thresholds ensure injection only starts once the system has evidence it's useful.

**Output format:**
```
[ai-mem] Relevant context for your prompt:
- [repo.my-project score=0.82] <snippet up to 600 chars>
- [global score=0.71] <snippet>
```
