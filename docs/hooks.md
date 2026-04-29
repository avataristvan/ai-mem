# Hook System

ai-mem registers four Claude Code lifecycle hooks. All hooks are silent on failure (wrapped in try/except) so they never interrupt the session.

## SessionStart (`hook.py`)

Fires at the start of every Claude Code session.

**What it does:**
1. Detects the repo context (git root + CLAUDE.md presence → collection name)
2. Fetches `current_focus` entry from the repo collection and global collection
3. Records a global injection stat (`record_injection`) regardless of whether focus was found
4. If the repo collection is empty (count == 0) and global `injection_rate >= 0.60`, calls `_try_seed` to populate from CLAUDE.md H2 sections
5. Prints `hookSpecificOutput` JSON with `additionalContext` containing focus entries + active collection routing hint

**Output format:**
```
[ai-mem]
[<scope_name> focus]
<repo focus text>

[global focus]
<global focus text>

Active collection: "repo.<name>". Pass collection="repo.<name>" to mem_add and mem_query in this session.
```

If no parts are generated, the hook exits silently (no output).

## SessionStop (`stop_hook.py`)

Fires at the end of every Claude Code session.

**What it does:**
1. `_try_demote()` — silently runs MEMORY.md auto-demotion (see `memory_index.py`)
2. Checks `git diff --name-only HEAD` for changed files
3. If files changed: prints a reminder to update `current_focus` and exits with code 2 (signals Claude to show the message)

Exit code 2 causes Claude Code to surface the reminder text to the agent.

## UserPromptSubmit (`userprompt_hook.py`)

Fires before each user prompt is processed.

**Activation thresholds** (both must pass for a scope):
- `labeled_count >= 10` — at least 10 labeled training examples in the buffer
- `avg_top_3_score >= 0.55` — average score of top-3 results is high enough

Checks global scope and repo scope independently; either qualifying triggers injection (OR logic).

**What it does (when qualified):**
1. Queries top-3 results from global + repo collections using the prompt text
2. Formats results as `[collection score=X.XX] <text snippet (≤600 chars)>`
3. Prints `hookSpecificOutput` JSON with `UserPromptSubmit` event
4. Records injection stat

**Constants:** `TOP_K = 3`, `MAX_CHARS_PER_HIT = 600`, `MIN_LABELED_EXAMPLES = 10`, `MIN_AVG_SCORE = 0.55`

## PreToolUse (`pretool_hook.py`)

Fires before each tool call. Injects brief relevant context from memory to inform tool decisions.

---

## Hook Registration

Hooks are registered in `~/.claude/settings.json` by `install.py`. Each hook has a timeout (SessionStart: 10s, Stop: 10s, UserPromptSubmit: 8s).

```json
{
  "hooks": {
    "SessionStart": [{"matcher": "", "hooks": [{"type": "command", "command": "python3 -m ai_mem.hook", "timeout": 10000}]}],
    "Stop": [{"matcher": "", "hooks": [{"type": "command", "command": "python3 -m ai_mem.stop_hook", "timeout": 10000}]}],
    "UserPromptSubmit": [{"matcher": "", "hooks": [{"type": "command", "command": "python3 -m ai_mem.userprompt_hook", "timeout": 8000}]}]
  }
}
```
