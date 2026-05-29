# skills/

This directory holds Claude Code skill files (`.md`) that integrate with ai-mem.

Skills placed in `~/.claude/skills/` (or your Claude Code skills directory) become
slash commands. Skills in this directory are **not auto-installed** — copy or symlink
the ones you want.

## examples/

Reference skills that demonstrate ai-mem integration patterns:

| File | Description |
|---|---|
| `reflect.md` | End-of-session standup: collect learnings, store to ai-mem, update focus |

These are starting points. Adapt them to your workflow, language, and agent setup
before installing.

## Writing your own skills

A skill file is a Markdown file with YAML frontmatter:

```markdown
---
name: my-skill
description: "One-line description shown in /help"
---

Instructions for the agent...
```

Skills can call any ai-mem MCP tool (`mem_add`, `mem_query`, `mem_train`, etc.).
Use the active collection from the session context rather than hardcoding one.
