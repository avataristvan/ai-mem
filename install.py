#!/usr/bin/env python3
"""ai-mem installer — registers the MCP server with Claude Code, Gemini CLI, and/or Cursor."""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.resolve()
HOME = Path.home()
SERVER_MODULE = "ai_mem.server"


# ── helpers ──────────────────────────────────────────────────────────────────

def run(cmd: list[str], check=True):
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=check)
    return result.returncode == 0


def patch_json(path: Path, updater):
    """Read JSON, apply updater(dict) → dict, write back."""
    data = json.loads(path.read_text()) if path.exists() else {}
    data = updater(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def python_exe() -> str:
    return sys.executable


# ── installation steps ────────────────────────────────────────────────────────

def install_package():
    print("\n📦 Installing ai-mem package...")
    run([python_exe(), "-m", "pip", "install", "-e", str(REPO_ROOT), "--quiet"])


def register_claude():
    """Add ai-mem MCP server + SessionStart hook to ~/.claude.json and ~/.claude/settings.json."""
    # MCP server → ~/.claude.json
    mcp_path = HOME / ".claude.json"
    print(f"\n🤖 Registering with Claude Code ({mcp_path})...")

    INSTRUCTIONS = (
        "ai-mem is a persistent semantic memory store for AI agents. "
        "When these tools are available, ai-mem is installed and active. "
        "\n\n"
        "THE LEARNING LOOP — ai-mem is not just storage. It is the persistence layer of an epistemology framework: "
        "a system for how an agent accumulates structured experience and becomes more effective over time. "
        "The loop is: Plan (query relevant context) → work → Reflect (store learnings, blockers, todos). "
        "Run /reflect after every task to close the loop. "
        "Run /mem-init to initialize memory for a new project scope. "
        "Over time, the adaptive re-ranker learns which entries matter for which queries — "
        "the agent becomes more competent not by using a larger model, but through better-organized knowledge. "
        "\n\n"
        "MEMORY SCOPING: Memory is scoped to the nearest CLAUDE.md. "
        "Use the repo collection injected at session start (e.g. 'repo.my-project') for project-specific context, "
        "'global' for cross-repo knowledge that should surface everywhere, "
        "and 'workspace' as the fallback when no CLAUDE.md is present. "
        "\n\n"
        "WHAT TO MEMORISE — store and retrieve these proactively:\n"
        "1. Current focus / session handoff: what is being worked on right now (id='current_focus'). "
        "Update this at the end of every productive session so the next session starts with context.\n"
        "2. Architectural decisions & rationale: why a design was chosen, what was rejected and why. "
        "Query before making architectural choices to avoid re-litigating settled decisions.\n"
        "3. Recurring gotchas & tribal knowledge: API quirks, hardware oddities, build system traps, "
        "environment setup steps — things that would take time to rediscover. "
        "Query when hitting a confusing error or environment issue.\n"
        "4. Backlog / TODO items: tasks with optional TTL so they auto-expire. "
        "Query at session start ('what's pending?') to resume work naturally.\n"
        "5. External research: summaries of docs, API behaviour, blog posts, library quirks looked up "
        "during a session. Query before fetching docs again.\n"
        "6. Cross-repo / global knowledge: workspace structure, available tools and their quirks, "
        "personal preferences that apply everywhere. Store in 'global' collection.\n"
        "7. Debug postmortems: root cause + fix for tricky bugs. "
        "Query when a similar symptom reappears ('have we seen this before?').\n"
        "8. Meeting / conversation outcomes: decisions made, blockers raised, people involved. "
        "Scoped to the relevant repo.\n"
        "\n"
        "WHEN TO QUERY: use mem_query proactively whenever the user asks about plans, prior decisions, "
        "context, errors, or anything that may have been stored — e.g. 'what's next?', "
        "'what did we decide?', 'have we seen this error?', 'what's the status?'. "
        "Do not wait to be asked. "
        "\n\n"
        "WHEN TO STORE: use mem_add after any decision, fix, or discovery worth remembering. "
        "Prefer short, precise entries. Use ttl_days for time-bounded items (tasks, reminders). "
        "Always update current_focus (id='current_focus') at the end of a working session. "
        "Run /reflect to do this as a structured ritual. "
        "\n\n"
        "ENTRY TYPES — use the 'type' field to control retrieval and lifecycle:\n"
        "- type='pattern': transferable mental models and design principles. No TTL — exempt from cleanup forever. "
        "Use for insights that apply across sessions and projects (e.g. architecture decisions, epistemological rules).\n"
        "- type='fact': time-sensitive information (API behaviour, version notes, env state). Use a short ttl_days (30-60). "
        "Aggressively cleaned by mem_cleanup.\n"
        "- type='feedback': agent or user guidance about how to work. No TTL.\n"
        "- type='anti-pattern': things tried that failed. Format: 'Tried: ...\\nFailed because: ...\\nInstead: ...'. "
        "Exempt from TTL cleanup — the failure history is permanent.\n"
        "- type='project': in-progress todos and project state. Use ttl_days=14 for short-lived tasks. "
        "\n\n"
        "CONSOLIDATION — use mem_dream when: the collection has grown large (>30 entries); "
        "you notice contradictions or stale entries while querying; or the user asks to 'clean up' memories. "
        "mem_dream returns a structured diff proposal — it does NOT apply changes automatically (unless auto_apply=true for deletes). "
        "Review the proposal and apply agreed changes with mem_add / mem_delete. "
        "The /reflect ritual prompts for consolidation automatically when the threshold is reached."
    )

    def update_mcp(data: dict) -> dict:
        data.setdefault("mcpServers", {})["ai-mem"] = {
            "type": "stdio",
            "command": python_exe(),
            "args": ["-m", SERVER_MODULE],
            "instructions": INSTRUCTIONS,
        }
        return data

    patch_json(mcp_path, update_mcp)

    # SessionStart + UserPromptSubmit hooks + permissions → ~/.claude/settings.json
    settings_path = HOME / ".claude" / "settings.json"
    hook_cmd = f"{python_exe()} -m ai_mem.hook 2>/dev/null || true"
    userprompt_cmd = f"{python_exe()} -m ai_mem.userprompt_hook 2>/dev/null || true"

    MCP_PERMISSIONS = [
        "mcp__ai-mem__mem_add",
        "mcp__ai-mem__mem_query",
        "mcp__ai-mem__mem_list",
        "mcp__ai-mem__mem_delete",
        "mcp__ai-mem__mem_dream",
        "mcp__ai-mem__mem_split",
        "mcp__ai-mem__mem_cleanup",
        "mcp__ai-mem__mem_train",
        "Bash(mem-dream *)",
    ]

    def update_settings(data: dict) -> dict:
        hooks = data.setdefault("hooks", {})

        session_hooks = hooks.setdefault("SessionStart", [])
        if not any(
            h.get("command") == hook_cmd
            for entry in session_hooks
            for h in entry.get("hooks", [])
        ):
            session_hooks.append({"hooks": [{"type": "command", "command": hook_cmd, "timeout": 10}]})

        up_hooks = hooks.setdefault("UserPromptSubmit", [])
        if not any(
            h.get("command") == userprompt_cmd
            for entry in up_hooks
            for h in entry.get("hooks", [])
        ):
            up_hooks.append({"hooks": [{"type": "command", "command": userprompt_cmd, "timeout": 8}]})

        allow = data.setdefault("permissions", {}).setdefault("allow", [])
        for perm in MCP_PERMISSIONS:
            if perm not in allow:
                allow.append(perm)

        return data

    patch_json(settings_path, update_settings)

    install_mem_init_command()
    install_reflect_command()
    print("   ✓ MCP server + hooks + permissions registered. Restart Claude Code to activate.")


_MEM_INIT_CONTENT = """\
Initialize or update the ai-mem memory for the current project scope.

1. Detect the active scope:
   - Find the nearest CLAUDE.md (or agent.md) walking up from the current directory
   - From that directory run: `git rev-parse --show-toplevel`
   - Try: `git remote get-url origin` and extract the trailing path component (strip .git suffix)
   - Fall back to the basename of the git root if no remote
   - If CLAUDE.md is not at the git root, append the relative subpath (dot-separated)
   - Sanitize each part: replace characters outside [a-zA-Z0-9._-] with underscore, strip leading/trailing ._-
   - Collection name: `repo.<scope>` (e.g. `repo.ai-mem`, `repo.mymonorepo.backend`)

2. Tell the user: "Initializing ai-mem for scope '<scope>' using collection 'repo.<scope>'."

3. Ask: "What are you currently working on in this scope? (This becomes current_focus.)"

4. Call mem_add:
   - documents: [<user's answer>]
   - ids: ["current_focus"]
   - collection: "repo.<scope>"

5. Confirm: "Done. Future sessions in this scope will use collection 'repo.<scope>'."

To set a global focus (surfaced in every session across all repos):
  Call mem_add with ids=["current_focus"] and collection="global".
"""


def install_mem_init_command():
    commands_dir = HOME / ".claude" / "commands"
    commands_dir.mkdir(parents=True, exist_ok=True)
    target = commands_dir / "mem-init.md"
    target.write_text(_MEM_INIT_CONTENT)
    print(f"   ✓ /mem-init command installed at {target}")


def install_reflect_command():
    source = REPO_ROOT / "skills" / "reflect.md"
    if not source.exists():
        print("   ⚠ skills/reflect.md not found, skipping /reflect install.")
        return
    commands_dir = HOME / ".claude" / "commands"
    commands_dir.mkdir(parents=True, exist_ok=True)
    target = commands_dir / "reflect.md"
    target.write_text(source.read_text())
    print(f"   ✓ /reflect command installed at {target}")


def register_gemini():
    """Add ai-mem to ~/.gemini/settings.json mcpServers."""
    config_path = HOME / ".gemini" / "settings.json"
    print(f"\n♊ Registering with Gemini CLI ({config_path})...")

    def update(data: dict) -> dict:
        data.setdefault("mcpServers", {})["ai-mem"] = {
            "command": python_exe(),
            "args": ["-m", SERVER_MODULE],
        }
        return data

    patch_json(config_path, update)
    print("   ✓ Registered. Restart Gemini CLI to activate.")


def register_cursor():
    """Add ai-mem to Cursor's MCP settings."""
    # Cursor stores MCP config at ~/.cursor/mcp.json (all platforms)
    config_path = HOME / ".cursor" / "mcp.json"
    print(f"\n🖱  Registering with Cursor ({config_path})...")

    def update(data: dict) -> dict:
        data.setdefault("mcpServers", {})["ai-mem"] = {
            "command": python_exe(),
            "args": ["-m", SERVER_MODULE],
        }
        return data

    patch_json(config_path, update)
    print("   ✓ Registered. Restart Cursor to activate.")


# ── UI ────────────────────────────────────────────────────────────────────────

TARGETS = {
    "1": ("Claude Code", register_claude),
    "2": ("Gemini CLI",  register_gemini),
    "3": ("Cursor",      register_cursor),
}


def prompt_targets() -> list:
    print("\nSelect AI tools to configure:")
    for key, (label, _) in TARGETS.items():
        print(f"  {key}) {label}")
    print("  a) All  [default]")
    choice = input("\nChoice [1/2/3/a]: ").strip().lower() or "a"

    if choice == "a":
        return list(TARGETS.values())
    selected = []
    for ch in choice:
        if ch in TARGETS:
            selected.append(TARGETS[ch])
        else:
            print(f"  ⚠ Unknown option '{ch}', skipping.")
    return selected


def main():
    print("=" * 50)
    print("  ai-mem installer")
    print("=" * 50)

    install_package()

    targets = prompt_targets()
    for label, register_fn in targets:
        try:
            register_fn()
        except Exception as e:
            print(f"   ✗ {label} registration failed: {e}")

    print("\n✅ Done! Restart your AI tool(s) to start using ai-mem.")
    print(f"\n   Data will be stored at: {HOME / '.local' / 'share' / 'ai-mem'}")
    print("   Override with: AI_MEM_PATH=/your/path")
    print("\n   Quick start:")
    print("     /mem-init   — initialize memory for your current project")
    print("     /reflect    — run after every task to close the learning loop")
    print(f"\n   How the framework works: {REPO_ROOT / 'docs' / 'epistemology-framework.md'}")


if __name__ == "__main__":
    main()
