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
        "ai-mem is a persistent semantic memory store for engineers. "
        "When these tools are available, ai-mem is installed and active. "
        "\n\n"
        "MEMORY SCOPING: Memory is scoped to the nearest CLAUDE.md. "
        "Use the repo collection injected at session start (e.g. 'repo.my-project') for project-specific context, "
        "'global' for cross-repo knowledge that should surface everywhere, "
        "and 'workspace' as the fallback when no CLAUDE.md is present. "
        "Run /mem-init to initialise a new scope. "
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
        "Always update current_focus (id='current_focus') at the end of a working session."
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

    # SessionStart hook → ~/.claude/settings.json
    settings_path = HOME / ".claude" / "settings.json"
    hook_cmd = f"{python_exe()} -m ai_mem.hook 2>/dev/null || true"

    def update_settings(data: dict) -> dict:
        hooks = data.setdefault("hooks", {})
        session_hooks = hooks.setdefault("SessionStart", [])
        # avoid duplicates
        already = any(
            h.get("command") == hook_cmd
            for entry in session_hooks
            for h in entry.get("hooks", [])
        )
        if not already:
            session_hooks.append({"hooks": [{"type": "command", "command": hook_cmd, "timeout": 10}]})
        return data

    # UserPromptSubmit hook → injects relevant memories before each user message (when ranker is ready)
    userprompt_cmd = f"{python_exe()} -m ai_mem.userprompt_hook 2>/dev/null || true"

    def update_userprompt_hook(data: dict) -> dict:
        hooks = data.setdefault("hooks", {})
        up_hooks = hooks.setdefault("UserPromptSubmit", [])
        already = any(
            h.get("command") == userprompt_cmd
            for entry in up_hooks
            for h in entry.get("hooks", [])
        )
        if not already:
            up_hooks.append({"hooks": [{"type": "command", "command": userprompt_cmd, "timeout": 8}]})
        return data

    patch_json(settings_path, update_userprompt_hook)

    # Stop hook → reminds Claude to update current_focus
    OLD_STOP_CMD = (
        "changed=$(git diff --name-only HEAD 2>/dev/null | wc -l); "
        "if [ \"${changed:-0}\" -gt 0 ]; then "
        "echo \"Files changed this session - update current_focus in ai-mem (mem_add id=current_focus).\"; "
        "exit 2; fi"
    )
    stop_cmd = f"{python_exe()} -m ai_mem.stop_hook 2>/dev/null || true"

    def update_stop_hook(data: dict) -> dict:
        hooks = data.setdefault("hooks", {})
        stop_hooks = hooks.setdefault("Stop", [])
        # Remove the old bash one-liner if present
        hooks["Stop"] = [
            entry for entry in stop_hooks
            if not any(h.get("command") == OLD_STOP_CMD for h in entry.get("hooks", []))
        ]
        stop_hooks = hooks["Stop"]
        already = any(
            h.get("command") == stop_cmd
            for entry in stop_hooks
            for h in entry.get("hooks", [])
        )
        if not already:
            stop_hooks.append({
                "hooks": [{
                    "type": "command",
                    "command": stop_cmd,
                    "asyncRewake": True,
                    "rewakeSummary": "ai-mem focus update reminder"
                }]
            })
        return data

    patch_json(settings_path, update_stop_hook)
    install_mem_init_command()
    print("   ✓ MCP server + SessionStart + UserPromptSubmit + Stop hooks registered. Restart Claude Code to activate.")


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


if __name__ == "__main__":
    main()
