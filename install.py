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

    def update_mcp(data: dict) -> dict:
        data.setdefault("mcpServers", {})["ai-mem"] = {
            "type": "stdio",
            "command": python_exe(),
            "args": ["-m", SERVER_MODULE],
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

    # Stop hook → reminds Claude to update current_focus
    stop_cmd = (
        "changed=$(git diff --name-only HEAD 2>/dev/null | wc -l); "
        "if [ \"${changed:-0}\" -gt 0 ]; then "
        "echo \"Files changed this session - update current_focus in ai-mem (mem_add id=current_focus).\"; "
        "exit 2; fi"
    )

    def update_stop_hook(data: dict) -> dict:
        hooks = data.setdefault("hooks", {})
        stop_hooks = hooks.setdefault("Stop", [])
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
    print("   ✓ MCP server + SessionStart + Stop hooks registered. Restart Claude Code to activate.")


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
