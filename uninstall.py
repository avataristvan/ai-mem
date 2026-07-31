#!/usr/bin/env python3
"""ai-mem uninstaller — removes MCP server, hooks, permissions, and commands.

Intentionally does NOT remove:
  ~/.local/share/ai-mem/   — ChromaDB memory data and ranker weights (user data)
  ~/.config/ai-mem/        — user config (agents.yaml, settings)

Run with --dry-run to preview what would be removed without changing anything.
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.resolve()
HOME = Path.home()
SERVER_MODULE = "ai_mem.server"
DB_PATH = Path(
    __import__("os").environ.get("AI_MEM_PATH", str(HOME / ".local" / "share" / "ai-mem"))
)


def stop_running_daemon(dry_run: bool) -> None:
    """Stop the background query daemon (ai_mem/daemon.py) and remove its socket/pid files."""
    print(f"\n  Background daemon ({DB_PATH})")
    if dry_run:
        print("    dry-run — skipping")
        return
    try:
        from ai_mem.daemon import stop_daemon
    except ImportError:
        print("    package not importable — nothing to stop")
        return
    if stop_daemon(DB_PATH):
        print("    stopped")
    else:
        print("    not running — nothing to remove")


def python_exe() -> str:
    return sys.executable


def _hook_commands() -> tuple[str, str, str]:
    py = python_exe()
    return (
        f"{py} -m ai_mem.hook 2>/dev/null || true",
        f"{py} -m ai_mem.userprompt_hook 2>/dev/null || true",
        f"{py} -m ai_mem.posttool_hook 2>/dev/null || true",
    )


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


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _save_json(path: Path, data: dict, dry_run: bool) -> None:
    if dry_run:
        return
    path.write_text(json.dumps(data, indent=2) + "\n")


def _remove_hook_entries(hook_list: list, commands: set[str]) -> tuple[list, int]:
    """Remove hook entries whose command matches any of *commands*. Returns (filtered, removed_count)."""
    kept, removed = [], 0
    for entry in hook_list:
        entry_cmds = {h.get("command") for h in entry.get("hooks", [])}
        if entry_cmds & commands:
            removed += 1
        else:
            kept.append(entry)
    return kept, removed


# ── target: Claude Code ───────────────────────────────────────────────────────

def unregister_claude(dry_run: bool) -> None:
    hook_cmd, userprompt_cmd, posttool_cmd = _hook_commands()
    hook_commands = {hook_cmd, userprompt_cmd, posttool_cmd}

    # MCP server — ~/.claude.json
    mcp_path = HOME / ".claude.json"
    print(f"\n  Claude Code MCP ({mcp_path})")
    data = _load_json(mcp_path)
    if "ai-mem" in data.get("mcpServers", {}):
        del data["mcpServers"]["ai-mem"]
        _save_json(mcp_path, data, dry_run)
        print("    removed mcpServers[\"ai-mem\"]")
    else:
        print("    not registered — nothing to remove")

    # Hooks + permissions — ~/.claude/settings.json
    settings_path = HOME / ".claude" / "settings.json"
    print(f"\n  Claude Code settings ({settings_path})")
    data = _load_json(settings_path)
    changed = False

    hooks = data.get("hooks", {})
    for event in ("SessionStart", "UserPromptSubmit", "PostToolUse"):
        before = hooks.get(event, [])
        after, n = _remove_hook_entries(before, hook_commands)
        if n:
            hooks[event] = after
            print(f"    removed {n} {event} hook(s)")
            changed = True

    allow = data.get("permissions", {}).get("allow", [])
    removed_perms = [p for p in MCP_PERMISSIONS if p in allow]
    if removed_perms:
        data["permissions"]["allow"] = [p for p in allow if p not in MCP_PERMISSIONS]
        print(f"    removed {len(removed_perms)} permission(s)")
        changed = True

    if changed:
        _save_json(settings_path, data, dry_run)
    else:
        print("    nothing to remove")

    # Commands
    for name in ("mem-init.md", "reflect.md"):
        cmd_file = HOME / ".claude" / "commands" / name
        print(f"\n  Command {cmd_file}")
        if cmd_file.exists():
            if not dry_run:
                cmd_file.unlink()
            print("    removed")
        else:
            print("    not found — nothing to remove")


# ── target: Gemini CLI ────────────────────────────────────────────────────────

def unregister_gemini(dry_run: bool) -> None:
    config_path = HOME / ".gemini" / "settings.json"
    print(f"\n  Gemini CLI ({config_path})")
    data = _load_json(config_path)
    if "ai-mem" in data.get("mcpServers", {}):
        del data["mcpServers"]["ai-mem"]
        _save_json(config_path, data, dry_run)
        print("    removed mcpServers[\"ai-mem\"]")
    else:
        print("    not registered — nothing to remove")


# ── target: Cursor ────────────────────────────────────────────────────────────

def unregister_cursor(dry_run: bool) -> None:
    config_path = HOME / ".cursor" / "mcp.json"
    print(f"\n  Cursor ({config_path})")
    data = _load_json(config_path)
    if "ai-mem" in data.get("mcpServers", {}):
        del data["mcpServers"]["ai-mem"]
        _save_json(config_path, data, dry_run)
        print("    removed mcpServers[\"ai-mem\"]")
    else:
        print("    not registered — nothing to remove")


# ── UI ────────────────────────────────────────────────────────────────────────

TARGETS = {
    "1": ("Claude Code", unregister_claude),
    "2": ("Gemini CLI",  unregister_gemini),
    "3": ("Cursor",      unregister_cursor),
}


def prompt_targets() -> list:
    print("\nSelect AI tools to unregister:")
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
            print(f"  Unknown option '{ch}', skipping.")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Uninstall ai-mem integration artifacts.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview removals without writing any files")
    args = parser.parse_args()

    print("=" * 50)
    print("  ai-mem uninstaller")
    if args.dry_run:
        print("  DRY RUN — no files will be modified")
    print("=" * 50)
    print(
        "\n  Memory data is preserved:\n"
        f"    {HOME / '.local' / 'share' / 'ai-mem'}  (ChromaDB + ranker weights)\n"
        f"    {HOME / '.config' / 'ai-mem'}           (user config)\n"
        "  Delete these manually if you want a full wipe."
    )

    targets = prompt_targets()
    for label, fn in targets:
        print(f"\n── {label} ──")
        try:
            fn(args.dry_run)
        except Exception as e:
            print(f"  Error: {e}")

    stop_running_daemon(args.dry_run)

    pip_answer = input("\nRemove the ai-mem Python package? [y/N] ").strip().lower()
    if pip_answer == "y":
        import subprocess
        cmd = [sys.executable, "-m", "pip", "uninstall", "ai-mem", "-y"]
        print(f"  $ {' '.join(cmd)}")
        if not args.dry_run:
            subprocess.run(cmd, check=False)
    else:
        print("  Package kept.")

    print("\nDone. Restart your AI tool(s) to apply changes.")


if __name__ == "__main__":
    main()
