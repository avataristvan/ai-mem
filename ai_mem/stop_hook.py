#!/usr/bin/env python3
"""SessionStop hook — reminds Claude to update current_focus after file changes."""
import subprocess
import sys


def main():
    try:
        r = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            capture_output=True, text=True,
        )
        changed = len([l for l in r.stdout.splitlines() if l.strip()])
        if changed == 0:
            return

        from ai_mem.repo_context import detect_repo_context, WORKSPACE_COLLECTION

        ctx = detect_repo_context()
        if ctx.collection != WORKSPACE_COLLECTION:
            hint = f'mem_add id=current_focus collection="{ctx.collection}"'
        else:
            hint = "mem_add id=current_focus"

        print(
            f"Files changed this session ({changed} file(s)) — "
            f"update current_focus in ai-mem ({hint})."
        )
        sys.exit(2)
    except Exception:
        pass


if __name__ == "__main__":
    main()
