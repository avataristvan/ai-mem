#!/usr/bin/env python3
"""SessionStart hook — injects current_focus from ai-mem into Claude's context.

Outputs hookSpecificOutput JSON if a 'current_focus' entry exists in the
'workspace' collection. Silent (exit 0, no output) if nothing is stored yet.
"""
import json
import os
import sys
from pathlib import Path

DB_PATH = Path(os.environ.get("AI_MEM_PATH", Path.home() / ".local" / "share" / "ai-mem"))


def main():
    if not DB_PATH.exists():
        return

    try:
        import chromadb
    except ImportError:
        return

    try:
        client = chromadb.PersistentClient(path=str(DB_PATH))
        col = client.get_collection("workspace")
        result = col.get(ids=["current_focus"])
        docs = result.get("documents") or []
        if docs and docs[0]:
            output = {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": f"[ai-mem] Current focus:\n{docs[0]}",
                }
            }
            print(json.dumps(output))
    except Exception:
        pass


if __name__ == "__main__":
    main()
