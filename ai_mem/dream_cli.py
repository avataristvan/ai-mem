"""mem-dream CLI — manually invoke memory consolidation."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from ai_mem.application.dream_memory import DreamMemoryUseCase, MODES
from ai_mem.infrastructure.chroma_repository import ChromaMemoryRepository

_DB_PATH = Path(os.environ.get("AI_MEM_PATH", Path.home() / ".local/share/ai-mem"))
_LOG_DIR = Path.home() / ".claude/dream-log"

_EXPERT_FOCUS_HINT = (
    "Cross-project learnings from specialized AI agent roles (subagent.* collections). "
    "Each collection is one agent's accumulated experience across all projects.\n"
    "- Flag entries that are too project-specific for a cross-project collection → prefer DELETE\n"
    "- Patterns applicable across all projects → flag for propagation to global\n"
    "- Prefer DELETE over MERGE for project-specific entries: losing a fact is better "
    "than keeping noise that corrupts cross-project intuition."
)


def _build_repo():
    try:
        from ai_mem.infrastructure.bm25_repository import BM25MemoryRepository
        return BM25MemoryRepository(ChromaMemoryRepository(_DB_PATH))
    except ImportError:
        return ChromaMemoryRepository(_DB_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate ai-mem memories using Claude models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Modes: {', '.join(MODES)}",
    )
    parser.add_argument("--mode", choices=MODES, default="hier",
                        help="Consolidation mode (default: hier)")
    parser.add_argument("--collection", metavar="NAME",
                        help="Limit to one collection (default: all)")
    parser.add_argument("--expert", action="store_true",
                        help="Consolidate all subagent.* collections with expert focus hint")
    parser.add_argument("--dry-run", action="store_true",
                        help="List entries that would be processed, no API calls")
    parser.add_argument("--no-save", action="store_true",
                        help="Print result only, don't write to dream-log/")
    args = parser.parse_args()

    repo = _build_repo()

    expert_collections: list[str] | None = None
    focus_hint: str | None = None
    if args.expert:
        all_cols = [col_info.name for col_info in repo.list_collections()]
        expert_collections = [col_name for col_name in all_cols if col_name.startswith("subagent.")]
        if not expert_collections:
            print("No subagent.* collections found.", file=sys.stderr)
            return
        focus_hint = _EXPERT_FOCUS_HINT
        print(f"[dream:expert] collections: {expert_collections}", file=sys.stderr)

    if args.dry_run:
        cols = expert_collections or ([args.collection] if args.collection else [col_info.name for col_info in repo.list_collections()])
        total = 0
        for col in cols:
            entries = repo.get_all(col)
            print(f"{col}: {len(entries)} entries")
            total += len(entries)
        print(f"Total: {total} entries across {len(cols)} collection(s)")
        return

    label = f"expert:{args.mode}" if args.expert else args.mode
    print(f"[dream:{label}] running...", file=sys.stderr)
    use_case = DreamMemoryUseCase(repo, injection_log_path=_DB_PATH / "injection_log.json")
    try:
        result = use_case.execute(
            args.collection,
            args.mode,
            focus_hint=focus_hint,
            collections_filter=expert_collections,
        )
    except (RuntimeError, subprocess.CalledProcessError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(result)

    if not args.no_save:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d-%H%M")
        log_path = _LOG_DIR / f"{ts}-{label}.md"
        log_path.write_text(result)
        print(f"\n→ saved: {log_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
