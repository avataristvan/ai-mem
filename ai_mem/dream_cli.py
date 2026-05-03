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
    parser.add_argument("--dry-run", action="store_true",
                        help="List entries that would be processed, no API calls")
    parser.add_argument("--no-save", action="store_true",
                        help="Print result only, don't write to dream-log/")
    args = parser.parse_args()

    repo = _build_repo()

    if args.dry_run:
        collections = [args.collection] if args.collection else [c.name for c in repo.list_collections()]
        total = 0
        for col in collections:
            entries = repo.get_all(col)
            print(f"{col}: {len(entries)} entries")
            total += len(entries)
        print(f"Total: {total} entries across {len(collections)} collection(s)")
        return

    print(f"[dream:{args.mode}] running...", file=sys.stderr)
    use_case = DreamMemoryUseCase(repo)
    try:
        result = use_case.execute(args.collection, args.mode)
    except (RuntimeError, subprocess.CalledProcessError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(result)

    if not args.no_save:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d-%H%M")
        log_path = _LOG_DIR / f"{ts}-{args.mode}.md"
        log_path.write_text(result)
        print(f"\n→ saved: {log_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
