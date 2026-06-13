#!/usr/bin/env python3
"""Migration: set boost_count=1 for pre-feature entries that qualify for [always-present].

Run once after upgrading ai-mem past the boost_count gate (commit 2088853).

Targets entries that:
  - have no boost_count key in metadata (created before the feature)
  - have confidence >= 1.0 (old default — never decayed)
  - have access_count >= 3 (actually used, not abandoned)

These entries were implicitly validated by heavy use before the explicit
boost_count signal existed. The migration restores their [always-present]
eligibility without touching entries that are new or have been decayed.

Usage:
    python3 scripts/migrate_boost_count.py [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DB_PATH = Path(os.environ.get("AI_MEM_PATH", Path.home() / ".local" / "share" / "ai-mem"))

MIN_CONFIDENCE = 1.0
MIN_ACCESS_COUNT = 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing.")
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"No ai-mem database found at {DB_PATH}. Nothing to migrate.")
        sys.exit(0)

    from ai_mem.infrastructure.chroma_repository import ChromaMemoryRepository

    repo = ChromaMemoryRepository(DB_PATH)
    collections = repo.list_collections()

    if not collections:
        print("No collections found.")
        sys.exit(0)

    total_updated = 0

    for col_info in collections:
        name = col_info.name
        entries = repo.get_all(name)
        to_update = []

        for entry in entries:
            if "boost_count" in entry.metadata:
                continue
            try:
                conf = float(entry.metadata.get("confidence", 1.0))
            except (ValueError, TypeError):
                continue
            try:
                ac = int(entry.metadata.get("access_count", 0))
            except (ValueError, TypeError):
                continue
            if conf >= MIN_CONFIDENCE and ac >= MIN_ACCESS_COUNT:
                to_update.append(entry)

        if not to_update:
            continue

        print(f"\n[{name}] {len(to_update)} entries to migrate:")
        for entry in to_update:
            conf = entry.metadata.get("confidence", "?")
            ac = entry.metadata.get("access_count", "?")
            print(f"  {entry.id}  confidence={conf}  access_count={ac}")
            if not args.dry_run:
                entry.metadata["boost_count"] = 1

        if not args.dry_run:
            repo.upsert(name, to_update)
            total_updated += len(to_update)

    if args.dry_run:
        print("\nDry run — no changes written.")
    else:
        print(f"\nMigrated {total_updated} entries across {len(collections)} collections.")


if __name__ == "__main__":
    main()
