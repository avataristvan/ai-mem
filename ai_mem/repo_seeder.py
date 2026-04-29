"""Seeds a repo.* collection with CLAUDE.md sections on first session."""
from __future__ import annotations

import re
import time
from pathlib import Path

from ai_mem.session_stats import injection_rate

SEED_THRESHOLD = 0.60


def _slug(heading: str) -> str:
    lowered = heading.lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_]", "", lowered)


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Return (heading, full_section_text) pairs for all H2 sections."""
    parts = re.split(r"\n(?=## )", text)
    sections = []
    for part in parts:
        part = part.strip()
        if not part or not part.startswith("## "):
            continue
        heading = part.splitlines()[0][3:].strip()
        sections.append((heading, part))
    return sections


def seed_collection(
    collection: str,
    claude_md_path: Path,
    add_uc,
    stats_path: Path,
    global_scope: str = "global",
) -> int:
    try:
        if injection_rate(stats_path, global_scope) < SEED_THRESHOLD:
            return 0
        if not claude_md_path.exists():
            return 0

        text = claude_md_path.read_text(encoding="utf-8")
        sections = _split_sections(text)
        if not sections:
            return 0

        seed_ts = time.time()
        documents = []
        ids = []
        metadatas = []
        for heading, body in sections:
            documents.append(body)
            ids.append(f"seed_{_slug(heading)}")
            metadatas.append({"seeded_from": "CLAUDE.md", "seed_ts": seed_ts})

        add_uc.execute(
            collection=collection,
            documents=documents,
            ids=ids,
            metadatas=metadatas,
        )
        return len(documents)
    except Exception:
        return 0
