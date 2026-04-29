"""Tests for CLAUDE.md → repo.* collection seeding."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ai_mem.repo_seeder import SEED_THRESHOLD, seed_collection
from ai_mem.session_stats import record_injection

CLAUDE_MD = """\
# Project

Intro line.

## Dev Commands

```bash
pytest tests/
```

## Architecture

Capability-centric DDD.
"""


def _add_uc() -> MagicMock:
    uc = MagicMock()
    uc.execute.return_value = 1
    return uc


def _stats_above_threshold(stats_path: Path) -> None:
    for _ in range(4):
        record_injection(stats_path, "global", injected=True)
    record_injection(stats_path, "global", injected=False)  # 4/5 = 0.8 > 0.6


def _stats_below_threshold(stats_path: Path) -> None:
    for _ in range(2):
        record_injection(stats_path, "global", injected=True)
    for _ in range(3):
        record_injection(stats_path, "global", injected=False)  # 2/5 = 0.4 < 0.6


def test_returns_zero_when_threshold_not_met(tmp_path: Path) -> None:
    stats = tmp_path / "stats.json"
    _stats_below_threshold(stats)
    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text(CLAUDE_MD, encoding="utf-8")
    add_uc = _add_uc()

    result = seed_collection("repo.x", claude_md, add_uc, stats)
    assert result == 0
    add_uc.execute.assert_not_called()


def test_returns_zero_when_claude_md_missing(tmp_path: Path) -> None:
    stats = tmp_path / "stats.json"
    _stats_above_threshold(stats)
    add_uc = _add_uc()

    result = seed_collection("repo.x", tmp_path / "CLAUDE.md", add_uc, stats)
    assert result == 0
    add_uc.execute.assert_not_called()


def test_parses_h2_sections_correctly(tmp_path: Path) -> None:
    stats = tmp_path / "stats.json"
    _stats_above_threshold(stats)
    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text(CLAUDE_MD, encoding="utf-8")
    add_uc = _add_uc()

    result = seed_collection("repo.x", claude_md, add_uc, stats)
    assert result == 2  # Dev Commands + Architecture

    _, kwargs = add_uc.execute.call_args
    assert len(kwargs["documents"]) == 2


def test_entry_ids_follow_seed_slug_format(tmp_path: Path) -> None:
    stats = tmp_path / "stats.json"
    _stats_above_threshold(stats)
    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text(CLAUDE_MD, encoding="utf-8")
    add_uc = _add_uc()

    seed_collection("repo.x", claude_md, add_uc, stats)
    _, kwargs = add_uc.execute.call_args
    assert kwargs["ids"] == ["seed_dev_commands", "seed_architecture"]


def test_seeded_entries_have_correct_metadata(tmp_path: Path) -> None:
    stats = tmp_path / "stats.json"
    _stats_above_threshold(stats)
    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text(CLAUDE_MD, encoding="utf-8")
    add_uc = _add_uc()

    seed_collection("repo.x", claude_md, add_uc, stats)
    _, kwargs = add_uc.execute.call_args
    for meta in kwargs["metadatas"]:
        assert meta["seeded_from"] == "CLAUDE.md"
        assert "seed_ts" in meta


def test_returns_zero_silently_on_add_uc_exception(tmp_path: Path) -> None:
    stats = tmp_path / "stats.json"
    _stats_above_threshold(stats)
    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text(CLAUDE_MD, encoding="utf-8")
    add_uc = _add_uc()
    add_uc.execute.side_effect = RuntimeError("db error")

    result = seed_collection("repo.x", claude_md, add_uc, stats)
    assert result == 0
