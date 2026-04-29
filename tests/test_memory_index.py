"""Tests for MEMORY.md auto-demotion logic."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ai_mem.memory_index import (
    IndexEntry,
    auto_demote,
    find_memory_dir,
    read_index,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_mem_file(memory_dir: Path, filename: str, mem_type: str, body: str = "body") -> Path:
    path = memory_dir / filename
    path.write_text(
        f"---\nname: {filename}\ndescription: test\ntype: {mem_type}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return path


def _write_memory_md(memory_dir: Path, entries: list[tuple[str, str]], extra_lines: int = 0) -> None:
    lines = ["# Memory Index\n\n"]
    for title, filename in entries:
        lines.append(f"- [{title}]({filename}) — description\n")
    for i in range(extra_lines):
        lines.append(f"# padding line {i}\n")
    (memory_dir / "MEMORY.md").write_text("".join(lines), encoding="utf-8")


def _mock_add_uc() -> MagicMock:
    uc = MagicMock()
    uc.execute.return_value = 1
    return uc


# ---------------------------------------------------------------------------
# find_memory_dir
# ---------------------------------------------------------------------------

def test_find_memory_dir_returns_none_when_missing(tmp_path: Path) -> None:
    result = find_memory_dir(cwd=tmp_path / "proj", home=tmp_path)
    assert result is None


def test_find_memory_dir_returns_path_when_exists(tmp_path: Path) -> None:
    cwd = tmp_path / "some" / "project"
    encoded = str(cwd).replace("/", "-")
    memory_dir = tmp_path / ".claude" / "projects" / encoded / "memory"
    memory_dir.mkdir(parents=True)
    assert find_memory_dir(cwd=cwd, home=tmp_path) == memory_dir


# ---------------------------------------------------------------------------
# read_index
# ---------------------------------------------------------------------------

def test_read_index_missing_memory_md(tmp_path: Path) -> None:
    entries, count = read_index(tmp_path)
    assert entries == []
    assert count == 0


def test_read_index_empty_memory_md(tmp_path: Path) -> None:
    (tmp_path / "MEMORY.md").write_text("# Memory Index\n\n", encoding="utf-8")
    entries, count = read_index(tmp_path)
    assert entries == []
    assert count == 2


def test_read_index_parses_entries(tmp_path: Path) -> None:
    _write_mem_file(tmp_path, "feedback_test.md", "feedback")
    _write_mem_file(tmp_path, "project_init.md", "project")
    _write_memory_md(tmp_path, [("Test feedback", "feedback_test.md"), ("Init project", "project_init.md")])

    entries, count = read_index(tmp_path)
    assert len(entries) == 2
    assert entries[0].title == "Test feedback"
    assert entries[0].mem_type == "feedback"
    assert entries[1].title == "Init project"
    assert entries[1].mem_type == "project"
    assert count == 4  # header + blank + 2 entries


def test_read_index_skips_missing_files(tmp_path: Path) -> None:
    _write_memory_md(tmp_path, [("Ghost", "ghost.md")])
    entries, _ = read_index(tmp_path)
    # entry still parsed but mem_type is empty and mtime is 0
    assert len(entries) == 1
    assert entries[0].mem_type == ""
    assert entries[0].mtime == 0.0


# ---------------------------------------------------------------------------
# auto_demote
# ---------------------------------------------------------------------------

_TEST_OPTS = dict(_threshold=10, _target_lines=5)  # keep tests small and fast


def test_auto_demote_noop_below_threshold(tmp_path: Path) -> None:
    _write_mem_file(tmp_path, "fb.md", "feedback")
    _write_memory_md(tmp_path, [("Feedback", "fb.md")])
    add_uc = _mock_add_uc()

    result = auto_demote(tmp_path, add_uc, _threshold=180, _target_lines=160)
    assert result == []
    add_uc.execute.assert_not_called()


def test_auto_demote_demotes_when_above_threshold(tmp_path: Path) -> None:
    _write_mem_file(tmp_path, "proj.md", "project", body="project body")
    _write_memory_md(tmp_path, [("A project", "proj.md")], extra_lines=12)
    add_uc = _mock_add_uc()

    result = auto_demote(tmp_path, add_uc, global_collection="global", project_collection="repo.x", **_TEST_OPTS)
    assert result == ["A project"]
    add_uc.execute.assert_called_once()


def test_auto_demote_project_type_uses_project_collection(tmp_path: Path) -> None:
    _write_mem_file(tmp_path, "proj.md", "project")
    _write_memory_md(tmp_path, [("A project", "proj.md")], extra_lines=12)
    add_uc = _mock_add_uc()

    auto_demote(tmp_path, add_uc, global_collection="global", project_collection="repo.x", **_TEST_OPTS)
    _, kwargs = add_uc.execute.call_args
    assert kwargs["collection"] == "repo.x"


def test_auto_demote_feedback_type_uses_global_collection(tmp_path: Path) -> None:
    _write_mem_file(tmp_path, "fb.md", "feedback")
    _write_memory_md(tmp_path, [("Feedback", "fb.md")], extra_lines=12)
    add_uc = _mock_add_uc()

    auto_demote(tmp_path, add_uc, global_collection="global", project_collection="repo.x", **_TEST_OPTS)
    _, kwargs = add_uc.execute.call_args
    assert kwargs["collection"] == "global"


def test_auto_demote_project_before_feedback(tmp_path: Path) -> None:
    """project entries are demoted before feedback entries (lower TYPE_PRIORITY)."""
    _write_mem_file(tmp_path, "fb.md", "feedback")
    _write_mem_file(tmp_path, "proj.md", "project")
    # threshold=10, target=5: to_free = line_count - 5; with 2 entries + 12 padding ≈ 16 lines → free 11
    # but we only have 2 demotable entries → both demoted, project first
    _write_memory_md(tmp_path, [("Feedback entry", "fb.md"), ("Project entry", "proj.md")], extra_lines=12)
    add_uc = _mock_add_uc()

    result = auto_demote(tmp_path, add_uc, global_collection="global", project_collection="repo.x", **_TEST_OPTS)
    assert result[0] == "Project entry"  # project demoted first
    assert "Feedback entry" in result


def test_auto_demote_removes_line_from_memory_md(tmp_path: Path) -> None:
    _write_mem_file(tmp_path, "proj.md", "project")
    _write_mem_file(tmp_path, "fb.md", "feedback")
    # header+blank+2 entries+6 padding = 10 lines; threshold=10, target=9 → to_free=1
    # → only project (lower priority) is demoted; feedback survives
    _write_memory_md(tmp_path, [("Project", "proj.md"), ("Feedback", "fb.md")], extra_lines=6)
    add_uc = _mock_add_uc()

    auto_demote(tmp_path, add_uc, _threshold=10, _target_lines=9)

    remaining = (tmp_path / "MEMORY.md").read_text(encoding="utf-8")
    assert "proj.md" not in remaining  # project was demoted first
    assert "fb.md" in remaining        # feedback was kept (freed enough after project)


def test_auto_demote_keeps_entry_on_add_failure(tmp_path: Path) -> None:
    _write_mem_file(tmp_path, "proj.md", "project")
    _write_memory_md(tmp_path, [("Project", "proj.md")], extra_lines=12)
    add_uc = _mock_add_uc()
    add_uc.execute.side_effect = RuntimeError("db error")

    result = auto_demote(tmp_path, add_uc, **_TEST_OPTS)
    assert result == []
    remaining = (tmp_path / "MEMORY.md").read_text(encoding="utf-8")
    assert "proj.md" in remaining


def test_auto_demote_removes_dangling_entries(tmp_path: Path) -> None:
    """Entries whose backing file is missing are removed without calling add_uc."""
    _write_memory_md(tmp_path, [("Ghost", "ghost.md")], extra_lines=12)
    add_uc = _mock_add_uc()

    result = auto_demote(tmp_path, add_uc, **_TEST_OPTS)
    assert result == []
    add_uc.execute.assert_not_called()
    remaining = (tmp_path / "MEMORY.md").read_text(encoding="utf-8")
    assert "ghost.md" not in remaining


def test_auto_demote_includes_entry_id_as_stem(tmp_path: Path) -> None:
    _write_mem_file(tmp_path, "feedback_rules.md", "feedback")
    _write_memory_md(tmp_path, [("Rules", "feedback_rules.md")], extra_lines=12)
    add_uc = _mock_add_uc()

    auto_demote(tmp_path, add_uc, **_TEST_OPTS)
    _, kwargs = add_uc.execute.call_args
    assert kwargs["ids"] == ["feedback_rules"]
