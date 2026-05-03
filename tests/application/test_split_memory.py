"""SplitMemoryUseCase: Claude-based entry splitting."""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai_mem.application.add_memory import AddMemoryUseCase
from ai_mem.application.detect_split_hints import SPLIT_MIN_TEXT_CHARS, SPLIT_THRESHOLD_ACCESSES
from ai_mem.application.split_memory import SplitMemoryUseCase, _parse_json
from ai_mem.domain.memory import MemoryEntry


# ---------------------------------------------------------------------------
# _parse_json unit tests
# ---------------------------------------------------------------------------

def test_parse_json_clean_array():
    data = [{"id": "x_1", "text": "part one"}, {"id": "x_2", "text": "part two"}]
    assert _parse_json(json.dumps(data)) == data


def test_parse_json_extracts_array_from_prose():
    raw = 'Sure, here you go:\n[{"id": "x_1", "text": "a"}, {"id": "x_2", "text": "b"}]\nDone.'
    result = _parse_json(raw)
    assert len(result) == 2
    assert result[0]["id"] == "x_1"


def test_parse_json_raises_on_non_array():
    with pytest.raises((json.JSONDecodeError, ValueError)):
        _parse_json('{"id": "x"}')


def test_parse_json_raises_on_missing_id_key():
    with pytest.raises(ValueError):
        _parse_json('[{"text": "no id here"}]')


# ---------------------------------------------------------------------------
# SplitMemoryUseCase integration tests (subprocess mocked)
# ---------------------------------------------------------------------------

_LONG_TEXT = "a" * SPLIT_MIN_TEXT_CHARS
_HIGH_ACCESS = SPLIT_THRESHOLD_ACCESSES

_SPLIT_RESPONSE = json.dumps([
    {"id": "entry_1", "text": "first sub-topic"},
    {"id": "entry_2", "text": "second sub-topic"},
])


def _make_uc(tmp_repo) -> SplitMemoryUseCase:
    return SplitMemoryUseCase(tmp_repo, AddMemoryUseCase(tmp_repo))


def _add_entry(tmp_repo, entry_id: str, text: str, access_count: int = 0) -> None:
    # Upsert directly so access_count is persisted as-is (AddMemoryUseCase
    # does not propagate caller-supplied access_count — it reads prior_meta).
    tmp_repo.upsert("col", [MemoryEntry(
        id=entry_id,
        text=text,
        metadata={"access_count": access_count, "created_at": time.time()},
    )])


@patch("subprocess.run")
def test_split_entry_by_id_deletes_original_and_adds_sub_entries(mock_run, tmp_repo):
    mock_run.return_value = MagicMock(stdout=_SPLIT_RESPONSE, returncode=0)
    _add_entry(tmp_repo, "entry", _LONG_TEXT)

    result = _make_uc(tmp_repo).execute("col", entry_id="entry")

    assert len(result) == 1
    r = result[0]
    assert r.original_id == "entry"
    assert r.new_ids == ["entry_1", "entry_2"]
    assert r.skipped is False

    remaining = tmp_repo.get_by_ids("col", ["entry"])
    assert remaining == []  # original deleted

    new_entries = tmp_repo.get_by_ids("col", ["entry_1", "entry_2"])
    assert len(new_entries) == 2


@patch("subprocess.run")
def test_split_all_hints_skips_low_access_entries(mock_run, tmp_repo):
    mock_run.return_value = MagicMock(stdout=_SPLIT_RESPONSE, returncode=0)
    _add_entry(tmp_repo, "hinted", _LONG_TEXT, access_count=_HIGH_ACCESS)
    _add_entry(tmp_repo, "not_hinted", _LONG_TEXT, access_count=_HIGH_ACCESS - 1)

    results = _make_uc(tmp_repo).execute("col")

    assert len(results) == 1
    assert results[0].original_id == "hinted"


@patch("subprocess.run")
def test_split_all_hints_skips_short_text(mock_run, tmp_repo):
    mock_run.return_value = MagicMock(stdout=_SPLIT_RESPONSE, returncode=0)
    short_text = "a" * (SPLIT_MIN_TEXT_CHARS - 1)
    _add_entry(tmp_repo, "short_entry", short_text, access_count=_HIGH_ACCESS)

    results = _make_uc(tmp_repo).execute("col")

    assert results == []


@patch("subprocess.run")
def test_split_returns_skipped_on_json_parse_error(mock_run, tmp_repo):
    mock_run.return_value = MagicMock(stdout="not valid json", returncode=0)
    _add_entry(tmp_repo, "entry", _LONG_TEXT)

    result = _make_uc(tmp_repo).execute("col", entry_id="entry")

    assert result[0].skipped is True
    assert "JSON parse error" in result[0].skip_reason

    still_there = tmp_repo.get_by_ids("col", ["entry"])
    assert len(still_there) == 1  # original preserved on failure


@patch("subprocess.run")
def test_split_returns_skipped_on_claude_cli_error(mock_run, tmp_repo):
    import subprocess as sp
    mock_run.side_effect = sp.CalledProcessError(1, "claude", stderr="error")
    _add_entry(tmp_repo, "entry", _LONG_TEXT)

    result = _make_uc(tmp_repo).execute("col", entry_id="entry")

    assert result[0].skipped is True
    assert "claude CLI error" in result[0].skip_reason


def test_split_empty_collection_returns_empty(tmp_repo):
    results = _make_uc(tmp_repo).execute("empty_col")
    assert results == []
