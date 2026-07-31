"""Tests for injection_log.py — push-event log used to measure pull-through."""
from __future__ import annotations

import json
from pathlib import Path

from ai_mem.injection_log import _MAX_EVENTS_PER_COLLECTION, load_events, record_pushed


def test_record_creates_file_on_first_call(tmp_path: Path) -> None:
    log = tmp_path / "injection_log.json"
    record_pushed(log, "global", ["id-1"])
    assert log.exists()


def test_load_events_returns_empty_list_for_missing_file(tmp_path: Path) -> None:
    log = tmp_path / "injection_log.json"
    assert load_events(log, "global") == []


def test_record_and_load_roundtrip(tmp_path: Path) -> None:
    log = tmp_path / "injection_log.json"
    record_pushed(log, "global", ["id-1", "id-2"])

    events = load_events(log, "global")
    assert {e["id"] for e in events} == {"id-1", "id-2"}
    assert all(isinstance(e["ts"], float) for e in events)


def test_no_ids_is_a_noop(tmp_path: Path) -> None:
    log = tmp_path / "injection_log.json"
    record_pushed(log, "global", [])
    assert not log.exists()


def test_collections_stored_independently(tmp_path: Path) -> None:
    log = tmp_path / "injection_log.json"
    record_pushed(log, "global", ["id-1"])
    record_pushed(log, "repo.ai-mem", ["id-2"])

    assert {e["id"] for e in load_events(log, "global")} == {"id-1"}
    assert {e["id"] for e in load_events(log, "repo.ai-mem")} == {"id-2"}


def test_events_trimmed_to_max_per_collection(tmp_path: Path) -> None:
    log = tmp_path / "injection_log.json"
    for i in range(_MAX_EVENTS_PER_COLLECTION + 10):
        record_pushed(log, "global", [f"id-{i}"])

    data = json.loads(log.read_text())
    assert len(data["global"]) == _MAX_EVENTS_PER_COLLECTION
    # The oldest events were dropped, so the tail (most recent) ids survive.
    assert data["global"][-1]["id"] == f"id-{_MAX_EVENTS_PER_COLLECTION + 9}"


def test_corrupt_log_file_is_ignored(tmp_path: Path) -> None:
    log = tmp_path / "injection_log.json"
    log.write_text("{corrupt json{{")

    assert load_events(log, "global") == []
    record_pushed(log, "global", ["id-1"])  # must not raise
    assert {e["id"] for e in load_events(log, "global")} == {"id-1"}
