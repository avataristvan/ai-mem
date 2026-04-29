"""Tests for session_stats rolling injection tracking."""
from __future__ import annotations

from pathlib import Path

from ai_mem.session_stats import WINDOW, injection_rate, record_injection


def test_record_creates_file_on_first_call(tmp_path: Path) -> None:
    stats = tmp_path / "session_stats.json"
    record_injection(stats, "global", injected=True)
    assert stats.exists()


def test_injection_rate_returns_zero_for_missing_file(tmp_path: Path) -> None:
    stats = tmp_path / "session_stats.json"
    assert injection_rate(stats, "global") == 0.0


def test_injection_rate_calculates_correctly(tmp_path: Path) -> None:
    stats = tmp_path / "session_stats.json"
    for injected in [True, True, True, False, False]:
        record_injection(stats, "global", injected=injected)
    assert injection_rate(stats, "global") == pytest.approx(0.6)


def test_window_trims_to_last_window_entries(tmp_path: Path) -> None:
    import json

    stats = tmp_path / "session_stats.json"
    for i in range(WINDOW + 5):
        record_injection(stats, "global", injected=(i % 2 == 0))
    data = json.loads(stats.read_text())
    assert len(data["global"]) == WINDOW


def test_multiple_scopes_stored_independently(tmp_path: Path) -> None:
    stats = tmp_path / "session_stats.json"
    record_injection(stats, "global", injected=True)
    record_injection(stats, "global", injected=True)
    record_injection(stats, "other", injected=False)

    assert injection_rate(stats, "global") == pytest.approx(1.0)
    assert injection_rate(stats, "other") == pytest.approx(0.0)


import pytest
