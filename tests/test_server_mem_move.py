"""Tests for the mem_move dispatch branch in server.py."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import ai_mem.server as server_module
from ai_mem.application.move_memory import MoveConflict, MoveResult


def _run(arguments: dict):
    return asyncio.run(server_module.call_tool("mem_move", arguments))


def test_call_tool_mem_move_clean():
    mock_move = MagicMock()
    mock_move.execute.return_value = MoveResult(moved_ids=["a", "b"])

    with patch.object(server_module, "_move", mock_move):
        result = _run({"ids": ["a", "b"], "collection": "source", "to_collection": "target"})

    mock_move.execute.assert_called_once_with(from_collection="source", to_collection="target", ids=["a", "b"])
    text = result[0].text
    assert "Moved 2 entry/entries" in text
    assert "'source'" in text
    assert "'target'" in text


def test_call_tool_mem_move_reports_not_found():
    mock_move = MagicMock()
    mock_move.execute.return_value = MoveResult(moved_ids=["a"], not_found_ids=["ghost"])

    with patch.object(server_module, "_move", mock_move):
        result = _run({"ids": ["a", "ghost"], "to_collection": "target"})

    text = result[0].text
    assert "not_found" in text
    assert "ghost" in text


def test_call_tool_mem_move_reports_conflicts():
    mock_move = MagicMock()
    mock_move.execute.return_value = MoveResult(
        moved_ids=[],
        conflicts=[MoveConflict(id="c1", existing_text_preview="target text", source_text_preview="source text")],
    )

    with patch.object(server_module, "_move", mock_move):
        result = _run({"ids": ["c1"], "to_collection": "target"})

    text = result[0].text
    assert "conflicts" in text
    assert '"id": "c1"' in text
    assert "target text" in text
    assert "source text" in text
