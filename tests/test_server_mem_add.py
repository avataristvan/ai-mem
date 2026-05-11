"""Tests for the mem_add anti-pattern → pattern link suggestion in server.py."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

import ai_mem.server as server_module
from ai_mem.server import _suggest_pattern_links


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_query_result(entry_id: str, score: float, text: str) -> MagicMock:
    r = MagicMock()
    r.id = entry_id
    r.score = score
    r.text = text
    r.metadata = {}
    return r


def _patch_query(results):
    """Return a context manager that replaces _query.execute with a fixed return value."""
    mock_query = MagicMock()
    mock_query.execute.return_value = results
    return patch.object(server_module, "_query", mock_query)


# ---------------------------------------------------------------------------
# _suggest_pattern_links unit tests
# ---------------------------------------------------------------------------

def test_suggest_returns_link_when_pattern_found_above_threshold():
    matching_pattern = _make_query_result("pat1", score=0.75, text="Always validate input before processing")

    with _patch_query([matching_pattern]):
        suggestion = _suggest_pattern_links(
            collection="workspace",
            documents=["Tried: skipping validation\nFailed because: security hole\nInstead: validate always"],
            stored_ids=["ap1"],
        )

    assert "💡 Related patterns found" in suggestion
    assert 'source_id="ap1"' in suggestion
    assert 'target_id="pat1"' in suggestion
    assert 'edge_type="contradicts"' in suggestion
    assert 'collection="workspace"' in suggestion
    assert "Always validate input" in suggestion


def test_suggest_returns_empty_when_no_results_above_threshold():
    low_score_result = _make_query_result("pat2", score=0.2, text="Some pattern text")

    with _patch_query([low_score_result]):
        suggestion = _suggest_pattern_links(
            collection="workspace",
            documents=["Tried: something\nFailed because: reason\nInstead: better way"],
            stored_ids=["ap2"],
        )

    assert suggestion == ""


def test_suggest_returns_empty_when_query_result_list_is_empty():
    with _patch_query([]):
        suggestion = _suggest_pattern_links(
            collection="workspace",
            documents=["Tried: something\nFailed because: reason\nInstead: better way"],
            stored_ids=["ap3"],
        )

    assert suggestion == ""


def test_suggest_silent_fail_when_query_raises():
    mock_query = MagicMock()
    mock_query.execute.side_effect = RuntimeError("chroma down")

    with patch.object(server_module, "_query", mock_query):
        suggestion = _suggest_pattern_links(
            collection="workspace",
            documents=["Tried: X\nFailed because: Y\nInstead: Z"],
            stored_ids=["ap4"],
        )

    assert suggestion == ""


# ---------------------------------------------------------------------------
# Integration: call_tool mem_add dispatches correctly
# ---------------------------------------------------------------------------

def test_call_tool_mem_add_non_antipattern_no_suggestion():
    """A regular mem_add (no type or type != anti-pattern) never triggers the query."""
    mock_add = MagicMock()
    mock_add.execute.return_value = 1
    mock_query = MagicMock()

    async def _run():
        return await server_module.call_tool("mem_add", {
            "documents": ["normal note"],
            "ids": ["note1"],
            "collection": "workspace",
        })

    with patch.object(server_module, "_add", mock_add), \
         patch.object(server_module, "_query", mock_query):
        result = asyncio.run(_run())

    mock_query.execute.assert_not_called()
    assert len(result) == 1
    text = result[0].text
    assert "Related patterns" not in text
    assert "Stored 1 entry" in text


def test_call_tool_mem_add_antipattern_appends_suggestion():
    """An anti-pattern mem_add with a matching pattern appends the suggestion block."""
    mock_add = MagicMock()
    mock_add.execute.return_value = 1

    matching_pattern = _make_query_result("pat_x", score=0.80, text="Best practice: always commit tests")

    mock_query = MagicMock()
    mock_query.execute.return_value = [matching_pattern]

    async def _run():
        return await server_module.call_tool("mem_add", {
            "documents": ["Tried: skipping tests\nFailed because: regression\nInstead: commit tests"],
            "ids": ["ap_x"],
            "collection": "workspace",
            "type": "anti-pattern",
        })

    with patch.object(server_module, "_add", mock_add), \
         patch.object(server_module, "_query", mock_query):
        result = asyncio.run(_run())

    assert len(result) == 1
    text = result[0].text
    assert "Stored 1 entry" in text
    assert "💡 Related patterns found" in text
    assert 'source_id="ap_x"' in text
    assert 'target_id="pat_x"' in text
