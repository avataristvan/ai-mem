"""Integration test for the mem_topics tool dispatch in server.py."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import ai_mem.server as server_module


def test_call_tool_mem_topics_dispatches_to_list_topics_with_resolved_collection():
    mock_list_topics = MagicMock()
    mock_list_topics.execute.return_value = [
        {"type": "pattern", "count": 1, "entries": [{"id": "p1", "title": "A pattern"}]},
    ]

    async def _run():
        return await server_module.call_tool("mem_topics", {"collection": "repo.ai-mem"})

    with patch.object(server_module, "_list_topics", mock_list_topics):
        result = asyncio.run(_run())

    mock_list_topics.execute.assert_called_once_with("repo.ai-mem")
    payload = json.loads(result[0].text)
    assert payload == [{"type": "pattern", "count": 1, "entries": [{"id": "p1", "title": "A pattern"}]}]


def test_call_tool_mem_topics_defaults_collection_when_omitted():
    mock_list_topics = MagicMock()
    mock_list_topics.execute.return_value = []

    async def _run():
        return await server_module.call_tool("mem_topics", {})

    with patch.object(server_module, "_list_topics", mock_list_topics):
        asyncio.run(_run())

    mock_list_topics.execute.assert_called_once_with(server_module.DEFAULT_COLLECTION)
