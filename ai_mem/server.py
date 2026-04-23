#!/usr/bin/env python3
"""ai-mem MCP server — thin adapter over the application layer."""
import asyncio
import json
import os
from pathlib import Path

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from ai_mem.application.add_memory import AddMemoryUseCase
from ai_mem.application.cleanup_memory import CleanupMemoryUseCase
from ai_mem.application.delete_memory import DeleteMemoryUseCase
from ai_mem.application.list_collections import ListCollectionsUseCase
from ai_mem.application.query_memory import QueryMemoryUseCase
from ai_mem.infrastructure.chroma_repository import ChromaMemoryRepository

DEFAULT_COLLECTION = "workspace"

_db_path = Path(os.environ.get("AI_MEM_PATH", Path.home() / ".local" / "share" / "ai-mem"))
_repo = ChromaMemoryRepository(_db_path)

_add = AddMemoryUseCase(_repo)
_query = QueryMemoryUseCase(_repo)
_list = ListCollectionsUseCase(_repo)
_delete = DeleteMemoryUseCase(_repo)
_cleanup = CleanupMemoryUseCase(_repo)

server = Server("ai-mem")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="mem_add",
            description=(
                "Store or update information in memory. "
                f"Leave 'collection' empty to use the default ('{DEFAULT_COLLECTION}'). "
                "Set 'ttl_days' to expire the entry automatically (e.g. 30 for one month)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "documents": {"type": "array", "items": {"type": "string"}, "description": "Text entries to store"},
                    "ids": {"type": "array", "items": {"type": "string"}, "description": "Unique ID per entry (used for updates)"},
                    "collection": {"type": "string", "description": f"Collection name (default: '{DEFAULT_COLLECTION}')"},
                    "metadatas": {"type": "array", "items": {"type": "object"}, "description": "Optional metadata per entry"},
                    "ttl_days": {"type": "number", "description": "Optional TTL in days — entry is deleted by mem_cleanup after expiry"},
                },
                "required": ["documents", "ids"],
            },
        ),
        types.Tool(
            name="mem_query",
            description=(
                "Search memory semantically. Returns ranked results with similarity scores. "
                f"Leave 'collection' empty to search the default ('{DEFAULT_COLLECTION}'). "
                "Use 'max_age_days' to exclude older entries."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query"},
                    "collection": {"type": "string", "description": f"Collection to search (default: '{DEFAULT_COLLECTION}')"},
                    "n_results": {"type": "integer", "default": 5, "description": "Number of results to return"},
                    "max_age_days": {"type": "number", "description": "Only return entries created within this many days"},
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="mem_list",
            description="List all memory collections with their entry counts.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="mem_delete",
            description="Delete entries from memory by ID. Omit 'ids' to drop the entire collection.",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection": {"type": "string", "description": f"Collection name (default: '{DEFAULT_COLLECTION}')"},
                    "ids": {"type": "array", "items": {"type": "string"}, "description": "Entry IDs to delete (omit to drop entire collection)"},
                },
                "required": [],
            },
        ),
        types.Tool(
            name="mem_cleanup",
            description=(
                "Delete all expired entries (those with a ttl_days set on mem_add that have passed). "
                "Omit 'collection' to clean all collections."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "collection": {"type": "string", "description": "Collection to clean (omit for all)"},
                },
                "required": [],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    collection = arguments.get("collection") or DEFAULT_COLLECTION

    if name == "mem_add":
        count = _add.execute(
            collection=collection,
            documents=arguments["documents"],
            ids=arguments["ids"],
            metadatas=arguments.get("metadatas"),
            ttl_days=arguments.get("ttl_days"),
        )
        return [types.TextContent(type="text", text=f"Stored {count} entry/entries in '{collection}'.")]

    if name == "mem_query":
        results = _query.execute(
            collection=collection,
            query=arguments["query"],
            n_results=arguments.get("n_results", 5),
            max_age_days=arguments.get("max_age_days"),
        )
        out = [{"rank": r.rank, "id": r.id, "score": r.score, "metadata": r.metadata, "text": r.text} for r in results]
        return [types.TextContent(type="text", text=json.dumps(out, indent=2, ensure_ascii=False))]

    if name == "mem_list":
        cols = _list.execute()
        return [types.TextContent(type="text", text=json.dumps([{"name": c.name, "count": c.count} for c in cols], indent=2))]

    if name == "mem_delete":
        affected = _delete.execute(collection=collection, ids=arguments.get("ids"))
        if affected == -1:
            return [types.TextContent(type="text", text=f"Dropped collection '{collection}'.")]
        return [types.TextContent(type="text", text=f"Deleted {affected} entry/entries from '{collection}'.")]

    if name == "mem_cleanup":
        col_arg = arguments.get("collection")
        result = _cleanup.execute(col_arg)
        total = sum(result.values())
        detail = json.dumps(result, indent=2)
        return [types.TextContent(type="text", text=f"Cleaned up {total} expired entry/entries.\n{detail}")]

    raise ValueError(f"Unknown tool: {name}")


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
