#!/usr/bin/env python3
"""ai-mem — semantic memory MCP server backed by ChromaDB.

Collections are project-scoped slugs ('exodeck', 'aide', …).
The default collection is 'workspace' — suitable for most use cases.

Data persists at $AI_MEM_PATH (default: ~/.local/share/ai-mem/).
"""
import asyncio
import json
import os
from pathlib import Path

import chromadb
import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

DEFAULT_COLLECTION = "workspace"
DB_PATH = Path(os.environ.get("AI_MEM_PATH", Path.home() / ".local" / "share" / "ai-mem"))
DB_PATH.mkdir(parents=True, exist_ok=True)

_client = chromadb.PersistentClient(path=str(DB_PATH))
server = Server("ai-mem")


def _col(name: str):
    return _client.get_or_create_collection(name)


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="mem_add",
            description=(
                "Store or update information in memory. "
                f"Leave 'collection' empty to use the default ('{DEFAULT_COLLECTION}'). "
                "Use project-slug names for project-specific memory: 'exodeck', 'aide', etc."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "documents": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Text entries to store",
                    },
                    "ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Unique ID for each entry (used for updates)",
                    },
                    "collection": {
                        "type": "string",
                        "description": f"Collection name (default: '{DEFAULT_COLLECTION}')",
                    },
                    "metadatas": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Optional metadata per entry (e.g. tags, date)",
                    },
                },
                "required": ["documents", "ids"],
            },
        ),
        types.Tool(
            name="mem_query",
            description=(
                "Search memory semantically. Returns ranked results with similarity scores. "
                f"Leave 'collection' empty to search the default ('{DEFAULT_COLLECTION}')."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query",
                    },
                    "collection": {
                        "type": "string",
                        "description": f"Collection to search (default: '{DEFAULT_COLLECTION}')",
                    },
                    "n_results": {
                        "type": "integer",
                        "default": 5,
                        "description": "Number of results to return",
                    },
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
            description=(
                "Delete entries from memory by ID. "
                "Omit 'ids' to drop the entire collection."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": f"Collection name (default: '{DEFAULT_COLLECTION}')",
                    },
                    "ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Entry IDs to delete (omit to drop entire collection)",
                    },
                },
                "required": [],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    collection = arguments.get("collection") or DEFAULT_COLLECTION

    if name == "mem_add":
        col = _col(collection)
        col.upsert(
            documents=arguments["documents"],
            ids=arguments["ids"],
            metadatas=arguments.get("metadatas"),
        )
        count = len(arguments["documents"])
        return [types.TextContent(type="text", text=f"Stored {count} entry/entries in '{collection}'.")]

    if name == "mem_query":
        col = _col(collection)
        n = arguments.get("n_results", 5)
        results = col.query(query_texts=[arguments["query"]], n_results=n)
        docs = results["documents"][0]
        ids = results["ids"][0]
        metas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        empty_meta: dict = {}
        out = [
            {
                "rank": i + 1,
                "id": id_,
                "score": round(1.0 - dist, 4),
                "metadata": meta or empty_meta,
                "text": doc,
            }
            for i, (doc, id_, meta, dist) in enumerate(
                zip(docs, ids, metas or [empty_meta] * len(docs), distances)
            )
        ]
        return [types.TextContent(type="text", text=json.dumps(out, indent=2, ensure_ascii=False))]

    if name == "mem_list":
        cols = _client.list_collections()
        result = [{"name": c.name, "count": c.count()} for c in cols]
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    if name == "mem_delete":
        ids = arguments.get("ids")
        if ids:
            col = _col(collection)
            col.delete(ids=ids)
            return [types.TextContent(type="text", text=f"Deleted {len(ids)} entry/entries from '{collection}'.")]
        _client.delete_collection(collection)
        return [types.TextContent(type="text", text=f"Dropped collection '{collection}'.")]

    raise ValueError(f"Unknown tool: {name}")


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
