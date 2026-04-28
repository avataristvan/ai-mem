#!/usr/bin/env python3
"""ai-mem MCP server — thin adapter over the application layer."""
import asyncio
import json
import os
import time
from pathlib import Path

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from ai_mem.application.add_memory import AddMemoryUseCase
from ai_mem.application.build_features import BuildFeaturesUseCase
from ai_mem.application.cleanup_memory import CleanupMemoryUseCase
from ai_mem.application.delete_memory import DeleteMemoryUseCase
from ai_mem.application.list_collections import ListCollectionsUseCase
from ai_mem.application.load_ranker_config import LoadRankerConfigUseCase
from ai_mem.application.query_memory import QueryMemoryUseCase
from ai_mem.application.ranker_registry import RankerRegistry
from ai_mem.application.track_access import TrackAccessUseCase
from ai_mem.application.train_ranker import TrainRankerUseCase
from ai_mem.domain.learning import RankerScope
from ai_mem.infrastructure.chroma_repository import ChromaMemoryRepository
from ai_mem.infrastructure.ranker_storage import RankerStorage

DEFAULT_COLLECTION = "workspace"

_db_path = Path(os.environ.get("AI_MEM_PATH", Path.home() / ".local" / "share" / "ai-mem"))
_repo = ChromaMemoryRepository(_db_path)
_storage = RankerStorage(_db_path / "rankers")

try:
    from ai_mem.infrastructure.torch_ranker import TorchMicroRanker as _RankerClass
except ImportError:
    from ai_mem.infrastructure.null_ranker import NullRanker as _RankerClass  # type: ignore[assignment]

_scope_map = LoadRankerConfigUseCase(_db_path / "ranker_config.json").execute()


def _scope_resolver(collection: str) -> RankerScope:
    return _scope_map.get(collection, RankerScope(name=collection, mode="isolated"))


_registry = RankerRegistry(
    scope_resolver=_scope_resolver,
    ranker_factory=_RankerClass,
    storage=_storage,
)

_track_access = TrackAccessUseCase(_repo)
_build_features = BuildFeaturesUseCase()
_train_ranker = TrainRankerUseCase(_repo, _storage, _RankerClass, scope_resolver=_scope_resolver)
_add = AddMemoryUseCase(_repo)
_query = QueryMemoryUseCase(_repo, _track_access, _build_features, _train_ranker, _registry)
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
                "Use the repo collection injected at session start (e.g. 'repo.ai-mem') for repo-specific context, "
                "or 'global' for cross-session knowledge shared across all repos. "
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
                "Use the repo collection injected at session start (e.g. 'repo.ai-mem') for repo-specific context, "
                "or 'global' for cross-session general knowledge. "
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
                "Delete expired entries (TTL-based). Optionally also delete stale entries — "
                "those whose last access is older than 'stale_after_days' (forgetting curve). "
                "Omit 'collection' to clean all collections."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "collection": {"type": "string", "description": "Collection to clean (omit for all)"},
                    "stale_after_days": {
                        "type": "number",
                        "description": "Also delete entries not accessed for this many days. Omit to skip stale cleanup.",
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="mem_train",
            description=(
                "Run a training step for the learned re-ranker. "
                "Reads the query buffer, assigns labels from access history, and performs one gradient step. "
                "Omit 'collection' to train all known collections."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "collection": {"type": "string", "description": "Collection to train (omit for all)"},
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
        result = _cleanup.execute(col_arg, stale_after_days=arguments.get("stale_after_days"))
        detail = json.dumps(
            {k: {"expired": v.expired, "stale": v.stale} for k, v in result.collections.items()},
            indent=2,
        )
        return [types.TextContent(type="text", text=f"Cleaned up {result.total} entry/entries.\n{detail}")]

    if name == "mem_train":
        now = time.time()
        col_arg = arguments.get("collection")
        if col_arg:
            metrics = _train_ranker.train_step(col_arg, now)
            out = {"collection": col_arg, "n": metrics.n, "loss": metrics.loss, "skipped": metrics.skipped}
        else:
            # Deduplicate by scope key: hybrid-mode group members all map to the
            # same scope, and training each member would re-train (and overwrite)
            # the shared weights using a buffer that the first pass already drained.
            collections = [c.name for c in _list.execute()]
            seen_scopes: set[str] = set()
            results_list = []
            for col in collections:
                key = _registry.scope_key(col)
                if key in seen_scopes:
                    continue
                seen_scopes.add(key)
                m = _train_ranker.train_step(col, now)
                results_list.append({"collection": col, "scope": key, "n": m.n, "loss": m.loss, "skipped": m.skipped})
            out = results_list  # type: ignore[assignment]
        return [types.TextContent(type="text", text=json.dumps(out, indent=2))]

    raise ValueError(f"Unknown tool: {name}")


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
