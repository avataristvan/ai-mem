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

from ai_mem.application.add_edge import AddEdgeUseCase
from ai_mem.application.add_memory import AddMemoryUseCase
from ai_mem.application.build_features import BuildFeaturesUseCase
from ai_mem.application.cleanup_memory import CleanupMemoryUseCase
from ai_mem.application.delete_memory import DeleteMemoryUseCase
from ai_mem.application.detect_split_hints import DetectSplitHintsUseCase
from ai_mem.application.dream_memory import DreamMemoryUseCase, MODES
from ai_mem.application.get_edges import GetEdgesUseCase
from ai_mem.application.list_collections import ListCollectionsUseCase
from ai_mem.application.list_entries import ListEntriesUseCase
from ai_mem.application.load_ranker_config import LoadRankerConfigUseCase
from ai_mem.application.query_memory import QueryMemoryUseCase
from ai_mem.application.ranker_registry import RankerRegistry
from ai_mem.application.split_memory import SplitMemoryUseCase
from ai_mem.application.track_access import TrackAccessUseCase
from ai_mem.application.train_ranker import TrainRankerUseCase
from ai_mem.domain.learning import RankerScope
from ai_mem.infrastructure.chroma_repository import ChromaMemoryRepository
from ai_mem.infrastructure.ranker_storage import RankerStorage

DEFAULT_COLLECTION = "workspace"

_db_path = Path(os.environ.get("AI_MEM_PATH", Path.home() / ".local" / "share" / "ai-mem"))
_inner_repo = ChromaMemoryRepository(_db_path)
try:
    from ai_mem.infrastructure.bm25_repository import BM25MemoryRepository
    _repo = BM25MemoryRepository(_inner_repo)
except ImportError:
    _repo = _inner_repo
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
_list_entries = ListEntriesUseCase(_repo)
_detect_split_hints = DetectSplitHintsUseCase()
_dream = DreamMemoryUseCase(_repo)
_split = SplitMemoryUseCase(_repo, _add)
_add_edge = AddEdgeUseCase(_repo)
_get_edges = GetEdgesUseCase(_repo)

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
                    "type": {"type": "string", "description": "Optional entry type tag (e.g. 'feedback', 'reference', 'project', 'user', 'anti-pattern') for later filtering. For type='anti-pattern', use this mandatory structure: 'Tried: <approach>\\nFailed because: <reason>\\nInstead: <alternative>'"},
                },
                "required": ["documents", "ids"],
            },
        ),
        types.Tool(
            name="mem_query",
            description=(
                "Search memory semantically. Returns an object with 'results' (ranked entries with "
                "similarity scores and confidence) and 'split_hints' (entries with high access_count "
                "and long text that may benefit from being split into more granular sub-topics). "
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
                    "type": {"type": "string", "description": "Only return entries with this type tag (e.g. 'feedback', 'reference')"},
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="mem_list",
            description=(
                "List memory collections with their entry counts. "
                "If 'collection' is provided, returns all entries in that collection as a list of {id, title} pairs "
                "(title = first non-empty line of the entry text, max 80 characters)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "collection": {"type": "string", "description": "Collection to list entries for (omit to list all collections)"},
                },
            },
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
            name="mem_dream",
            description=(
                "Consolidate memories using Claude models — identifies contradictions, redundancies, "
                "stale entries, and undocumented emergent principles. "
                "Modes: 'single-haiku', 'single-sonnet', 'hier' (Haiku fast pass → Sonnet synthesis, default), "
                "'team' (4-turn Haiku↔Sonnet exchange). "
                "Invoked via the claude CLI — no API key required. "
                "Returns a structured diff proposal. Set 'auto_apply' to true to automatically "
                "execute DELETE actions identified by the synthesis. "
                "Use 'focus_hint' to steer consolidation — e.g. for expert collections: "
                "'These are cross-project learnings. Flag entries too project-specific to be worth keeping.'"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "collection": {"type": "string", "description": "Collection to consolidate (omit for all)"},
                    "mode": {
                        "type": "string",
                        "enum": list(MODES),
                        "default": "hier",
                        "description": "Consolidation mode (default: hier)",
                    },
                    "auto_apply": {
                        "type": "boolean",
                        "default": False,
                        "description": "Automatically delete entries identified as safe to remove",
                    },
                    "focus_hint": {
                        "type": "string",
                        "description": (
                            "Optional instruction to steer consolidation. "
                            "For expert collections: 'These are cross-project learnings from a <role> agent. "
                            "Flag entries too project-specific to keep cross-project. "
                            "Prefer DELETE over MERGE for project-specific entries.'"
                        ),
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
                "Pass 'negative_ids' to explicitly mark entries as unhelpful (0.0) — "
                "overrides the automatic positive label set at query time. "
                "Omit 'collection' to train all known collections (negative_ids ignored in that mode)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "collection": {"type": "string", "description": "Collection to train (omit for all)"},
                    "negative_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Memory IDs to label as 0.0 (not useful this session). Requires 'collection'.",
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="mem_split",
            description=(
                "Split a coarse memory entry into 2-3 focused sub-entries using Claude. "
                "If 'entry_id' is omitted, auto-splits all hinted entries in the collection "
                f"(access_count ≥ {5} and text ≥ {150} chars). "
                "The original entry is deleted and replaced by the sub-entries. "
                f"Leave 'collection' empty to use the default ('{DEFAULT_COLLECTION}')."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "collection": {"type": "string", "description": f"Collection to split in (default: '{DEFAULT_COLLECTION}')"},
                    "entry_id": {"type": "string", "description": "ID of a specific entry to split (omit to auto-split all hinted entries)"},
                },
                "required": [],
            },
        ),
        types.Tool(
            name="mem_link",
            description=(
                "Add a typed causal edge between two memory entries. "
                "Useful for linking an anti-pattern to the pattern it contradicts, "
                "a bug fix to the cause, or related concepts. "
                "When either entry is retrieved by mem_query, its linked partner is automatically "
                "appended to the results (1-hop, budget: 2 linked entries per query). "
                "Edge types: 'contradicts' | 'fixes' | 'causes' | 'related'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source_id": {"type": "string", "description": "ID of the source entry"},
                    "target_id": {"type": "string", "description": "ID of the target entry"},
                    "edge_type": {
                        "type": "string",
                        "enum": ["contradicts", "fixes", "causes", "related"],
                        "description": "Relationship type from source to target",
                    },
                    "collection": {"type": "string", "description": f"Collection containing both entries (default: '{DEFAULT_COLLECTION}')"},
                },
                "required": ["source_id", "target_id", "edge_type"],
            },
        ),
        types.Tool(
            name="mem_edges",
            description="List all outgoing causal edges for a memory entry.",
            inputSchema={
                "type": "object",
                "properties": {
                    "entry_id": {"type": "string", "description": "ID of the entry to inspect"},
                    "collection": {"type": "string", "description": f"Collection containing the entry (default: '{DEFAULT_COLLECTION}')"},
                },
                "required": ["entry_id"],
            },
        ),
    ]


_PATTERN_LINK_THRESHOLD = 0.4
_PATTERN_LINK_MAX = 2
CONTRADICTION_THRESHOLD = 0.75
_CONTRADICTION_MAX = 3


def _suggest_pattern_links(collection: str, documents: list[str], stored_ids: list[str]) -> str:
    """Return a suggestion block for linking anti-pattern entries to related patterns.

    Queries for type=pattern entries similar to each stored document. Returns an
    empty string when no matches exceed the threshold or when the query fails.
    """
    try:
        lines: list[str] = []
        seen_pairs: set[tuple[str, str]] = set()
        for doc, source_id in zip(documents, stored_ids):
            results = _query.execute(
                collection=collection,
                query=doc,
                n_results=_PATTERN_LINK_MAX,
                type_filter="pattern",
            )
            for r in results:
                if r.score < _PATTERN_LINK_THRESHOLD:
                    continue
                pair = (source_id, r.id)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                preview = r.text[:60].replace("\n", " ")
                lines.append(
                    f'  mem_link(source_id="{source_id}", target_id="{r.id}",'
                    f' edge_type="contradicts", collection="{collection}")'
                )
                lines.append(f"  # {preview}")
        if not lines:
            return ""
        return "💡 Related patterns found — consider linking:\n" + "\n".join(lines)
    except Exception:
        return ""


def _detect_contradictions(collection: str, type_tag: str, documents: list[str]) -> list[dict]:
    """Query the opposite type to find possible contradictions above the threshold.

    Returns a list of {id, score, preview} dicts for hits at or above
    CONTRADICTION_THRESHOLD. Returns an empty list when the query fails.
    """
    opposite = "anti-pattern" if type_tag == "pattern" else "pattern"
    hits: list[dict] = []
    try:
        for doc in documents:
            results = _query.execute(
                collection=collection,
                query=doc,
                n_results=_CONTRADICTION_MAX,
                type_filter=opposite,
            )
            seen_ids = {h["id"] for h in hits}
            for r in results:
                if (r.score or 0.0) >= CONTRADICTION_THRESHOLD and r.id not in seen_ids:
                    hits.append({"id": r.id, "score": round(r.score, 2), "preview": r.text[:120]})
    except Exception:
        pass
    return hits


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    collection = arguments.get("collection") or DEFAULT_COLLECTION

    if name == "mem_add":
        metadatas = arguments.get("metadatas")
        type_tag = arguments.get("type")
        if type_tag is not None:
            n = len(arguments["documents"])
            if metadatas is None:
                metadatas = [{} for _ in range(n)]
            for m in metadatas:
                m.setdefault("type", type_tag)
        count = _add.execute(
            collection=collection,
            documents=arguments["documents"],
            ids=arguments["ids"],
            metadatas=metadatas,
            ttl_days=arguments.get("ttl_days"),
        )
        base_text = f"Stored {count} entry/entries in '{collection}'."

        if type_tag == "anti-pattern":
            suggestion = _suggest_pattern_links(
                collection=collection,
                documents=arguments["documents"],
                stored_ids=arguments["ids"],
            )
            if suggestion:
                base_text = f"{base_text}\n\n{suggestion}"

        if type_tag in ("pattern", "anti-pattern"):
            contradictions = _detect_contradictions(
                collection=collection,
                type_tag=type_tag,
                documents=arguments["documents"],
            )
            if contradictions:
                base_text = (
                    f"{base_text}\n\npossible_contradictions: "
                    + json.dumps(contradictions, ensure_ascii=False)
                )

        return [types.TextContent(type="text", text=base_text)]

    if name == "mem_query":
        results = _query.execute(
            collection=collection,
            query=arguments["query"],
            n_results=arguments.get("n_results", 5),
            max_age_days=arguments.get("max_age_days"),
            type_filter=arguments.get("type"),
        )
        split_hints = _detect_split_hints.execute(results)
        out = {
            "results": [
                {
                    "rank": r.rank,
                    "id": r.id,
                    "score": r.score,
                    "confidence": int(r.metadata.get("access_count", 0)),
                    "metadata": r.metadata,
                    "text": r.text,
                }
                for r in results
            ],
            "split_hints": [
                {"id": h.id, "text_preview": h.text_preview, "access_count": h.access_count}
                for h in split_hints
            ],
        }
        return [types.TextContent(type="text", text=json.dumps(out, indent=2, ensure_ascii=False))]

    if name == "mem_list":
        col_arg = arguments.get("collection")
        if col_arg:
            entries = _list_entries.execute(col_arg)
            return [types.TextContent(type="text", text=json.dumps(entries, indent=2, ensure_ascii=False))]
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

    if name == "mem_dream":
        mode = arguments.get("mode") or "hier"
        col_arg = arguments.get("collection") or None
        auto_apply = bool(arguments.get("auto_apply", False))
        focus_hint = arguments.get("focus_hint") or None
        result = await asyncio.to_thread(_dream.execute, col_arg, mode, auto_apply, focus_hint)
        return [types.TextContent(type="text", text=result)]

    if name == "mem_train":
        now = time.time()
        col_arg = arguments.get("collection")
        negative_ids = arguments.get("negative_ids") or []
        explicit_labels = {id_: 0.0 for id_ in negative_ids} if negative_ids else None
        if col_arg:
            metrics = _train_ranker.train_step(col_arg, now, explicit_labels=explicit_labels)
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

    if name == "mem_split":
        entry_id = arguments.get("entry_id") or None
        results = await asyncio.to_thread(_split.execute, collection, entry_id)
        out = [
            {
                "original_id": r.original_id,
                "new_ids": r.new_ids,
                "skipped": r.skipped,
                **({"skip_reason": r.skip_reason} if r.skipped else {}),
            }
            for r in results
        ]
        total = len(results)
        succeeded = sum(1 for r in results if not r.skipped)
        summary = f"Split {succeeded}/{total} entries."
        return [types.TextContent(type="text", text=f"{summary}\n{json.dumps(out, indent=2)}")]

    if name == "mem_link":
        try:
            _add_edge.execute(
                collection=collection,
                source_id=arguments["source_id"],
                target_id=arguments["target_id"],
                edge_type=arguments["edge_type"],
            )
            return [types.TextContent(
                type="text",
                text=f"Linked '{arguments['source_id']}' --[{arguments['edge_type']}]--> '{arguments['target_id']}' in '{collection}'.",
            )]
        except ValueError as exc:
            return [types.TextContent(type="text", text=f"Error: {exc}")]

    if name == "mem_edges":
        edges = _get_edges.execute(
            collection=collection,
            entry_id=arguments["entry_id"],
        )
        out = [{"target_id": e.target_id, "edge_type": e.edge_type} for e in edges]
        return [types.TextContent(type="text", text=json.dumps(out, indent=2))]

    raise ValueError(f"Unknown tool: {name}")


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
