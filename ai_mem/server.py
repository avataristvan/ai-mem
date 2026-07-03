#!/usr/bin/env python3
"""ai-mem MCP server — thin adapter over the application layer."""
import asyncio
import json
import os
from pathlib import Path

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from ai_mem.application.add_edge import AddEdgeUseCase
from ai_mem.application.add_memory import AddMemoryUseCase
from ai_mem.application.boost_confidence import BoostConfidenceUseCase
from ai_mem.application.cleanup_memory import CleanupMemoryUseCase
from ai_mem.application.delete_memory import DeleteMemoryUseCase
from ai_mem.application.detect_split_hints import DetectSplitHintsUseCase
from ai_mem.application.dream_memory import DreamMemoryUseCase, MODES
from ai_mem.application.get_edges import GetEdgesUseCase
from ai_mem.application.get_memory import GetMemoryUseCase
from ai_mem.application.list_collections import ListCollectionsUseCase
from ai_mem.application.list_entries import ListEntriesUseCase
from ai_mem.application.move_memory import MoveMemoryUseCase
from ai_mem.application.query_memory import QueryMemoryUseCase
from ai_mem.application.split_memory import SplitMemoryUseCase
from ai_mem.application.track_access import TrackAccessUseCase
from ai_mem.infrastructure.chroma_repository import ChromaMemoryRepository

DEFAULT_COLLECTION = "workspace"

_db_path = Path(os.environ.get("AI_MEM_PATH", Path.home() / ".local" / "share" / "ai-mem"))
_inner_repo = ChromaMemoryRepository(_db_path)
try:
    from ai_mem.infrastructure.bm25_repository import BM25MemoryRepository
    _repo = BM25MemoryRepository(_inner_repo)
except ImportError:
    _repo = _inner_repo

_track_access = TrackAccessUseCase(_repo)
_add = AddMemoryUseCase(_repo)
_query = QueryMemoryUseCase(_repo, _track_access)
_list = ListCollectionsUseCase(_repo)
_delete = DeleteMemoryUseCase(_repo)
_cleanup = CleanupMemoryUseCase(_repo)
_list_entries = ListEntriesUseCase(_repo)
_detect_split_hints = DetectSplitHintsUseCase()
_dream = DreamMemoryUseCase(_repo)
_split = SplitMemoryUseCase(_repo, _add)
_add_edge = AddEdgeUseCase(_repo)
_get_edges = GetEdgesUseCase(_repo)
_get_memory = GetMemoryUseCase(_repo)
_move = MoveMemoryUseCase(_repo)

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
                "When the user asks to store something for a different project (e.g. 'note for my-app'), "
                "pass that project's collection explicitly (e.g. 'repo.my-app') — never rely on the active collection default for cross-project intent. "
                "Set 'ttl_days' to expire the entry automatically (e.g. 30 for one month). "
                "When updating an existing entry (same id), existing metadata is preserved by default — "
                "confidence, access_count, boost_count, created_at, edges, and type are all kept intact. "
                "You do NOT need to re-pass existing metadata fields for a text-only update."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "documents": {"type": "array", "items": {"type": "string"}, "description": "Text entries to store"},
                    "ids": {"type": "array", "items": {"type": "string"}, "description": "Unique ID per entry (used for updates)"},
                    "collection": {"type": "string", "description": f"Collection name (default: '{DEFAULT_COLLECTION}')"},
                    "metadatas": {"type": "array", "items": {"type": "object"}, "description": "Optional metadata per entry"},
                    "ttl_days": {"type": "number", "description": "Optional TTL in days — entry is deleted by mem_cleanup after expiry"},
                    "type": {"type": "string", "description": "Optional entry type tag (e.g. 'feedback', 'reference', 'project', 'user', 'anti-pattern', 'dilemma') for later filtering. For type='anti-pattern', use this mandatory structure: 'Tried: <approach>\\nFailed because: <reason>\\nInstead: <alternative>'. When the anti-pattern involves human stakeholders (people treated as variables, trust broken, social dynamics), add: 'Affected: <who was affected and how>'. For type='dilemma' (genuine value conflicts with no single correct answer, e.g. across cultures), use: 'Tension: <value A> vs. <value B>\\nContext A: <when value A applies and why>\\nContext B: <when value B applies and why>\\nAffected A: <who, how — from frame A>\\nAffected B: <who, how — from frame B>\\nQuestions: <what to ask to determine which context applies>'. Dilemmas encode the tension itself and the right questions, not a resolved answer."},
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
                "When the user asks to search in a different project, pass that project's collection explicitly (e.g. 'repo.my-app'). "
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
            name="mem_get",
            description="Fetch specific memory entries by ID. Use when you know the exact ID(s) — bypasses semantic search and ranking entirely.",
            inputSchema={
                "type": "object",
                "properties": {
                    "ids": {"type": "array", "items": {"type": "string"}, "description": "Entry IDs to fetch"},
                    "collection": {"type": "string", "description": f"Collection name (default: '{DEFAULT_COLLECTION}')"},
                },
                "required": ["ids"],
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
            name="mem_move",
            description=(
                "Move entries from one collection to another, preserving id, text, and all metadata "
                "verbatim (confidence, access_count, edges, type, etc.) — no manual retyping needed. "
                "If an id already exists at the target with different text, it is skipped and reported "
                "under 'conflicts' rather than overwritten (identical text at the target is treated as "
                "a no-op merge and still moved). "
                "Caveat: edges reference target_id only, with no collection field — other entries in "
                "the source collection that referenced a moved entry are not updated and may become "
                "dangling references."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ids": {"type": "array", "items": {"type": "string"}, "description": "Entry IDs to move"},
                    "collection": {"type": "string", "description": f"Source collection (default: '{DEFAULT_COLLECTION}')"},
                    "to_collection": {"type": "string", "description": "Target collection name"},
                },
                "required": ["ids", "to_collection"],
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
                "Returns a structured diff proposal. Set 'auto_delete' to true to automatically "
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
                    "auto_delete": {
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
        types.Tool(
            name="mem_boost",
            description=(
                "Apply a confidence delta to existing memory entries. "
                "Use in /reflect to boost entries that proved crucial in a session (+0.1), "
                "or to decay entries that turned out misleading (-0.1). "
                "Entries not found are silently skipped. Delta is clamped to keep confidence in [0.0, 1.0]."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ids": {"type": "array", "items": {"type": "string"}, "description": "Entry IDs to boost or decay"},
                    "delta": {"type": "number", "description": "Confidence delta, e.g. 0.1 for boost, -0.1 for decay"},
                    "collection": {"type": "string", "description": f"Collection name (default: '{DEFAULT_COLLECTION}')"},
                },
                "required": ["ids", "delta"],
            },
        ),
    ]


_PATTERN_LINK_THRESHOLD = 0.4
_PATTERN_LINK_MAX = 2
CONTRADICTION_THRESHOLD = 0.75
_CONTRADICTION_UPPER = 0.97  # near-identical embeddings are format artifacts, not semantic contradictions
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
            for result in results:
                if result.score < _PATTERN_LINK_THRESHOLD:
                    continue
                pair = (source_id, result.id)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                preview = result.text[:60].replace("\n", " ")
                lines.append(
                    f'  mem_link(source_id="{source_id}", target_id="{result.id}",'
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
            seen_ids = {hit["id"] for hit in hits}
            for result in results:
                if CONTRADICTION_THRESHOLD <= (result.score or 0.0) < _CONTRADICTION_UPPER and result.id not in seen_ids:
                    hits.append({"id": result.id, "score": round(result.score, 2), "preview": result.text[:120]})
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
            count = len(arguments["documents"])
            if metadatas is None:
                metadatas = [{} for _ in range(count)]
            for meta_entry in metadatas:
                meta_entry.setdefault("type", type_tag)
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
                    "rank": result.rank,
                    "id": result.id,
                    "score": result.score,
                    "confidence": float(result.metadata.get("confidence", 0.0)),
                    "metadata": result.metadata,
                    "text": result.text,
                }
                for result in results
            ],
            "split_hints": [
                {"id": hint.id, "text_preview": hint.text_preview, "access_count": hint.access_count}
                for hint in split_hints
            ],
        }
        return [types.TextContent(type="text", text=json.dumps(out, indent=2, ensure_ascii=False))]

    if name == "mem_get":
        entries = _get_memory.execute(collection, arguments["ids"])
        out = [{"id": entry.id, "text": entry.text, "metadata": entry.metadata} for entry in entries]
        return [types.TextContent(type="text", text=json.dumps(out, indent=2, ensure_ascii=False))]

    if name == "mem_list":
        col_arg = arguments.get("collection")
        if col_arg:
            entries = _list_entries.execute(col_arg)
            return [types.TextContent(type="text", text=json.dumps(entries, indent=2, ensure_ascii=False))]
        cols = _list.execute()
        return [types.TextContent(type="text", text=json.dumps([{"name": col_info.name, "count": col_info.count} for col_info in cols], indent=2))]

    if name == "mem_delete":
        affected = _delete.execute(collection=collection, ids=arguments.get("ids"))
        if affected == -1:
            return [types.TextContent(type="text", text=f"Dropped collection '{collection}'.")]
        return [types.TextContent(type="text", text=f"Deleted {affected} entry/entries from '{collection}'.")]

    if name == "mem_move":
        to_collection = arguments["to_collection"]
        try:
            result = _move.execute(from_collection=collection, to_collection=to_collection, ids=arguments["ids"])
        except ValueError as exc:
            return [types.TextContent(type="text", text=f"Error: {exc}")]
        lines = [f"Moved {len(result.moved_ids)} entry/entries from '{collection}' to '{to_collection}'."]
        if result.not_found_ids:
            lines.append(f"not_found: {result.not_found_ids}")
        if result.conflicts:
            lines.append("conflicts: " + json.dumps(
                [{"id": c.id, "existing_text_preview": c.existing_text_preview, "source_text_preview": c.source_text_preview} for c in result.conflicts],
                ensure_ascii=False,
            ))
        return [types.TextContent(type="text", text="\n".join(lines))]

    if name == "mem_cleanup":
        col_arg = arguments.get("collection")
        result = _cleanup.execute(col_arg, stale_after_days=arguments.get("stale_after_days"))
        detail = json.dumps(
            {col_name: {"expired": col_stats.expired, "stale": col_stats.stale} for col_name, col_stats in result.collections.items()},
            indent=2,
        )
        return [types.TextContent(type="text", text=f"Cleaned up {result.total} entry/entries.\n{detail}")]

    if name == "mem_dream":
        mode = arguments.get("mode") or "hier"
        col_arg = arguments.get("collection") or None
        auto_delete = bool(arguments.get("auto_delete", False))
        focus_hint = arguments.get("focus_hint") or None
        result = await asyncio.to_thread(_dream.execute, col_arg, mode, auto_delete, focus_hint)
        return [types.TextContent(type="text", text=result)]

    if name == "mem_split":
        entry_id = arguments.get("entry_id") or None
        results = await asyncio.to_thread(_split.execute, collection, entry_id)
        out = [
            {
                "original_id": result.original_id,
                "new_ids": result.new_ids,
                "skipped": result.skipped,
                **({"skip_reason": result.skip_reason} if result.skipped else {}),
            }
            for result in results
        ]
        total = len(results)
        succeeded = sum(1 for result in results if not result.skipped)
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
        out = [{"target_id": edge.target_id, "edge_type": edge.edge_type} for edge in edges]
        return [types.TextContent(type="text", text=json.dumps(out, indent=2))]

    if name == "mem_boost":
        col = arguments.get("collection") or DEFAULT_COLLECTION
        ids = arguments["ids"]
        delta = float(arguments["delta"])
        boosted = BoostConfidenceUseCase(_repo).execute(col, ids, delta)
        return [types.TextContent(type="text", text=f"Boosted {boosted} entr{'y' if boosted == 1 else 'ies'} in '{col}'.")]

    raise ValueError(f"Unknown tool: {name}")


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
