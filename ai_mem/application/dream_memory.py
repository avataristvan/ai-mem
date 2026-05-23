"""DreamMemoryUseCase — consolidate memories via Claude models."""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime

from ai_mem.domain.memory import MemoryEntry, MemoryRepository

_DELETE_RE = re.compile(r"^\s*[-*•]\s+DELETE\s+(\S+)\s*:", re.MULTILINE | re.IGNORECASE)
_ADD_TARGET_RE = re.compile(
    r"^\s*[-*•]\s+ADD\s+(\S+)\s+\[target=([^\]]+)\]\s*:",
    re.MULTILINE | re.IGNORECASE,
)
_MERGE_RE = re.compile(
    r"^\s*[-*•]\s+MERGE\s+(\S+)\s*\+\s*(\S+)\s*:",
    re.MULTILINE | re.IGNORECASE,
)
_LINK_RE = re.compile(
    r"^\s*[-*•]\s+LINK\s+(\S+)\s*->\s*(\S+)\s+\[type=([^\]]+)\]\s*:",
    re.MULTILINE | re.IGNORECASE,
)

_VALID_EDGE_TYPES = frozenset({"contradicts", "fixes", "causes", "related"})

MODELS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
}

MODES = ("single-haiku", "single-sonnet", "hier", "team")

_TYPE_RULES = """\
TYPE RULES — apply these strictly:
- "pattern": a reusable rule or best practice. Canonical format: Rule: / When: / Why:
- "anti-pattern": a documented failure mode. Canonical format: Tried: / Failed because: / Instead:
- "feedback", "project", "reference", "user": descriptive entries — no fixed format.

Consolidation constraints:
- NEVER propose MERGE between a "pattern" and an "anti-pattern". They contradict by design. \
If they share an edge (shown in metadata), verify the relationship is accurate — do not collapse it.
- Entries with access_count ≥ 5 are load-bearing (frequently retrieved). \
Avoid DELETE or MERGE unless clearly redundant.
- If two entries share an edge, the edge makes both necessary. \
Verify the relationship before proposing structural changes.
- If two related entries have no edge yet, propose LINK to document \
the relationship explicitly instead of leaving it implicit."""

_COLLECTION_CONTEXT = """\
COLLECTIONS PRESENT:
{collections}

COLLECTION HIERARCHY (lower → higher scope):
  repo.<name>  →  workspace  →  global

A pattern documented in a repo collection that recurs across multiple projects belongs \
in a higher-level collection. Use the target= field on ADD proposals to express this."""

_ACTION_FORMAT = """\
Format each action as:
- UPDATE <id>: <what to change>
- MERGE <id1> + <id2>: <into what>
- DELETE <id>: <reason>
- ADD <suggested-id> [target=<collection>]: <content summary>
- LINK <source-id> -> <target-id> [type=<edge_type>]: <reason>
  (edge_type: contradicts | fixes | causes | related)

For ADD, set target= to the collection where the entry belongs:
  - Same collection as the source entry if it is project-specific.
  - A higher-level collection (workspace or global) if the pattern applies broadly \
across multiple projects — this flags it as a propagation candidate.

For LINK: use when two entries are related but must stay separate \
(e.g. an anti-pattern contradicting a pattern, a fix addressing a cause). \
Prefer LINK over MERGE when entries have different types or already share an edge."""

_P_SINGLE = """\
You are a memory consolidation agent. The memories below come from AI assistant sessions \
(stored as text entries with metadata). Analyze them and return a structured, actionable proposal:

1. **Contradictions** — entries that conflict with each other (name entry IDs)
2. **Redundancies** — entries that overlap and could be merged
3. **Stale entries** — entries likely outdated (explain why)
4. **Missing principles** — patterns that emerge across entries but aren't yet documented

{action_format}

{collection_context}

MEMORIES:
{memories}"""

_P_HAIKU = """\
You are doing a fast first-pass memory consolidation. Focus on what's obvious:
- Direct contradictions between entries
- Clear redundancies (same fact stated in multiple entries)
- Entries with explicit dates or version references that are likely stale
- Patterns that recur across repo collections and belong in a higher-level collection

Be concise. Reference entry IDs and source collections explicitly. \
A more capable model will review your output.

{collection_context}

MEMORIES:
{memories}"""

_P_SONNET_HIER = """\
You are reviewing a fast first-pass memory consolidation. Deepen and validate it.

- Confirm or correct the first-pass findings
- Add what it missed (subtle contradictions, cross-entry patterns)
- Identify emergent principles across entries not yet documented
- Identify cross-project patterns that should propagate to workspace or global

{action_format}

{collection_context}

MEMORIES:
{memories}

FIRST-PASS ANALYSIS:
{a}"""

_P_SONNET_CRITIQUE = """\
You are the second voice in a memory consolidation debate. A faster model gave an initial analysis.
Challenge it: what did it miss? Where is it wrong? What subtle patterns does it overlook?
Also add your own findings. Be direct.

{collection_context}

MEMORIES:
{memories}

INITIAL ANALYSIS:
{a}"""

_P_HAIKU_REBUTTAL = """\
You gave an initial memory analysis. A more capable model critiqued it. Respond:
- Defend what you got right
- Concede where the critique is valid
- Add anything this exchange surfaced

YOUR INITIAL ANALYSIS:
{a}

CRITIQUE:
{b}"""

_P_SONNET_FINAL = """\
Synthesize the best insights from this full debate into one clean, actionable proposal.

{action_format}

{collection_context}

INITIAL ANALYSIS:
{a}

CRITIQUE:
{b}

REBUTTAL:
{c}"""


def _build_edge_index(entries: list[MemoryEntry]) -> dict[tuple[str, str], str]:
    """Return {(source_id, target_id): edge_type} from loaded entries' metadata."""
    index: dict[tuple[str, str], str] = {}
    for entry in entries:
        raw = entry.metadata.get("edges", "[]")
        try:
            for ed in json.loads(raw):
                index[(entry.id, ed["target_id"])] = ed["edge_type"]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return index


def _check_merge_conflicts(
    synthesis: str, edge_index: dict[tuple[str, str], str]
) -> list[str]:
    """Return warning strings for MERGE proposals that conflict with a contradicts edge."""
    warnings = []
    for m in _MERGE_RE.finditer(synthesis):
        a, b = m.group(1), m.group(2)
        if edge_index.get((a, b)) == "contradicts" or edge_index.get((b, a)) == "contradicts":
            warnings.append(
                f"⚠ MERGE {a} + {b} conflicts with an existing `contradicts` edge — "
                f"use LINK to document the relationship instead."
            )
    return warnings


def _parse_link_proposals(synthesis: str) -> list[tuple[str, str, str]]:
    """Return (source_id, target_id, edge_type) for each LINK proposal in synthesis."""
    return [
        (m.group(1), m.group(2), m.group(3).strip())
        for m in _LINK_RE.finditer(synthesis)
    ]


def _format_entries(entries: list[MemoryEntry]) -> str:
    parts = []
    for e in entries:
        col = e.metadata.get("_collection", "?")
        shown: dict[str, object] = {}
        if t := e.metadata.get("type"):
            shown["type"] = t
        if ac := e.metadata.get("access_count"):
            shown["access_count"] = int(ac)
        if raw_edges := e.metadata.get("edges"):
            try:
                edges = json.loads(raw_edges)
                if edges:
                    shown["edges"] = [f"{ed['target_id']}({ed['edge_type']})" for ed in edges]
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
        meta_str = (
            " [" + ", ".join(f"{k}={v}" for k, v in shown.items()) + "]"
            if shown else ""
        )
        parts.append(f"[{e.id}] (collection: {col}){meta_str}\n{e.text}")
    return "\n\n---\n\n".join(parts) if parts else "(empty)"


def _call(model_key: str, prompt: str) -> str:
    result = subprocess.run(
        ["claude", "--print", "--model", MODELS[model_key]],
        input=prompt,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _collection_context(collections: list[str]) -> str:
    return _COLLECTION_CONTEXT.format(collections="\n".join(f"  - {c}" for c in collections))


def _propagation_candidates(synthesis: str, source_collections: set[str]) -> list[tuple[str, str]]:
    """Return (entry_id, target_collection) pairs where target differs from all source collections."""
    candidates = []
    for m in _ADD_TARGET_RE.finditer(synthesis):
        entry_id, target = m.group(1), m.group(2).strip()
        if target not in source_collections:
            candidates.append((entry_id, target))
    return candidates


class DreamMemoryUseCase:
    def __init__(self, repo: MemoryRepository) -> None:
        self._repo = repo

    def execute(
        self,
        collection: str | None,
        mode: str,
        auto_delete: bool = False,
        auto_link: bool = False,
        focus_hint: str | None = None,
    ) -> str:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")

        collections = (
            [collection]
            if collection
            else [c.name for c in self._repo.list_collections()]
        )

        all_entries = []
        for col in collections:
            entries = self._repo.get_all(col)
            for e in entries:
                e.metadata["_collection"] = col
            all_entries.extend(entries)

        if not all_entries:
            return "No memories found."

        preamble_parts = [_TYPE_RULES]
        if focus_hint:
            preamble_parts.append(f"FOCUS:\n{focus_hint}")
        memories = "\n\n".join(preamble_parts) + "\n\n---\n\n" + _format_entries(all_entries)
        col_ctx = _collection_context(collections)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")

        if mode in ("single-haiku", "single-sonnet"):
            synthesis, report = self._run_single(mode.split("-", 1)[1], ts, memories, col_ctx)
        elif mode == "hier":
            synthesis, report = self._run_hier(ts, memories, col_ctx)
        else:
            synthesis, report = self._run_team(ts, memories, col_ctx)

        edge_index = _build_edge_index(all_entries)
        conflicts = _check_merge_conflicts(synthesis, edge_index)
        if conflicts:
            report += "\n\n---\n\n## ⚠ Merge Conflicts\n\n" + "\n".join(conflicts)

        propagation = _propagation_candidates(synthesis, set(collections))
        if propagation:
            lines = "\n".join(f"- `{eid}` → **{target}**" for eid, target in propagation)
            report += (
                "\n\n---\n\n## Propagation Candidates\n\n"
                "These ADD proposals target a higher-level collection "
                "(review and apply with `mem_add` if confirmed):\n\n" + lines
            )

        if auto_link:
            linked = self._auto_apply_links(synthesis, all_entries)
            if linked:
                report += "\n\n---\n\n## Auto-Applied Links\n\n" + "\n".join(
                    f"- Linked `{l}`" for l in linked
                )

        if auto_delete:
            deleted = self._auto_apply_deletes(synthesis, all_entries)
            if deleted:
                report += "\n\n---\n\n## Auto-Applied Deletions\n\n" + "\n".join(
                    f"- Deleted `{d}`" for d in deleted
                )
            else:
                report += "\n\n---\n\n## Auto-Applied Deletions\n\nNone (no high-confidence DELETE actions found)."

        return report

    def _run_single(self, model_key: str, ts: str, memories: str, col_ctx: str) -> tuple[str, str]:
        result = _call(model_key, _P_SINGLE.format(
            memories=memories, collection_context=col_ctx, action_format=_ACTION_FORMAT,
        ))
        return result, f"# Dream Log — {ts} — single:{model_key}\n\n{result}"

    def _run_hier(self, ts: str, memories: str, col_ctx: str) -> tuple[str, str]:
        a = _call("haiku", _P_HAIKU.format(memories=memories, collection_context=col_ctx))
        b = _call("sonnet", _P_SONNET_HIER.format(
            memories=memories, collection_context=col_ctx, action_format=_ACTION_FORMAT, a=a,
        ))
        report = (
            f"# Dream Log — {ts} — hier\n\n"
            f"## Haiku: First Pass\n\n{a}\n\n---\n\n"
            f"## Sonnet: Synthesis\n\n{b}"
        )
        return b, report

    def _run_team(self, ts: str, memories: str, col_ctx: str) -> tuple[str, str]:
        a = _call("haiku", _P_HAIKU.format(memories=memories, collection_context=col_ctx))
        b = _call("sonnet", _P_SONNET_CRITIQUE.format(
            memories=memories, collection_context=col_ctx, a=a,
        ))
        c = _call("haiku", _P_HAIKU_REBUTTAL.format(a=a, b=b))
        d = _call("sonnet", _P_SONNET_FINAL.format(
            collection_context=col_ctx, action_format=_ACTION_FORMAT, a=a, b=b, c=c,
        ))
        report = (
            f"# Dream Log — {ts} — team\n\n"
            f"## Haiku: Initial Analysis\n\n{a}\n\n---\n\n"
            f"## Sonnet: Critique\n\n{b}\n\n---\n\n"
            f"## Haiku: Rebuttal\n\n{c}\n\n---\n\n"
            f"## Sonnet: Final Synthesis\n\n{d}"
        )
        return d, report

    def _auto_apply_deletes(self, synthesis: str, all_entries: list[MemoryEntry]) -> list[str]:
        """Parse DELETE <id>: lines from synthesis and delete matching entries."""
        id_to_col = {e.id: e.metadata["_collection"] for e in all_entries}
        found_ids = _DELETE_RE.findall(synthesis)
        deleted = []
        for id_ in found_ids:
            col = id_to_col.get(id_)
            if col is None:
                continue
            try:
                self._repo.delete(col, [id_])
                deleted.append(id_)
            except Exception:
                pass
        return deleted

    def _auto_apply_links(self, synthesis: str, all_entries: list[MemoryEntry]) -> list[str]:
        """Parse LINK proposals from synthesis and apply them as edges."""
        from ai_mem.application.add_edge import AddEdgeUseCase

        id_to_col = {e.id: e.metadata["_collection"] for e in all_entries}
        applied = []
        for source_id, target_id, edge_type in _parse_link_proposals(synthesis):
            if edge_type not in _VALID_EDGE_TYPES:
                continue
            col = id_to_col.get(source_id)
            if col is None:
                continue
            try:
                AddEdgeUseCase(self._repo).execute(col, source_id, target_id, edge_type)  # type: ignore[arg-type]
                applied.append(f"{source_id} -> {target_id} [{edge_type}]")
            except Exception:
                pass
        return applied
