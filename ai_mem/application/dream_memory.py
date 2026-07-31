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
_SPLIT_RE = re.compile(r"^\s*[-*•]\s+SPLIT\s+(\S+)\s*:", re.MULTILINE | re.IGNORECASE)

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
- Entries covering multiple independent concepts (multiple ## sections, multiple Tried: blocks, \
or 4+ enumerated items that don't build on each other) should be proposed for SPLIT, not MERGE. \
Each sub-entry should cover exactly one concept so queries pull in only what's relevant."""

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
- SPLIT <id>: <list the sub-concept titles to split into>

For ADD, set target= to the collection where the entry belongs:
  - Same collection as the source entry if it is project-specific.
  - A higher-level collection (workspace or global) if the pattern applies broadly \
across multiple projects — this flags it as a propagation candidate."""

_P_SINGLE = """\
You are a memory consolidation agent. The memories below come from AI assistant sessions \
(stored as text entries with metadata). Analyze them and return a structured, actionable proposal:

1. **Contradictions** — entries that conflict with each other (name entry IDs)
2. **Redundancies** — entries that overlap and could be merged
3. **Stale entries** — entries likely outdated (explain why)
4. **Missing principles** — patterns that emerge across entries but aren't yet documented
5. **Over-broad entries** — entries covering multiple independent concepts that should be split so queries pull in only the relevant piece

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
{haiku_pass}"""

_P_SONNET_CRITIQUE = """\
You are the second voice in a memory consolidation debate. A faster model gave an initial analysis.
Challenge it: what did it miss? Where is it wrong? What subtle patterns does it overlook?
Also add your own findings. Be direct.

{collection_context}

MEMORIES:
{memories}

INITIAL ANALYSIS:
{haiku_analysis}"""

_P_HAIKU_REBUTTAL = """\
You gave an initial memory analysis. A more capable model critiqued it. Respond:
- Defend what you got right
- Concede where the critique is valid
- Add anything this exchange surfaced

YOUR INITIAL ANALYSIS:
{haiku_analysis}

CRITIQUE:
{sonnet_critique}"""

_P_SONNET_FINAL = """\
Synthesize the best insights from this full debate into one clean, actionable proposal.

{action_format}

{collection_context}

INITIAL ANALYSIS:
{haiku_analysis}

CRITIQUE:
{sonnet_critique}

REBUTTAL:
{haiku_rebuttal}"""


def _format_entries(entries: list[MemoryEntry]) -> str:
    parts = []
    for entry in entries:
        col = entry.metadata.get("_collection", "?")
        shown: dict[str, object] = {}
        if entry_type := entry.metadata.get("type"):
            shown["type"] = entry_type
        if access_count := entry.metadata.get("access_count"):
            shown["access_count"] = int(access_count)
        if raw_edges := entry.metadata.get("edges"):
            try:
                edges = json.loads(raw_edges)
                if edges:
                    shown["edges"] = [f"{ed['target_id']}({ed['edge_type']})" for ed in edges]
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
        meta_str = (
            " [" + ", ".join(f"{key}={val}" for key, val in shown.items()) + "]"
            if shown else ""
        )
        parts.append(f"[{entry.id}] (collection: {col}){meta_str}\n{entry.text}")
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
    return _COLLECTION_CONTEXT.format(collections="\n".join(f"  - {col_name}" for col_name in collections))


def _confidence_report(entries: list[MemoryEntry]) -> str:
    """Return a markdown report of decay and promotion candidates based on confidence metadata.

    Entries without a 'confidence' field are silently skipped.
    Returns an empty string when no candidates exist in either category.
    """
    decay: list[tuple[str, str, float, int]] = []    # (id, collection, confidence, access_count)
    promote: list[tuple[str, str, float, int, str]] = []  # (id, collection, confidence, access_count, type)

    for entry in entries:
        raw = entry.metadata.get("confidence")
        if raw is None:
            continue
        try:
            conf = float(raw)
        except (ValueError, TypeError):
            continue

        col = entry.metadata.get("_collection", "?")
        access_count = int(entry.metadata.get("access_count", 0))
        entry_type = str(entry.metadata.get("type", ""))

        if conf < 0.3:
            decay.append((entry.id, col, conf, access_count))
        if conf > 0.9 and access_count >= 3 and not entry.metadata.get("keep_in_ai_mem"):
            promote.append((entry.id, col, conf, access_count, entry_type))

    if not decay and not promote:
        return ""

    sections: list[str] = ["---", "", "## Confidence Report"]

    if decay:
        sections += [
            "",
            "### Decay Candidates",
            "",
            "Consider DELETE or mem-dream review:",
            "",
        ]
        for eid, col, conf, access_count in decay:
            sections.append(f"- `{eid}` ({col}) — confidence={conf:.2f}, access_count={access_count}")

    if promote:
        sections += [
            "",
            "### Promotion Candidates",
            "",
            "Consider promoting to CLAUDE.md (stable, frequently accessed):",
            "",
        ]
        for eid, col, conf, access_count, etype in promote:
            type_tag = f", type={etype}" if etype else ""
            sections.append(f"- `{eid}` ({col}) — confidence={conf:.2f}, access_count={access_count}{type_tag}")

    return "\n" + "\n".join(sections)


def _split_candidates(synthesis: str, all_entries: list[MemoryEntry]) -> str:
    """Return a formatted section listing SPLIT proposals from the synthesis, or empty string."""
    id_to_preview: dict[str, str] = {entry.id: entry.text[:60].replace("\n", " ") for entry in all_entries}
    found = _SPLIT_RE.findall(synthesis)
    if not found:
        return ""
    lines = []
    for eid in found:
        preview = id_to_preview.get(eid, "")
        preview_str = f" — {preview}…" if preview else ""
        lines.append(f"- `{eid}`{preview_str}")
    return (
        "\n\n---\n\n## Split Candidates\n\n"
        "These entries cover multiple independent concepts — consider splitting with `mem_split`:\n\n"
        + "\n".join(lines)
    )


def _propagation_candidates(synthesis: str, source_collections: set[str]) -> list[tuple[str, str]]:
    """Return (entry_id, target_collection) pairs where target differs from all source collections."""
    candidates = []
    for match in _ADD_TARGET_RE.finditer(synthesis):
        entry_id, target = match.group(1), match.group(2).strip()
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
        focus_hint: str | None = None,
        collections_filter: list[str] | None = None,
    ) -> str:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")

        if collections_filter is not None:
            collections = collections_filter
        elif collection:
            collections = [collection]
        else:
            collections = [col_info.name for col_info in self._repo.list_collections()]

        all_entries = []
        for col in collections:
            entries = self._repo.get_all(col)
            for entry in entries:
                entry.metadata["_collection"] = col
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

        propagation = _propagation_candidates(synthesis, set(collections))
        if propagation:
            lines = "\n".join(f"- `{eid}` → **{target}**" for eid, target in propagation)
            report += (
                "\n\n---\n\n## Propagation Candidates\n\n"
                "These ADD proposals target a higher-level collection "
                "(review and apply with `mem_add` if confirmed):\n\n" + lines
            )

        split_report = _split_candidates(synthesis, all_entries)
        if split_report:
            report += split_report

        conf_report = _confidence_report(all_entries)
        if conf_report:
            report += "\n\n" + conf_report

        if auto_delete:
            deleted = self._auto_apply_deletes(synthesis, all_entries)
            if deleted:
                report += "\n\n---\n\n## Auto-Applied Deletions\n\n" + "\n".join(
                    f"- Deleted `{deleted_id}`" for deleted_id in deleted
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
        haiku_pass = _call("haiku", _P_HAIKU.format(memories=memories, collection_context=col_ctx))
        sonnet_synthesis = _call("sonnet", _P_SONNET_HIER.format(
            memories=memories, collection_context=col_ctx, action_format=_ACTION_FORMAT, haiku_pass=haiku_pass,
        ))
        report = (
            f"# Dream Log — {ts} — hier\n\n"
            f"## Haiku: First Pass\n\n{haiku_pass}\n\n---\n\n"
            f"## Sonnet: Synthesis\n\n{sonnet_synthesis}"
        )
        return sonnet_synthesis, report

    def _run_team(self, ts: str, memories: str, col_ctx: str) -> tuple[str, str]:
        haiku_analysis = _call("haiku", _P_HAIKU.format(memories=memories, collection_context=col_ctx))
        sonnet_critique = _call("sonnet", _P_SONNET_CRITIQUE.format(
            memories=memories, collection_context=col_ctx, haiku_analysis=haiku_analysis,
        ))
        haiku_rebuttal = _call("haiku", _P_HAIKU_REBUTTAL.format(haiku_analysis=haiku_analysis, sonnet_critique=sonnet_critique))
        sonnet_final = _call("sonnet", _P_SONNET_FINAL.format(
            collection_context=col_ctx, action_format=_ACTION_FORMAT,
            haiku_analysis=haiku_analysis, sonnet_critique=sonnet_critique, haiku_rebuttal=haiku_rebuttal,
        ))
        report = (
            f"# Dream Log — {ts} — team\n\n"
            f"## Haiku: Initial Analysis\n\n{haiku_analysis}\n\n---\n\n"
            f"## Sonnet: Critique\n\n{sonnet_critique}\n\n---\n\n"
            f"## Haiku: Rebuttal\n\n{haiku_rebuttal}\n\n---\n\n"
            f"## Sonnet: Final Synthesis\n\n{sonnet_final}"
        )
        return sonnet_final, report

    def _auto_apply_deletes(self, synthesis: str, all_entries: list[MemoryEntry]) -> list[str]:
        """Parse DELETE <id>: lines from synthesis and delete matching entries."""
        id_to_col = {entry.id: entry.metadata["_collection"] for entry in all_entries}
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
