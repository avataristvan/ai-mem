"""DreamMemoryUseCase — consolidate memories via Claude models."""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime

from ai_mem.domain.memory import MemoryRepository

_DELETE_RE = re.compile(r"^\s*[-*•]\s+DELETE\s+(\S+)\s*:", re.MULTILINE | re.IGNORECASE)
_ADD_TARGET_RE = re.compile(
    r"^\s*[-*•]\s+ADD\s+(\S+)\s+\[target=([^\]]+)\]\s*:",
    re.MULTILINE | re.IGNORECASE,
)

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
Verify the relationship before proposing structural changes."""

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


def _format_entries(entries) -> str:
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
            except Exception:
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
        auto_apply: bool = False,
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
        preamble = "\n\n".join(preamble_parts)

        raw_memories = _format_entries(all_entries)
        memories = preamble + "\n\n---\n\n" + raw_memories
        col_ctx = _collection_context(collections)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")

        if mode == "single-haiku":
            result = _call("haiku", _P_SINGLE.format(
                memories=memories, collection_context=col_ctx, action_format=_ACTION_FORMAT,
            ))
            synthesis = result
            report = f"# Dream Log — {ts} — single:haiku\n\n{result}"

        elif mode == "single-sonnet":
            result = _call("sonnet", _P_SINGLE.format(
                memories=memories, collection_context=col_ctx, action_format=_ACTION_FORMAT,
            ))
            synthesis = result
            report = f"# Dream Log — {ts} — single:sonnet\n\n{result}"

        elif mode == "hier":
            a = _call("haiku", _P_HAIKU.format(memories=memories, collection_context=col_ctx))
            b = _call("sonnet", _P_SONNET_HIER.format(
                memories=memories, collection_context=col_ctx, action_format=_ACTION_FORMAT, a=a,
            ))
            synthesis = b
            report = (
                f"# Dream Log — {ts} — hier\n\n"
                f"## Haiku: First Pass\n\n{a}\n\n---\n\n"
                f"## Sonnet: Synthesis\n\n{b}"
            )

        else:  # team
            a = _call("haiku", _P_HAIKU.format(memories=memories, collection_context=col_ctx))
            b = _call("sonnet", _P_SONNET_CRITIQUE.format(
                memories=memories, collection_context=col_ctx, a=a,
            ))
            c = _call("haiku", _P_HAIKU_REBUTTAL.format(a=a, b=b))
            d = _call("sonnet", _P_SONNET_FINAL.format(
                collection_context=col_ctx, action_format=_ACTION_FORMAT, a=a, b=b, c=c,
            ))
            synthesis = d
            report = (
                f"# Dream Log — {ts} — team\n\n"
                f"## Haiku: Initial Analysis\n\n{a}\n\n---\n\n"
                f"## Sonnet: Critique\n\n{b}\n\n---\n\n"
                f"## Haiku: Rebuttal\n\n{c}\n\n---\n\n"
                f"## Sonnet: Final Synthesis\n\n{d}"
            )

        propagation = _propagation_candidates(synthesis, set(collections))
        if propagation:
            lines = "\n".join(
                f"- `{eid}` → **{target}**" for eid, target in propagation
            )
            report += (
                "\n\n---\n\n## Propagation Candidates\n\n"
                "These ADD proposals target a higher-level collection "
                "(review and apply with `mem_add` if confirmed):\n\n" + lines
            )

        if auto_apply:
            deleted = self._auto_apply_deletes(synthesis, all_entries)
            if deleted:
                report += "\n\n---\n\n## Auto-Applied Deletions\n\n" + "\n".join(
                    f"- Deleted `{d}`" for d in deleted
                )
            else:
                report += "\n\n---\n\n## Auto-Applied Deletions\n\nNone (no high-confidence DELETE actions found)."

        return report

    def _auto_apply_deletes(self, synthesis: str, all_entries) -> list[str]:
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
