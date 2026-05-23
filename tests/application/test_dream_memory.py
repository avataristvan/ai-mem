"""Tests for dream_memory propagation candidate parsing and entry formatting."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ai_mem.application.dream_memory import (
    _ADD_TARGET_RE,
    _TYPE_RULES,
    _build_edge_index,
    _check_merge_conflicts,
    _format_entries,
    _parse_link_proposals,
    _propagation_candidates,
)
from ai_mem.domain.memory import MemoryEntry


def _entry(id: str, text: str, **meta) -> MemoryEntry:
    meta["_collection"] = meta.pop("collection", "repo.test")
    return MemoryEntry(id=id, text=text, metadata=meta)


# ── _format_entries ───────────────────────────────────────────────────────────

class TestFormatEntries:
    def test_shows_type_metadata(self):
        e = _entry("p1", "Rule: X\nWhen: Y\nWhy: Z", type="pattern")
        out = _format_entries([e])
        assert "type=pattern" in out

    def test_shows_access_count(self):
        e = _entry("f1", "some feedback", access_count=7)
        out = _format_entries([e])
        assert "access_count=7" in out

    def test_decodes_edges_to_readable_form(self):
        edges = json.dumps([{"target_id": "p2", "edge_type": "contradicts"}])
        e = _entry("ap1", "anti-pattern text", type="anti-pattern", edges=edges)
        out = _format_entries([e])
        assert "p2(contradicts)" in out

    def test_excludes_raw_timestamps(self):
        e = _entry("e1", "text", created_at=1234567890.0, last_accessed_at=9999.0)
        out = _format_entries([e])
        assert "created_at" not in out
        assert "last_accessed_at" not in out

    def test_empty_edges_field_not_shown(self):
        e = _entry("e2", "text", edges=json.dumps([]))
        out = _format_entries([e])
        assert "edges" not in out

    def test_malformed_edges_silently_skipped(self):
        e = _entry("e3", "text", edges="not-json")
        out = _format_entries([e])
        assert "edges" not in out

    def test_no_metadata_no_brackets(self):
        e = _entry("e4", "plain text")
        out = _format_entries([e])
        assert "[]" not in out
        assert "[e4]" in out

    def test_empty_list_returns_empty_marker(self):
        assert _format_entries([]) == "(empty)"


# ── _TYPE_RULES preamble presence ─────────────────────────────────────────────

class TestTypeRules:
    def test_type_rules_mentions_pattern_format(self):
        assert "Rule:" in _TYPE_RULES
        assert "When:" in _TYPE_RULES
        assert "Why:" in _TYPE_RULES

    def test_type_rules_mentions_anti_pattern_format(self):
        assert "Tried:" in _TYPE_RULES
        assert "Failed because:" in _TYPE_RULES

    def test_type_rules_warns_about_merge_constraint(self):
        assert "MERGE" in _TYPE_RULES
        assert "anti-pattern" in _TYPE_RULES


# ── _ADD_TARGET_RE ────────────────────────────────────────────────────────────

class TestAddTargetRe:
    def test_parses_dash_bullet(self):
        text = "- ADD my_entry [target=global]: some content"
        m = _ADD_TARGET_RE.search(text)
        assert m is not None
        assert m.group(1) == "my_entry"
        assert m.group(2).strip() == "global"

    def test_parses_star_bullet(self):
        text = "* ADD my_entry [target=workspace]: some content"
        m = _ADD_TARGET_RE.search(text)
        assert m is not None
        assert m.group(2).strip() == "workspace"

    def test_parses_dot_bullet(self):
        text = "• ADD my_entry [target=repo.my-app]: some content"
        m = _ADD_TARGET_RE.search(text)
        assert m is not None
        assert m.group(2).strip() == "repo.my-app"

    def test_case_insensitive(self):
        text = "- add my_entry [target=global]: some content"
        assert _ADD_TARGET_RE.search(text) is not None

    def test_no_match_without_target_field(self):
        text = "- ADD my_entry: some content without target"
        assert _ADD_TARGET_RE.search(text) is None

    def test_no_match_on_delete_line(self):
        text = "- DELETE old_entry: reason"
        assert _ADD_TARGET_RE.search(text) is None

    def test_multiline_finds_all(self):
        text = (
            "- ADD entry_a [target=global]: first\n"
            "- ADD entry_b [target=workspace]: second\n"
            "- ADD entry_c: no target\n"
        )
        matches = _ADD_TARGET_RE.findall(text)
        assert len(matches) == 2
        assert ("entry_a", "global") in matches
        assert ("entry_b", "workspace") in matches


# ── _propagation_candidates ───────────────────────────────────────────────────

class TestPropagationCandidates:
    def test_returns_candidates_with_foreign_target(self):
        synthesis = "- ADD new_pattern [target=global]: a broadly useful pattern"
        result = _propagation_candidates(synthesis, {"repo.my-app"})
        assert result == [("new_pattern", "global")]

    def test_excludes_candidates_in_source_collections(self):
        synthesis = "- ADD same_col_entry [target=repo.my-app]: stays local"
        result = _propagation_candidates(synthesis, {"repo.my-app"})
        assert result == []

    def test_empty_synthesis_returns_empty(self):
        assert _propagation_candidates("", {"repo.ai-mem"}) == []

    def test_no_add_with_target_returns_empty(self):
        synthesis = "- DELETE old_entry: stale\n- UPDATE some_id: improve wording"
        assert _propagation_candidates(synthesis, {"repo.ai-mem"}) == []

    def test_mixed_targets_only_returns_foreign(self):
        synthesis = (
            "- ADD local_tip [target=repo.my-app]: project-specific\n"
            "- ADD global_pattern [target=global]: universal\n"
            "- ADD workspace_tip [target=workspace]: cross-project\n"
        )
        result = _propagation_candidates(synthesis, {"repo.my-app"})
        assert ("local_tip", "global") not in result
        assert ("global_pattern", "global") in result
        assert ("workspace_tip", "workspace") in result
        assert len(result) == 2

    def test_multiple_source_collections(self):
        synthesis = (
            "- ADD entry_a [target=repo.my-app]: already a source\n"
            "- ADD entry_b [target=global]: propagate this\n"
        )
        result = _propagation_candidates(synthesis, {"repo.my-app", "repo.ai-mem"})
        assert result == [("entry_b", "global")]

    def test_whitespace_in_target_is_stripped(self):
        synthesis = "- ADD entry_x [target=  global  ]: content"
        result = _propagation_candidates(synthesis, {"repo.ai-mem"})
        assert result == [("entry_x", "global")]


# ── _build_edge_index ─────────────────────────────────────────────────────────

class TestBuildEdgeIndex:
    def _entry(self, id: str, edges: list[dict]) -> MemoryEntry:
        raw = json.dumps(edges)
        return MemoryEntry(id=id, text="t", metadata={"edges": raw, "_collection": "col"})

    def test_extracts_edges_from_metadata(self):
        e = self._entry("a", [{"target_id": "b", "edge_type": "contradicts"}])
        index = _build_edge_index([e])
        assert index == {("a", "b"): "contradicts"}

    def test_multiple_edges_on_one_entry(self):
        e = self._entry("a", [
            {"target_id": "b", "edge_type": "contradicts"},
            {"target_id": "c", "edge_type": "fixes"},
        ])
        index = _build_edge_index([e])
        assert index[("a", "b")] == "contradicts"
        assert index[("a", "c")] == "fixes"

    def test_empty_edges_returns_empty_index(self):
        e = self._entry("a", [])
        assert _build_edge_index([e]) == {}

    def test_missing_edges_key_returns_empty_index(self):
        e = MemoryEntry(id="a", text="t", metadata={"_collection": "col"})
        assert _build_edge_index([e]) == {}

    def test_malformed_json_silently_skipped(self):
        e = MemoryEntry(id="a", text="t", metadata={"edges": "not-json", "_collection": "col"})
        assert _build_edge_index([e]) == {}

    def test_multiple_entries_merged(self):
        a = self._entry("a", [{"target_id": "b", "edge_type": "causes"}])
        b = self._entry("b", [{"target_id": "c", "edge_type": "related"}])
        index = _build_edge_index([a, b])
        assert index == {("a", "b"): "causes", ("b", "c"): "related"}


# ── _check_merge_conflicts ────────────────────────────────────────────────────

class TestCheckMergeConflicts:
    def test_flags_contradicts_edge(self):
        index = {("a", "b"): "contradicts"}
        warnings = _check_merge_conflicts("- MERGE a + b: combine", index)
        assert len(warnings) == 1
        assert "contradicts" in warnings[0]
        assert "a" in warnings[0] and "b" in warnings[0]

    def test_bidirectional_detection(self):
        # Edge stored on b (b → a), but MERGE proposes a + b
        index = {("b", "a"): "contradicts"}
        warnings = _check_merge_conflicts("- MERGE a + b: combine", index)
        assert len(warnings) == 1

    def test_related_edge_does_not_block_merge(self):
        index = {("a", "b"): "related"}
        assert _check_merge_conflicts("- MERGE a + b: combine", index) == []

    def test_no_merge_in_text_returns_empty(self):
        index = {("a", "b"): "contradicts"}
        assert _check_merge_conflicts("- DELETE a: stale", index) == []

    def test_no_edge_returns_empty(self):
        assert _check_merge_conflicts("- MERGE a + b: combine", {}) == []

    def test_multiple_merges_only_flags_conflicting(self):
        index = {("a", "b"): "contradicts"}
        synthesis = "- MERGE a + b: combine\n- MERGE c + d: also combine"
        warnings = _check_merge_conflicts(synthesis, index)
        assert len(warnings) == 1
        assert "a" in warnings[0] and "b" in warnings[0]


# ── _parse_link_proposals ─────────────────────────────────────────────────────

class TestParseLinkProposals:
    def test_parses_dash_bullet(self):
        result = _parse_link_proposals("- LINK src -> tgt [type=contradicts]: reason")
        assert result == [("src", "tgt", "contradicts")]

    def test_parses_star_bullet(self):
        result = _parse_link_proposals("* LINK src -> tgt [type=fixes]: reason")
        assert result == [("src", "tgt", "fixes")]

    def test_parses_dot_bullet(self):
        result = _parse_link_proposals("• LINK src -> tgt [type=related]: reason")
        assert result == [("src", "tgt", "related")]

    def test_case_insensitive(self):
        result = _parse_link_proposals("- link src -> tgt [type=causes]: reason")
        assert result == [("src", "tgt", "causes")]

    def test_whitespace_in_type_stripped(self):
        result = _parse_link_proposals("- LINK a -> b [type=  related  ]: reason")
        assert result == [("a", "b", "related")]

    def test_multiple_links(self):
        synthesis = (
            "- LINK a -> b [type=contradicts]: opposed\n"
            "- LINK c -> d [type=fixes]: correction\n"
        )
        result = _parse_link_proposals(synthesis)
        assert ("a", "b", "contradicts") in result
        assert ("c", "d", "fixes") in result
        assert len(result) == 2

    def test_no_type_bracket_not_matched(self):
        result = _parse_link_proposals("- LINK a -> b: no type here")
        assert result == []

    def test_empty_synthesis(self):
        assert _parse_link_proposals("") == []


# ── Integration: execute() with mocked _call ─────────────────────────────────

def _seed(repo, collection: str, docs: dict[str, str], metadatas: list[dict] | None = None) -> None:
    from ai_mem.application.add_memory import AddMemoryUseCase
    metas = metadatas or [{}] * len(docs)
    AddMemoryUseCase(repo).execute(
        collection=collection,
        documents=list(docs.values()),
        ids=list(docs.keys()),
        metadatas=metas,
    )


class TestDreamExecuteMergeConflict:
    def test_merge_conflict_flagged_in_report(self, tmp_repo, tmp_path: Path):
        from ai_mem.application.add_edge import AddEdgeUseCase
        from ai_mem.application.dream_memory import DreamMemoryUseCase

        _seed(tmp_repo, "col",
              {"p1": "Rule: always validate\nWhen: input received\nWhy: safety",
               "ap1": "Tried: skip validation\nFailed because: injection\nInstead: validate"},
              [{"type": "pattern"}, {"type": "anti-pattern"}])
        AddEdgeUseCase(tmp_repo).execute("col", "ap1", "p1", "contradicts")

        dream_uc = DreamMemoryUseCase(tmp_repo)
        synthesis = "- MERGE ap1 + p1: combine into unified guidance"

        with patch("ai_mem.application.dream_memory._call", return_value=synthesis):
            report = dream_uc.execute("col", mode="single-haiku")

        assert "## ⚠ Merge Conflicts" in report
        assert "contradicts" in report

    def test_merge_without_contradicts_edge_not_flagged(self, tmp_repo, tmp_path: Path):
        from ai_mem.application.dream_memory import DreamMemoryUseCase

        _seed(tmp_repo, "col", {"e1": "entry one text", "e2": "entry two text"})

        dream_uc = DreamMemoryUseCase(tmp_repo)
        synthesis = "- MERGE e1 + e2: duplicate content"

        with patch("ai_mem.application.dream_memory._call", return_value=synthesis):
            report = dream_uc.execute("col", mode="single-haiku")

        assert "## ⚠ Merge Conflicts" not in report

    def test_bidirectional_edge_also_flagged(self, tmp_repo, tmp_path: Path):
        from ai_mem.application.add_edge import AddEdgeUseCase
        from ai_mem.application.dream_memory import DreamMemoryUseCase

        _seed(tmp_repo, "col", {"p1": "pattern text", "ap1": "anti-pattern text"})
        # Edge stored on p1 pointing to ap1 (reverse of MERGE order)
        AddEdgeUseCase(tmp_repo).execute("col", "p1", "ap1", "contradicts")

        dream_uc = DreamMemoryUseCase(tmp_repo)
        synthesis = "- MERGE ap1 + p1: combine"

        with patch("ai_mem.application.dream_memory._call", return_value=synthesis):
            report = dream_uc.execute("col", mode="single-haiku")

        assert "## ⚠ Merge Conflicts" in report


class TestDreamExecuteAutoLink:
    def test_auto_link_creates_edge(self, tmp_repo, tmp_path: Path):
        from ai_mem.application.dream_memory import DreamMemoryUseCase
        from ai_mem.application.get_edges import GetEdgesUseCase

        _seed(tmp_repo, "col", {"e1": "pattern text here", "e2": "anti-pattern text here"})

        dream_uc = DreamMemoryUseCase(tmp_repo)
        synthesis = "- LINK e1 -> e2 [type=related]: they are connected"

        with patch("ai_mem.application.dream_memory._call", return_value=synthesis):
            report = dream_uc.execute("col", mode="single-haiku", auto_link=True)

        assert "Auto-Applied Links" in report
        edges = GetEdgesUseCase(tmp_repo).execute("col", "e1")
        assert any(e.target_id == "e2" and e.edge_type == "related" for e in edges)

    def test_auto_link_not_applied_when_flag_false(self, tmp_repo, tmp_path: Path):
        from ai_mem.application.dream_memory import DreamMemoryUseCase
        from ai_mem.application.get_edges import GetEdgesUseCase

        _seed(tmp_repo, "col", {"e1": "text", "e2": "text"})

        dream_uc = DreamMemoryUseCase(tmp_repo)
        synthesis = "- LINK e1 -> e2 [type=related]: connected"

        with patch("ai_mem.application.dream_memory._call", return_value=synthesis):
            report = dream_uc.execute("col", mode="single-haiku", auto_link=False)

        assert "Auto-Applied Links" not in report
        assert GetEdgesUseCase(tmp_repo).execute("col", "e1") == []

    def test_auto_link_skips_nonexistent_source(self, tmp_repo, tmp_path: Path):
        from ai_mem.application.dream_memory import DreamMemoryUseCase

        _seed(tmp_repo, "col", {"e2": "target text"})

        dream_uc = DreamMemoryUseCase(tmp_repo)
        synthesis = "- LINK ghost -> e2 [type=related]: ghost does not exist"

        with patch("ai_mem.application.dream_memory._call", return_value=synthesis):
            report = dream_uc.execute("col", mode="single-haiku", auto_link=True)

        # Must not crash; no link entry for ghost
        assert "ghost" not in report or "Auto-Applied Links" not in report

    def test_auto_link_skips_invalid_edge_type(self, tmp_repo, tmp_path: Path):
        from ai_mem.application.dream_memory import DreamMemoryUseCase
        from ai_mem.application.get_edges import GetEdgesUseCase

        _seed(tmp_repo, "col", {"e1": "text", "e2": "text"})

        dream_uc = DreamMemoryUseCase(tmp_repo)
        synthesis = "- LINK e1 -> e2 [type=invalid_type]: wrong type"

        with patch("ai_mem.application.dream_memory._call", return_value=synthesis):
            report = dream_uc.execute("col", mode="single-haiku", auto_link=True)

        assert GetEdgesUseCase(tmp_repo).execute("col", "e1") == []
