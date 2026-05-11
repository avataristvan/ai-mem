"""Tests for dream_memory propagation candidate parsing and entry formatting."""
from __future__ import annotations

import json

from ai_mem.application.dream_memory import (
    _ADD_TARGET_RE,
    _TYPE_RULES,
    _format_entries,
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
        text = "• ADD my_entry [target=repo.ExoDeck]: some content"
        m = _ADD_TARGET_RE.search(text)
        assert m is not None
        assert m.group(2).strip() == "repo.ExoDeck"

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
        result = _propagation_candidates(synthesis, {"repo.ExoDeck"})
        assert result == [("new_pattern", "global")]

    def test_excludes_candidates_in_source_collections(self):
        synthesis = "- ADD same_col_entry [target=repo.ExoDeck]: stays local"
        result = _propagation_candidates(synthesis, {"repo.ExoDeck"})
        assert result == []

    def test_empty_synthesis_returns_empty(self):
        assert _propagation_candidates("", {"repo.ai-mem"}) == []

    def test_no_add_with_target_returns_empty(self):
        synthesis = "- DELETE old_entry: stale\n- UPDATE some_id: improve wording"
        assert _propagation_candidates(synthesis, {"repo.ai-mem"}) == []

    def test_mixed_targets_only_returns_foreign(self):
        synthesis = (
            "- ADD local_tip [target=repo.ExoDeck]: project-specific\n"
            "- ADD global_pattern [target=global]: universal\n"
            "- ADD workspace_tip [target=workspace]: cross-project\n"
        )
        result = _propagation_candidates(synthesis, {"repo.ExoDeck"})
        assert ("local_tip", "global") not in result
        assert ("global_pattern", "global") in result
        assert ("workspace_tip", "workspace") in result
        assert len(result) == 2

    def test_multiple_source_collections(self):
        synthesis = (
            "- ADD entry_a [target=repo.ExoDeck]: already a source\n"
            "- ADD entry_b [target=global]: propagate this\n"
        )
        result = _propagation_candidates(synthesis, {"repo.ExoDeck", "repo.ai-mem"})
        assert result == [("entry_b", "global")]

    def test_whitespace_in_target_is_stripped(self):
        synthesis = "- ADD entry_x [target=  global  ]: content"
        result = _propagation_candidates(synthesis, {"repo.ai-mem"})
        assert result == [("entry_x", "global")]
