"""Tests for dream_memory propagation candidate parsing and entry formatting."""
from __future__ import annotations

import json

from ai_mem.application.dream_memory import (
    _ADD_TARGET_RE,
    _TYPE_RULES,
    _confidence_report,
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


# ── _confidence_report ────────────────────────────────────────────────────────

class TestConfidenceReport:
    def test_empty_when_no_entries_have_confidence(self):
        entries = [_entry("e1", "text", access_count=5, type="pattern")]
        assert _confidence_report(entries) == ""

    def test_empty_when_no_candidates_in_either_category(self):
        entries = [_entry("e1", "text", confidence=0.5, access_count=1)]
        assert _confidence_report(entries) == ""

    def test_decay_candidate_below_threshold(self):
        entries = [_entry("low_conf", "stale text", confidence=0.1, access_count=0, collection="repo.x")]
        out = _confidence_report(entries)
        assert "Decay Candidates" in out
        assert "`low_conf`" in out
        assert "confidence=0.10" in out
        assert "access_count=0" in out
        assert "Consider DELETE or mem-dream review" in out

    def test_promotion_candidate_above_threshold_with_enough_access(self):
        entries = [_entry("high_conf", "stable rule", confidence=0.95, access_count=5, type="pattern", collection="repo.x")]
        out = _confidence_report(entries)
        assert "Promotion Candidates" in out
        assert "`high_conf`" in out
        assert "confidence=0.95" in out
        assert "access_count=5" in out
        assert "type=pattern" in out
        assert "Consider promoting to CLAUDE.md" in out

    def test_promotion_requires_access_count_ge_3(self):
        entries = [_entry("almost", "text", confidence=0.95, access_count=2)]
        assert _confidence_report(entries) == ""

    def test_boundary_confidence_exactly_0_3_is_not_decay(self):
        entries = [_entry("boundary", "text", confidence=0.3, access_count=0)]
        assert _confidence_report(entries) == ""

    def test_boundary_confidence_exactly_0_9_is_not_promotion(self):
        entries = [_entry("boundary", "text", confidence=0.9, access_count=5)]
        assert _confidence_report(entries) == ""

    def test_entry_without_confidence_is_skipped(self):
        entries = [
            _entry("no_conf", "text", access_count=10),
            _entry("has_conf", "text", confidence=0.1, access_count=0),
        ]
        out = _confidence_report(entries)
        assert "`no_conf`" not in out
        assert "`has_conf`" in out

    def test_decay_section_absent_when_only_promotion(self):
        entries = [_entry("promo", "text", confidence=0.99, access_count=10)]
        out = _confidence_report(entries)
        assert "Promotion Candidates" in out
        assert "Decay Candidates" not in out

    def test_promotion_section_absent_when_only_decay(self):
        entries = [_entry("decay", "text", confidence=0.05, access_count=0)]
        out = _confidence_report(entries)
        assert "Decay Candidates" in out
        assert "Promotion Candidates" not in out

    def test_both_sections_present_when_mixed(self):
        entries = [
            _entry("decay_one", "text", confidence=0.1, access_count=0),
            _entry("promo_one", "text", confidence=0.95, access_count=4),
        ]
        out = _confidence_report(entries)
        assert "Decay Candidates" in out
        assert "Promotion Candidates" in out

    def test_invalid_confidence_value_is_skipped(self):
        entries = [_entry("bad", "text", confidence="not-a-number", access_count=0)]
        assert _confidence_report(entries) == ""

    def test_promotion_without_type_omits_type_tag(self):
        entries = [_entry("promo", "text", confidence=0.95, access_count=5)]
        out = _confidence_report(entries)
        assert "type=" not in out

    def test_keep_in_ai_mem_suppresses_promotion(self):
        entries = [_entry("promo", "text", confidence=0.95, access_count=5, keep_in_ai_mem=True)]
        assert _confidence_report(entries) == ""

    def test_keep_in_ai_mem_does_not_suppress_decay(self):
        entries = [_entry("decay", "text", confidence=0.1, access_count=0, keep_in_ai_mem=True)]
        out = _confidence_report(entries)
        assert "Decay Candidates" in out
