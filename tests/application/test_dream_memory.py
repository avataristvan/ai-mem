"""Tests for dream_memory propagation candidate parsing."""
from __future__ import annotations

import pytest

from ai_mem.application.dream_memory import _ADD_TARGET_RE, _propagation_candidates


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
