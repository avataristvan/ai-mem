"""Tests for repo_context.detect_repo_context — workspace-root and git-based paths."""
from __future__ import annotations

from pathlib import Path

import pytest

from ai_mem.repo_context import (
    WORKSPACE_COLLECTION,
    detect_repo_context,
    _get_workspace_root,
    _sanitize,
)


# ---------------------------------------------------------------------------
# _sanitize
# ---------------------------------------------------------------------------

class TestSanitize:
    def test_replaces_special_chars(self):
        assert _sanitize("foo/bar baz") == "foo_bar_baz"

    def test_strips_leading_trailing(self):
        assert _sanitize(".-foo-.") == "foo"

    def test_passthrough_valid(self):
        assert _sanitize("ExoDeck-project") == "ExoDeck-project"


# ---------------------------------------------------------------------------
# _get_workspace_root
# ---------------------------------------------------------------------------

class TestGetWorkspaceRoot:
    def test_returns_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("AI_MEM_WORKSPACE_ROOT", raising=False)
        assert _get_workspace_root() is None

    def test_returns_path_when_set(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AI_MEM_WORKSPACE_ROOT", str(tmp_path))
        assert _get_workspace_root() == tmp_path.resolve()

    def test_returns_none_for_nonexistent_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AI_MEM_WORKSPACE_ROOT", str(tmp_path / "nonexistent"))
        assert _get_workspace_root() is None

    def test_expands_tilde(self, monkeypatch):
        monkeypatch.setenv("AI_MEM_WORKSPACE_ROOT", "~")
        result = _get_workspace_root()
        assert result == Path.home().resolve()


# ---------------------------------------------------------------------------
# detect_repo_context — workspace-root mode
# ---------------------------------------------------------------------------

class TestDetectWorkspaceRoot:
    def _make_claude_md(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "CLAUDE.md").write_text("# test")
        return directory

    def test_single_level(self, monkeypatch, tmp_path):
        workspace = tmp_path / "workspace"
        project = workspace / "ai-mem"
        self._make_claude_md(project)
        monkeypatch.setenv("AI_MEM_WORKSPACE_ROOT", str(workspace))

        ctx = detect_repo_context(project / "src")
        assert ctx.collection == "repo.ai-mem"
        assert ctx.scope_name == "ai-mem"
        assert ctx.has_claude_md is True
        assert ctx.git_root is None  # workspace mode does not set git_root

    def test_two_levels(self, monkeypatch, tmp_path):
        workspace = tmp_path / "workspace"
        sub = workspace / "ExoDeck-project" / "filming"
        self._make_claude_md(sub)
        monkeypatch.setenv("AI_MEM_WORKSPACE_ROOT", str(workspace))

        ctx = detect_repo_context(sub / "rushes")
        assert ctx.collection == "repo.ExoDeck-project.filming"
        assert ctx.scope_name == "ExoDeck-project.filming"

    def test_claude_md_at_workspace_root_falls_back_to_workspace(self, monkeypatch, tmp_path):
        workspace = tmp_path / "workspace"
        self._make_claude_md(workspace)
        monkeypatch.setenv("AI_MEM_WORKSPACE_ROOT", str(workspace))

        ctx = detect_repo_context(workspace)
        assert ctx.collection == WORKSPACE_COLLECTION
        assert ctx.scope_name is None

    def test_sanitizes_folder_names(self, monkeypatch, tmp_path):
        workspace = tmp_path / "workspace"
        sub = workspace / "my project" / "sub dir"
        self._make_claude_md(sub)
        monkeypatch.setenv("AI_MEM_WORKSPACE_ROOT", str(workspace))

        ctx = detect_repo_context(sub)
        assert ctx.collection == "repo.my_project.sub_dir"

    def test_no_claude_md_returns_workspace(self, monkeypatch, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        project = workspace / "empty-project"
        project.mkdir()
        monkeypatch.setenv("AI_MEM_WORKSPACE_ROOT", str(workspace))

        ctx = detect_repo_context(project)
        assert ctx.collection == WORKSPACE_COLLECTION
        assert ctx.has_claude_md is False

    def test_outside_workspace_falls_through_to_git(self, monkeypatch, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        outside = tmp_path / "other-project"
        outside.mkdir()
        (outside / "CLAUDE.md").write_text("# test")
        monkeypatch.setenv("AI_MEM_WORKSPACE_ROOT", str(workspace))

        ctx = detect_repo_context(outside)
        # No git root either → scoped by its own directory name, not the
        # shared workspace collection (avoids polluting/being polluted by
        # unrelated projects that also lack a repo-scoped collection).
        assert ctx.collection == "repo.other-project"
        assert ctx.scope_name == "other-project"
        assert ctx.has_claude_md is True


# ---------------------------------------------------------------------------
# detect_repo_context — git fallback (no workspace root)
# ---------------------------------------------------------------------------

class TestDetectGitFallback:
    def test_no_claude_md_no_git(self, monkeypatch, tmp_path):
        monkeypatch.delenv("AI_MEM_WORKSPACE_ROOT", raising=False)
        empty = tmp_path / "empty"
        empty.mkdir()

        ctx = detect_repo_context(empty)
        assert ctx.collection == WORKSPACE_COLLECTION
        assert ctx.has_claude_md is False
