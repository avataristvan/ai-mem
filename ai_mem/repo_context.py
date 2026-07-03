from __future__ import annotations
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

WORKSPACE_COLLECTION = "workspace"
GLOBAL_COLLECTION = "global"


@dataclass
class RepoContext:
    scope_name: str | None
    collection: str
    has_claude_md: bool
    claude_md_dir: Path | None
    git_root: Path | None


def _run(cmd: list[str], cwd: Path) -> str | None:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd))
        return proc.stdout.strip() if proc.returncode == 0 else None
    except Exception:
        return None


def _sanitize(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9._-]', '_', name).strip('._-')


def _get_workspace_root() -> Path | None:
    env_root = os.environ.get("AI_MEM_WORKSPACE_ROOT")
    if env_root:
        p = Path(env_root).expanduser().resolve()
        return p if p.is_dir() else None
    return None


def _find_claude_md(start: Path) -> Path | None:
    for directory in [start, *start.parents]:
        if (directory / "CLAUDE.md").exists() or (directory / "agent.md").exists():
            return directory
    return None


def detect_repo_context(cwd: Path | None = None) -> RepoContext:
    try:
        cwd = cwd or Path.cwd()
        claude_md_dir = _find_claude_md(cwd)
        if claude_md_dir is None:
            return RepoContext(None, WORKSPACE_COLLECTION, False, None, None)

        workspace_root = _get_workspace_root()
        if workspace_root:
            try:
                rel = claude_md_dir.resolve().relative_to(workspace_root)
                rel_parts = [_sanitize(p) for p in rel.parts if p not in (".", "")]
                if rel_parts:
                    scope_name = ".".join(rel_parts)
                    return RepoContext(scope_name, f"repo.{scope_name}", True, claude_md_dir, None)
                else:
                    # CLAUDE.md is at the workspace root itself → workspace
                    return RepoContext(None, WORKSPACE_COLLECTION, True, claude_md_dir, None)
            except ValueError:
                pass  # claude_md_dir outside workspace_root → fall through to git

        git_root_str = _run(["git", "rev-parse", "--show-toplevel"], cwd=claude_md_dir)
        if git_root_str is None:
            return RepoContext(None, WORKSPACE_COLLECTION, True, claude_md_dir, None)

        git_root = Path(git_root_str)

        remote = _run(["git", "remote", "get-url", "origin"], cwd=git_root)
        if remote:
            match = re.search(r'[:/]([^/]+?)(?:\.git)?$', remote)
            repo_name = match.group(1) if match else git_root.name
        else:
            repo_name = git_root.name

        try:
            rel = claude_md_dir.relative_to(git_root)
            rel_parts = [part for part in rel.parts if part != "."]
        except ValueError:
            rel_parts = []

        parts = [_sanitize(repo_name)] + [_sanitize(part) for part in rel_parts]
        scope_name = ".".join(part for part in parts if part)
        collection = f"repo.{scope_name}" if scope_name else WORKSPACE_COLLECTION

        return RepoContext(scope_name, collection, True, claude_md_dir, git_root)
    except Exception:
        return RepoContext(None, WORKSPACE_COLLECTION, False, None, None)
