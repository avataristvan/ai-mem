from __future__ import annotations
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
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd))
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _sanitize(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9._-]', '_', name).strip('._-')


def _find_claude_md(start: Path) -> Path | None:
    for d in [start, *start.parents]:
        if (d / "CLAUDE.md").exists() or (d / "agent.md").exists():
            return d
    return None


def detect_repo_context(cwd: Path | None = None) -> RepoContext:
    try:
        cwd = cwd or Path.cwd()
        claude_md_dir = _find_claude_md(cwd)
        if claude_md_dir is None:
            return RepoContext(None, WORKSPACE_COLLECTION, False, None, None)

        git_root_str = _run(["git", "rev-parse", "--show-toplevel"], cwd=claude_md_dir)
        if git_root_str is None:
            return RepoContext(None, WORKSPACE_COLLECTION, True, claude_md_dir, None)

        git_root = Path(git_root_str)

        remote = _run(["git", "remote", "get-url", "origin"], cwd=git_root)
        if remote:
            m = re.search(r'[:/]([^/]+?)(?:\.git)?$', remote)
            repo_name = m.group(1) if m else git_root.name
        else:
            repo_name = git_root.name

        try:
            rel = claude_md_dir.relative_to(git_root)
            rel_parts = [p for p in rel.parts if p != '.']
        except ValueError:
            rel_parts = []

        parts = [_sanitize(repo_name)] + [_sanitize(p) for p in rel_parts]
        scope_name = ".".join(p for p in parts if p)
        collection = f"repo.{scope_name}" if scope_name else WORKSPACE_COLLECTION

        return RepoContext(scope_name, collection, True, claude_md_dir, git_root)
    except Exception:
        return RepoContext(None, WORKSPACE_COLLECTION, False, None, None)
