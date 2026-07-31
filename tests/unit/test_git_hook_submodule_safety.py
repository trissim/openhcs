"""Regression coverage for repository-local Git hook safety."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HOOK_NAMES = ("post-checkout", "post-merge", "report-submodule-status")


def _git(repository: Path, *arguments: str) -> str:
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_AUTHOR_EMAIL": "test@example.invalid",
            "GIT_AUTHOR_NAME": "OpenHCS hook test",
            "GIT_COMMITTER_EMAIL": "test@example.invalid",
            "GIT_COMMITTER_NAME": "OpenHCS hook test",
        }
    )
    completed = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    )
    return completed.stdout.strip()


def test_repository_hooks_never_move_submodule_worktrees(tmp_path: Path) -> None:
    """Checkout/restore and merge hooks only diagnose gitlink differences."""

    submodule_remote = tmp_path / "submodule-remote"
    parent = tmp_path / "parent"
    _git(tmp_path, "init", "--initial-branch=main", str(submodule_remote))
    _git(submodule_remote, "commit", "--allow-empty", "-m", "first")
    recorded_submodule_head = _git(submodule_remote, "rev-parse", "HEAD")

    _git(tmp_path, "init", "--initial-branch=main", str(parent))
    _git(
        parent,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(submodule_remote),
        "external/demo",
    )
    _git(parent, "commit", "--all", "-m", "add submodule")

    hook_directory = parent / "hooks"
    hook_directory.mkdir()
    for hook_name in HOOK_NAMES:
        shutil.copy2(REPOSITORY_ROOT / "hooks" / hook_name, hook_directory)
    _git(parent, "config", "core.hooksPath", "hooks")

    _git(submodule_remote, "commit", "--allow-empty", "-m", "second")
    assert _git(submodule_remote, "rev-parse", "HEAD") != recorded_submodule_head

    marker = parent / "marker"
    marker.write_text("staged only", encoding="utf-8")
    _git(parent, "add", marker.name)
    _git(parent, "restore", "--staged", marker.name)
    assert (
        _git(parent / "external/demo", "rev-parse", "HEAD")
        == recorded_submodule_head
    )

    _git(parent, "checkout", "-b", "fixture-merge")
    _git(parent, "commit", "--allow-empty", "-m", "fixture merge")
    _git(parent, "checkout", "main")
    _git(parent, "merge", "--no-ff", "fixture-merge", "-m", "fixture merge")
    assert (
        _git(parent / "external/demo", "rev-parse", "HEAD")
        == recorded_submodule_head
    )
