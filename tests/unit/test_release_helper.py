from __future__ import annotations

import subprocess
import sys

import pytest

from scripts import release


def _approve_release(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(release, "get_current_version", lambda: "0.5.22")
    monkeypatch.setattr(release, "get_pypi_version", lambda: "0.5.21")
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")


def test_release_preflights_metadata_before_tag_creation_and_push(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _approve_release(monkeypatch)
    calls: list[list[str]] = []

    def record_run(command: list[str], *, check: bool) -> subprocess.CompletedProcess:
        assert check is True
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(release.subprocess, "run", record_run)

    release.main()

    assert calls == [
        [
            sys.executable,
            "scripts/sync_mcp_release_metadata.py",
            "--check",
            "--expected-version",
            "0.5.22",
        ],
        ["git", "tag", "-a", "v0.5.22", "-m", "Release version 0.5.22"],
        ["git", "push", "origin", "v0.5.22"],
    ]
    assert release.ACTIONS_URL in capsys.readouterr().out


def test_release_preflight_failure_prevents_tag_and_push(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _approve_release(monkeypatch)
    calls: list[list[str]] = []

    def fail_preflight(command: list[str], *, check: bool) -> None:
        assert check is True
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(release.subprocess, "run", fail_preflight)

    with pytest.raises(SystemExit, match="1"):
        release.main()

    assert calls == [
        [
            sys.executable,
            "scripts/sync_mcp_release_metadata.py",
            "--check",
            "--expected-version",
            "0.5.22",
        ]
    ]
