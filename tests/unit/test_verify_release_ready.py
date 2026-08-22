from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts import verify_release_ready


def test_pyproject_check_reads_dynamic_version_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """
[build-system]
build-backend = "setuptools.build_meta"

[project]
name = "openhcs"
dynamic = ["version"]
description = "Microscopy processing"
authors = [{name = "OpenHCS"}]
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert verify_release_ready.check_pyproject_toml()


def test_build_check_uses_active_interpreter_for_build_and_twine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def run(
        command: list[str],
        *,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess:
        assert capture_output is text is check is True
        commands.append(command)
        if command[2] == "build":
            output_dir = Path(command[-1])
            (output_dir / "openhcs.whl").touch()
            (output_dir / "openhcs.tar.gz").touch()
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(verify_release_ready.subprocess, "run", run)

    assert verify_release_ready.try_build()
    assert commands[0][:4] == [sys.executable, "-m", "build", "--outdir"]
    assert commands[1][:3] == [
        sys.executable,
        "-m",
        "scripts.validate_wheel_deployment",
    ]
    assert commands[2][:4] == [sys.executable, "-m", "twine", "check"]
    assert {Path(path).name for path in commands[2][4:]} == {
        "openhcs.whl",
        "openhcs.tar.gz",
    }


def test_build_check_fails_when_twine_rejects_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(
        command: list[str],
        *,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess:
        assert capture_output is text is check is True
        if command[2] == "build":
            (Path(command[-1]) / "openhcs.whl").touch()
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[2] == "scripts.validate_wheel_deployment":
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise subprocess.CalledProcessError(1, command, stderr="invalid metadata")

    monkeypatch.setattr(verify_release_ready.subprocess, "run", run)

    assert not verify_release_ready.try_build()


def test_build_check_fails_when_wheel_contains_unsafe_deployment_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(
        command: list[str],
        *,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess:
        assert capture_output is text is check is True
        if command[2] == "build":
            output_dir = Path(command[-1])
            (output_dir / "openhcs.whl").touch()
            (output_dir / "openhcs.tar.gz").touch()
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise subprocess.CalledProcessError(
            2,
            command,
            stderr="openhcs/build/stale.py: nested build output",
        )

    monkeypatch.setattr(verify_release_ready.subprocess, "run", run)

    assert not verify_release_ready.try_build()


def test_git_check_rejects_untracked_release_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        (
            subprocess.CompletedProcess(
                ["git", "status", "--porcelain"],
                0,
                stdout="?? announcement.md\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                ["git", "branch", "--show-current"],
                0,
                stdout="main\n",
                stderr="",
            ),
        )
    )
    monkeypatch.setattr(
        verify_release_ready.subprocess,
        "run",
        lambda *args, **kwargs: next(responses),
    )

    assert not verify_release_ready.check_git_status()
