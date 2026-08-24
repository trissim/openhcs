from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from packaging.version import Version

from scripts import release, release_readiness

EXACT_SHA = "a" * 40
OTHER_SHA = "b" * 40
REMOTE_SHA = "c" * 40


def test_release_command_runner_preserves_leading_status_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        release_readiness.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=" abc external/owner\n",
            stderr="",
        ),
    )

    output = release_readiness.ReleaseCommandRunner(tmp_path).run(
        "git", "submodule", "status", "--recursive"
    )

    assert output == " abc external/owner"


def test_github_repository_is_derived_from_the_tracked_remote() -> None:
    assert (
        release_readiness.ReleaseRepository.github_identity(
            "https://github.com/OpenHCSDev/openhcs.git"
        )
        == "OpenHCSDev/openhcs"
    )
    assert (
        release_readiness.ReleaseRepository.github_identity(
            "git@github.com:OpenHCSDev/openhcs.git"
        )
        == "OpenHCSDev/openhcs"
    )


def test_github_repository_rejects_a_non_github_release_remote() -> None:
    with pytest.raises(release_readiness.ReleaseReadinessError):
        release_readiness.ReleaseRepository.github_identity(
            "https://example.com/OpenHCSDev/openhcs.git"
        )


def test_release_commit_rejects_an_abbreviated_sha() -> None:
    with pytest.raises(
        release_readiness.ReleaseReadinessError,
        match="Invalid release commit SHA",
    ):
        release_readiness.ReleaseCommit.from_sha("abc123")


def test_release_repository_uses_the_configured_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = {
        ("git", "status", "--porcelain", "--untracked-files=all"): "",
        ("git", "branch", "--show-current"): "main",
        ("git", "config", "--get", "branch.main.remote"): "openhcsdev",
        ("git", "config", "--get", "branch.main.merge"): "refs/heads/main",
        ("git", "rev-parse", "HEAD"): EXACT_SHA,
        ("git", "ls-remote", "openhcsdev", "refs/heads/main"): (
            f"{EXACT_SHA}\trefs/heads/main"
        ),
        ("git", "submodule", "status", "--recursive"): " abc external/owner",
        (
            "git",
            "submodule",
            "foreach",
            "--recursive",
            "--quiet",
            "git status --porcelain --untracked-files=all",
        ): "",
        ("git", "remote", "get-url", "openhcsdev"): (
            "https://github.com/OpenHCSDev/openhcs.git"
        ),
    }
    monkeypatch.setattr(
        release_readiness.ReleaseCommandRunner,
        "run",
        lambda self, *command: responses[command],
    )

    repository = release_readiness.ReleaseRepository.inspect()

    assert repository == release_readiness.ReleaseRepository(
        remote="openhcsdev",
        branch="main",
        commit=release_readiness.ReleaseCommit(EXACT_SHA),
        github_repository="OpenHCSDev/openhcs",
    )


def test_release_repository_rejects_untracked_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        release_readiness.ReleaseCommandRunner,
        "run",
        lambda self, *command: "?? announcement.md",
    )

    with pytest.raises(
        release_readiness.ReleaseReadinessError,
        match="untracked changes",
    ):
        release_readiness.ReleaseRepository.inspect()


def test_release_repository_rejects_remote_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = {
        ("git", "status", "--porcelain", "--untracked-files=all"): "",
        ("git", "branch", "--show-current"): "main",
        ("git", "config", "--get", "branch.main.remote"): "openhcsdev",
        ("git", "config", "--get", "branch.main.merge"): "refs/heads/main",
        ("git", "rev-parse", "HEAD"): EXACT_SHA,
        ("git", "ls-remote", "openhcsdev", "refs/heads/main"): (
            f"{REMOTE_SHA}\trefs/heads/main"
        ),
    }
    monkeypatch.setattr(
        release_readiness.ReleaseCommandRunner,
        "run",
        lambda self, *command: responses[command],
    )

    with pytest.raises(
        release_readiness.ReleaseReadinessError,
        match="does not match",
    ):
        release_readiness.ReleaseRepository.inspect()


def test_workflow_gate_requires_a_successful_exact_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        release_readiness.ReleaseCommandRunner,
        "run",
        lambda self, *command: json.dumps(
            [
                {
                    "headSha": OTHER_SHA,
                    "status": "completed",
                    "conclusion": "success",
                    "url": "https://example.test/other",
                },
                {
                    "headSha": EXACT_SHA,
                    "status": "completed",
                    "conclusion": "success",
                    "url": "https://example.test/exact",
                },
            ]
        ),
    )

    assert (
        release_readiness.ReleaseWorkflowGate.INTEGRATION.successful_run(
            "OpenHCSDev/openhcs",
            release_readiness.ReleaseCommit(EXACT_SHA),
        )
        == "https://example.test/exact"
    )


def test_workflow_gate_rejects_failed_exact_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        release_readiness.ReleaseCommandRunner,
        "run",
        lambda self, *command: json.dumps(
            [
                {
                    "headSha": EXACT_SHA,
                    "status": "completed",
                    "conclusion": "failure",
                    "url": "https://example.test/exact",
                }
            ]
        ),
    )

    with pytest.raises(
        release_readiness.ReleaseReadinessError,
        match="No successful integration-tests.yml",
    ):
        release_readiness.ReleaseWorkflowGate.INTEGRATION.successful_run(
            "OpenHCSDev/openhcs",
            release_readiness.ReleaseCommit(EXACT_SHA),
        )


def test_workflow_gate_rejects_malformed_external_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        release_readiness.ReleaseCommandRunner,
        "run",
        lambda self, *command: json.dumps(
            [
                {
                    "headSha": EXACT_SHA,
                    "status": "completed",
                    "conclusion": "success",
                }
            ]
        ),
    )

    with pytest.raises(
        release_readiness.ReleaseReadinessError,
        match="Malformed GitHub workflow run evidence",
    ):
        release_readiness.ReleaseWorkflowGate.INTEGRATION.successful_run(
            "OpenHCSDev/openhcs",
            release_readiness.ReleaseCommit(EXACT_SHA),
        )


def test_release_tag_gate_rejects_an_existing_local_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = release_readiness.ReleaseRepository(
        remote="openhcsdev",
        branch="main",
        commit=release_readiness.ReleaseCommit(EXACT_SHA),
        github_repository="OpenHCSDev/openhcs",
    )
    monkeypatch.setattr(
        release_readiness.ReleaseCommandRunner,
        "run",
        lambda self, *command: "v1.2.3",
    )

    with pytest.raises(
        release_readiness.ReleaseReadinessError,
        match="already exists locally",
    ):
        release_readiness.ReleaseReadiness._require_release_tag_absent(
            repository,
            Version("1.2.3"),
            release_readiness.RELEASE_COMMANDS,
        )


def test_pypi_version_lookup_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_lookup(*args: object, **kwargs: object) -> None:
        raise release_readiness.requests.ConnectionError("offline")

    monkeypatch.setattr(release_readiness.requests, "get", fail_lookup)

    with pytest.raises(
        release_readiness.ReleaseReadinessError,
        match="Could not prove the latest published OpenHCS version",
    ):
        release_readiness.ReleaseReadiness._require_unpublished_version(
            Version("1.2.3")
        )


def test_build_uses_active_interpreter_and_validates_every_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[tuple[str, ...]] = []

    def run(self: release_readiness.ReleaseCommandRunner, *command: str) -> str:
        commands.append(command)
        if command[1:3] == ("-m", "build"):
            output_dir = Path(command[-1])
            (output_dir / "openhcs.whl").touch()
            (output_dir / "openhcs.tar.gz").touch()
        return ""

    monkeypatch.setattr(release_readiness.ReleaseCommandRunner, "run", run)

    release_readiness.ReleaseReadiness._require_build(
        release_readiness.RELEASE_COMMANDS
    )

    assert commands[0][:3] == (sys.executable, "-m", "build")
    assert commands[1][:3] == (
        sys.executable,
        "-m",
        "scripts.validate_wheel_deployment",
    )
    assert commands[2][:4] == (sys.executable, "-m", "twine", "check")
    assert {Path(path).name for path in commands[2][4:]} == {
        "openhcs.whl",
        "openhcs.tar.gz",
    }


def test_release_pushes_the_tag_through_the_proven_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness = release_readiness.ReleaseReadiness(
        package_version=Version("1.2.3"),
        repository=release_readiness.ReleaseRepository(
            remote="openhcsdev",
            branch="main",
            commit=release_readiness.ReleaseCommit(EXACT_SHA),
            github_repository="OpenHCSDev/openhcs",
        ),
        workflow_runs=("integration", "documentation"),
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(
        release_readiness.ReleaseReadiness,
        "prove",
        classmethod(lambda cls: readiness),
    )
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    monkeypatch.setattr(
        release.subprocess,
        "run",
        lambda command, check: commands.append(command)
        or subprocess.CompletedProcess(command, 0),
    )

    assert release.main() == 0
    assert commands == [
        ["git", "tag", "-a", "v1.2.3", "-m", "Release version 1.2.3"],
        ["git", "push", "openhcsdev", "v1.2.3"],
    ]
