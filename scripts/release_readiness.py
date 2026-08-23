#!/usr/bin/env python3
"""Prove that the current OpenHCS commit is safe to tag for publication."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse

import requests
from packaging.version import Version

from scripts.sync_mcp_release_metadata import read_package_version, synchronize

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class ReleaseReadinessError(RuntimeError):
    """Raised when release readiness cannot be proven."""


@dataclass(frozen=True)
class ReleaseCommandRunner:
    """Execute release proof commands from the canonical repository root."""

    repository_root: Path

    def run(self, *command: str) -> str:
        """Return command output or translate process failure into proof failure."""
        try:
            result = subprocess.run(
                command,
                cwd=self.repository_root,
                capture_output=True,
                text=True,
                check=True,
            )
        except FileNotFoundError as exc:
            raise ReleaseReadinessError(
                f"Required command is unavailable: {command[0]}"
            ) from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            raise ReleaseReadinessError(
                f"Command failed ({' '.join(command)}): {detail}"
            ) from exc
        return result.stdout.strip()


RELEASE_COMMANDS = ReleaseCommandRunner(REPOSITORY_ROOT)


@dataclass(frozen=True)
class ReleaseCommit:
    """Validated immutable commit identity used by every release proof gate."""

    sha: str

    @classmethod
    def from_sha(cls, sha: str) -> ReleaseCommit:
        """Validate and construct one exact Git commit identity."""
        if re.fullmatch(r"[0-9a-f]{40}", sha) is None:
            raise ReleaseReadinessError(f"Invalid release commit SHA: {sha!r}")
        return cls(sha=sha)

    def successful_workflow_runs(
        self,
        repository: str,
        command_runner: ReleaseCommandRunner = RELEASE_COMMANDS,
    ) -> tuple[str, ...]:
        """Verify every release workflow gate for this commit."""
        if re.fullmatch(r"[^/]+/[^/]+", repository) is None:
            raise ReleaseReadinessError(
                f"Invalid GitHub repository identity: {repository!r}"
            )
        return tuple(
            gate.successful_run(repository, self, command_runner)
            for gate in ReleaseWorkflowGate
        )


@dataclass(frozen=True)
class GitHubWorkflowRun:
    """Validated GitHub workflow evidence for one observed run."""

    commit: ReleaseCommit
    status: str
    conclusion: str | None
    url: str

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, str | None],
    ) -> GitHubWorkflowRun:
        """Validate the GitHub CLI payload before it reaches release policy."""
        try:
            commit_sha = payload["headSha"]
            status = payload["status"]
            conclusion = payload["conclusion"]
            url = payload["url"]
        except KeyError as exc:
            raise ReleaseReadinessError(
                "Malformed GitHub workflow run evidence."
            ) from exc
        if (
            not isinstance(commit_sha, str)
            or not isinstance(status, str)
            or (conclusion is not None and not isinstance(conclusion, str))
            or not isinstance(url, str)
        ):
            raise ReleaseReadinessError("Malformed GitHub workflow run evidence.")
        return cls(
            commit=ReleaseCommit.from_sha(commit_sha),
            status=status,
            conclusion=conclusion,
            url=url,
        )

    def proves_success(self, commit: ReleaseCommit) -> bool:
        """Return whether this run proves success for the requested commit."""
        return (
            self.commit == commit
            and self.status == "completed"
            and self.conclusion == "success"
        )


class ReleaseWorkflowGate(Enum):
    """Required exact-commit workflow evidence for an OpenHCS release."""

    INTEGRATION = "integration-tests.yml"
    DOCUMENTATION = "docs.yml"

    def successful_run(
        self,
        repository: str,
        commit: ReleaseCommit,
        command_runner: ReleaseCommandRunner = RELEASE_COMMANDS,
    ) -> str:
        """Return the successful exact-commit run URL or fail closed."""
        output = command_runner.run(
            "gh",
            "run",
            "list",
            "--repo",
            repository,
            "--commit",
            commit.sha,
            "--workflow",
            self.value,
            "--event",
            "push",
            "--limit",
            "10",
            "--json",
            "headSha,status,conclusion,url",
        )
        try:
            payload = json.loads(output)
        except json.JSONDecodeError as exc:
            raise ReleaseReadinessError(
                f"Could not parse {self.value} workflow evidence: {exc}"
            ) from exc
        if not isinstance(payload, list) or not all(
            isinstance(run, dict) for run in payload
        ):
            raise ReleaseReadinessError(f"Malformed {self.value} workflow evidence.")
        runs = tuple(GitHubWorkflowRun.from_payload(run) for run in payload)
        successful = next(
            (run for run in runs if run.proves_success(commit)),
            None,
        )
        if successful is None:
            raise ReleaseReadinessError(
                f"No successful {self.value} push run exists for {commit.sha}."
            )
        return successful.url


@dataclass(frozen=True)
class ReleaseRepository:
    """Exact local, upstream, and GitHub identity of the release checkout."""

    remote: str
    branch: str
    commit: ReleaseCommit
    github_repository: str

    @classmethod
    def inspect(
        cls,
        command_runner: ReleaseCommandRunner = RELEASE_COMMANDS,
    ) -> ReleaseRepository:
        """Resolve and validate the release checkout without guessed remotes."""
        status = command_runner.run(
            "git", "status", "--porcelain", "--untracked-files=all"
        )
        if status:
            raise ReleaseReadinessError(
                "Release checkout has staged, unstaged, or untracked changes."
            )

        branch = command_runner.run("git", "branch", "--show-current")
        if branch != "main":
            raise ReleaseReadinessError(
                f"Release checkout is on {branch!r}; expected 'main'."
            )

        remote = command_runner.run("git", "config", "--get", f"branch.{branch}.remote")
        tracked_ref = command_runner.run(
            "git", "config", "--get", f"branch.{branch}.merge"
        )
        expected_ref = f"refs/heads/{branch}"
        if not remote or tracked_ref != expected_ref:
            raise ReleaseReadinessError(
                f"Branch {branch!r} does not track one {expected_ref!r} upstream."
            )

        commit = ReleaseCommit.from_sha(command_runner.run("git", "rev-parse", "HEAD"))
        upstream_rows = command_runner.run(
            "git", "ls-remote", remote, expected_ref
        ).splitlines()
        if len(upstream_rows) != 1:
            raise ReleaseReadinessError(
                f"Could not resolve exactly one {remote}/{branch} upstream commit."
            )
        upstream_sha, resolved_ref = upstream_rows[0].split(maxsplit=1)
        if resolved_ref != expected_ref or upstream_sha != commit.sha:
            raise ReleaseReadinessError(
                f"Local {commit.sha} does not match {remote}/{branch} at "
                f"{upstream_sha}."
            )

        submodule_status = command_runner.run(
            "git", "submodule", "status", "--recursive"
        )
        invalid_submodules = tuple(
            line for line in submodule_status.splitlines() if not line.startswith(" ")
        )
        if invalid_submodules:
            raise ReleaseReadinessError(
                "Recursive submodules are not at their recorded commits: "
                + "; ".join(invalid_submodules)
            )
        dirty_submodules = command_runner.run(
            "git",
            "submodule",
            "foreach",
            "--recursive",
            "--quiet",
            "git status --porcelain --untracked-files=all",
        )
        if dirty_submodules:
            raise ReleaseReadinessError(
                "Recursive submodules contain uncommitted or untracked files."
            )

        remote_url = command_runner.run("git", "remote", "get-url", remote)
        return cls(
            remote=remote,
            branch=branch,
            commit=commit,
            github_repository=cls.github_identity(remote_url),
        )

    @staticmethod
    def github_identity(remote_url: str) -> str:
        """Derive the GitHub owner/name identity from the proven upstream URL."""
        if remote_url.startswith("git@github.com:"):
            repository = remote_url.removeprefix("git@github.com:")
        else:
            parsed = urlparse(remote_url)
            if parsed.hostname != "github.com":
                raise ReleaseReadinessError(
                    "Tracked release remote is not hosted on github.com: "
                    f"{remote_url}"
                )
            repository = parsed.path.lstrip("/")
        repository = repository.removesuffix(".git")
        if re.fullmatch(r"[^/]+/[^/]+", repository) is None:
            raise ReleaseReadinessError(
                "Could not derive a GitHub repository from release remote: "
                f"{remote_url}"
            )
        return repository


@dataclass(frozen=True)
class ReleaseReadiness:
    """Proof consumed by the tag-producing release command."""

    package_version: Version
    repository: ReleaseRepository
    workflow_runs: tuple[str, ...]

    @classmethod
    def prove(
        cls,
        command_runner: ReleaseCommandRunner = RELEASE_COMMANDS,
    ) -> ReleaseReadiness:
        """Validate every release prerequisite and return its exact evidence."""
        cls._require_build_dependencies()
        package_version = read_package_version()
        cls._require_release_metadata(package_version)
        cls._require_unpublished_version(package_version)
        repository = ReleaseRepository.inspect(command_runner)
        cls._require_release_tag_absent(
            repository,
            package_version,
            command_runner,
        )
        workflow_runs = repository.commit.successful_workflow_runs(
            repository.github_repository,
            command_runner,
        )
        cls._require_build(command_runner)
        return cls(
            package_version=package_version,
            repository=repository,
            workflow_runs=workflow_runs,
        )

    @staticmethod
    def _require_build_dependencies() -> None:
        missing_modules = tuple(
            name
            for name in ("build", "packaging", "requests", "twine")
            if importlib.util.find_spec(name) is None
        )
        if missing_modules:
            raise ReleaseReadinessError(
                "Missing release dependencies: " + ", ".join(missing_modules)
            )
        if shutil.which("git") is None or shutil.which("gh") is None:
            raise ReleaseReadinessError("Release validation requires git and gh.")

    @staticmethod
    def _require_release_metadata(package_version: Version) -> None:
        drift = synchronize(check=True, package_version=package_version)
        if drift:
            relative = ", ".join(
                str(path.relative_to(REPOSITORY_ROOT)) for path in drift
            )
            raise ReleaseReadinessError(
                f"Generated release metadata is out of sync: {relative}"
            )

    @staticmethod
    def _require_unpublished_version(package_version: Version) -> None:
        try:
            response = requests.get(
                "https://pypi.org/pypi/openhcs/json",
                timeout=30,
            )
            response.raise_for_status()
            published_version = Version(response.json()["info"]["version"])
        except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
            raise ReleaseReadinessError(
                f"Could not prove the latest published OpenHCS version: {exc}"
            ) from exc
        if package_version <= published_version:
            raise ReleaseReadinessError(
                f"Package version {package_version} is not newer than PyPI "
                f"version {published_version}."
            )

    @staticmethod
    def _require_release_tag_absent(
        repository: ReleaseRepository,
        package_version: Version,
        command_runner: ReleaseCommandRunner,
    ) -> None:
        tag = f"v{package_version}"
        tag_ref = f"refs/tags/{tag}"
        if command_runner.run("git", "tag", "--list", tag):
            raise ReleaseReadinessError(
                f"Release tag already exists locally: {tag_ref}"
            )
        if command_runner.run("git", "ls-remote", "--tags", repository.remote, tag_ref):
            raise ReleaseReadinessError(
                f"Release tag already exists upstream: {tag_ref}"
            )

    @staticmethod
    def _require_build(command_runner: ReleaseCommandRunner) -> None:
        with tempfile.TemporaryDirectory(prefix="openhcs-release-") as output_dir:
            command_runner.run(
                sys.executable,
                "-m",
                "build",
                "--outdir",
                output_dir,
            )
            distributions = tuple(sorted(Path(output_dir).iterdir()))
            wheels = tuple(path for path in distributions if path.suffix == ".whl")
            if not distributions or not wheels:
                raise ReleaseReadinessError(
                    "The release build did not produce both validated "
                    "distributions and a wheel."
                )
            for wheel in wheels:
                command_runner.run(
                    sys.executable,
                    "-m",
                    "scripts.validate_wheel_deployment",
                    str(wheel),
                )
            command_runner.run(
                sys.executable,
                "-m",
                "twine",
                "check",
                *(str(path) for path in distributions),
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        help="GitHub owner/name used for exact workflow evidence verification.",
    )
    parser.add_argument(
        "--commit",
        help="Full commit SHA used for exact workflow evidence verification.",
    )
    args = parser.parse_args()
    try:
        if (args.repository is None) != (args.commit is None):
            raise ReleaseReadinessError(
                "--repository and --commit must be provided together."
            )
        if args.repository is not None and args.commit is not None:
            commit = ReleaseCommit.from_sha(args.commit)
            for workflow_run in commit.successful_workflow_runs(args.repository):
                print(workflow_run)
            return 0
        readiness = ReleaseReadiness.prove()
    except ReleaseReadinessError as exc:
        print(f"Release readiness failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"OpenHCS {readiness.package_version} is ready at "
        f"{readiness.repository.commit.sha}."
    )
    for workflow_run in readiness.workflow_runs:
        print(workflow_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
