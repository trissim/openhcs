"""Resolve the newest ancestor whose named GitHub Actions job succeeded."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any


@dataclass(frozen=True, slots=True)
class WorkflowRun:
    """Workflow-run identity needed by the job-owned baseline resolver."""

    run_id: int
    head_sha: str


@dataclass(frozen=True, slots=True)
class WorkflowJob:
    """Normalized job outcome at the GitHub API boundary."""

    name: str
    successful: bool


def select_successful_job_baseline(
    runs: Sequence[WorkflowRun],
    *,
    current_head_sha: str,
    job_name: str,
    jobs_for_run: Callable[[int], Sequence[WorkflowJob]],
    is_ancestor: Callable[[str, str], bool],
) -> WorkflowRun | None:
    """Return the newest ancestor run whose named job succeeded."""

    for run in runs:
        if run.head_sha == current_head_sha:
            continue
        if not is_ancestor(run.head_sha, current_head_sha):
            continue
        if any(
            job.name == job_name and job.successful for job in jobs_for_run(run.run_id)
        ):
            return run
    return None


class GitHubActionsClient:
    """Read the workflow and job records exposed by the GitHub CLI."""

    def __init__(self, repository: str) -> None:
        self._repository = repository

    def workflow_runs(
        self,
        *,
        workflow: str,
        branch: str,
        maximum_runs: int,
    ) -> tuple[WorkflowRun, ...]:
        payload = self._get_json(
            f"repos/{self._repository}/actions/workflows/{workflow}/runs",
            fields=(
                ("branch", branch),
                ("per_page", str(maximum_runs)),
            ),
        )
        return tuple(
            WorkflowRun(
                run_id=_required_int(record, "id"),
                head_sha=_required_string(record, "head_sha"),
            )
            for record in _required_object_sequence(payload, "workflow_runs")
        )

    def jobs_for_run(self, run_id: int) -> tuple[WorkflowJob, ...]:
        payload = self._get_json(
            f"repos/{self._repository}/actions/runs/{run_id}/jobs",
            fields=(("per_page", "100"),),
        )
        return tuple(
            WorkflowJob(
                name=_required_string(record, "name"),
                successful=(
                    record.get("status") == "completed"
                    and record.get("conclusion") == "success"
                ),
            )
            for record in _required_object_sequence(payload, "jobs")
        )

    @staticmethod
    def _get_json(
        endpoint: str,
        *,
        fields: Sequence[tuple[str, str]],
    ) -> Mapping[str, Any]:
        command = ["gh", "api", "--method", "GET", endpoint]
        for name, value in fields:
            command.extend(("-f", f"{name}={value}"))
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"GitHub API request failed for {endpoint}: {detail}")
        payload = json.loads(completed.stdout)
        if not isinstance(payload, Mapping):
            raise TypeError(f"GitHub API response for {endpoint} is not an object")
        return payload


def git_is_ancestor(candidate_sha: str, head_sha: str) -> bool:
    """Return whether ``candidate_sha`` is an ancestor of ``head_sha``."""

    completed = subprocess.run(
        ("git", "merge-base", "--is-ancestor", candidate_sha, head_sha),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        return True
    if completed.returncode == 1:
        return False
    detail = completed.stderr.strip() or completed.stdout.strip()
    raise RuntimeError(
        f"Git ancestry check failed for {candidate_sha} and {head_sha}: {detail}"
    )


def _required_object_sequence(
    payload: Mapping[str, Any],
    field_name: str,
) -> tuple[Mapping[str, Any], ...]:
    value = payload.get(field_name)
    if not isinstance(value, list) or not all(
        isinstance(item, Mapping) for item in value
    ):
        raise TypeError(f"GitHub API field {field_name!r} is not an object array")
    return tuple(value)


def _required_string(record: Mapping[str, Any], field_name: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value:
        raise TypeError(f"GitHub API field {field_name!r} is not a non-empty string")
    return value


def _required_int(record: Mapping[str, Any], field_name: str) -> int:
    value = record.get(field_name)
    if not isinstance(value, int):
        raise TypeError(f"GitHub API field {field_name!r} is not an integer")
    return value


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Required GitHub Actions environment variable is absent: {name}"
        )
    return value


def _workflow_file_from_environment() -> str:
    workflow_ref = _required_environment("GITHUB_WORKFLOW_REF")
    workflow_path = workflow_ref.split("@", maxsplit=1)[0]
    workflow_file = PurePosixPath(workflow_path).name
    if not workflow_file:
        raise RuntimeError(f"Cannot derive workflow file from {workflow_ref!r}")
    return workflow_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--workflow", default=None)
    parser.add_argument("--branch", default=os.environ.get("GITHUB_REF_NAME"))
    parser.add_argument("--head-sha", default=os.environ.get("GITHUB_SHA"))
    parser.add_argument("--job-name", default=os.environ.get("GITHUB_JOB"))
    parser.add_argument("--maximum-runs", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = args.repository or _required_environment("GITHUB_REPOSITORY")
    workflow = args.workflow or _workflow_file_from_environment()
    branch = args.branch or _required_environment("GITHUB_REF_NAME")
    head_sha = args.head_sha or _required_environment("GITHUB_SHA")
    job_name = args.job_name or _required_environment("GITHUB_JOB")
    if args.maximum_runs < 1 or args.maximum_runs > 100:
        raise ValueError("maximum-runs must be between 1 and 100")

    client = GitHubActionsClient(repository)
    baseline = select_successful_job_baseline(
        client.workflow_runs(
            workflow=workflow,
            branch=branch,
            maximum_runs=args.maximum_runs,
        ),
        current_head_sha=head_sha,
        job_name=job_name,
        jobs_for_run=client.jobs_for_run,
        is_ancestor=git_is_ancestor,
    )
    if baseline is None:
        print(
            f"No successful {job_name!r} job exists on an ancestor workflow run.",
            file=sys.stderr,
        )
        return 1
    print(baseline.head_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
