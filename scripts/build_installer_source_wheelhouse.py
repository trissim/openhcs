#!/usr/bin/env python3
"""Build the candidate wheels consumed by installer source tests.

Project metadata remains the package authority.  Submodule mode discovers
first-party dependency candidates from ``external/*/pyproject.toml`` instead
of maintaining a second package list in the workflow.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
import subprocess
import sys

from scripts.validate_local_release_floors import discover_local_projects


REPO_ROOT = Path(__file__).resolve().parents[1]
BuildRunner = Callable[[Sequence[str]], None]


def source_projects(
    repo_root: Path,
    dependency_source: str,
) -> tuple[Path, ...]:
    """Return source trees whose wheels belong in the installer wheelhouse."""

    projects = [repo_root]
    if dependency_source == "submodules":
        projects.extend(
            project.path.parent for project in discover_local_projects(repo_root)
        )
    return tuple(projects)


def _run_build(command: Sequence[str]) -> None:
    subprocess.run(command, check=True)


def build_wheelhouse(
    output_directory: Path,
    dependency_source: str,
    *,
    repo_root: Path = REPO_ROOT,
    runner: BuildRunner = _run_build,
) -> tuple[Path, ...]:
    """Build the root candidate and any metadata-discovered local dependencies."""

    output_directory.mkdir(parents=True, exist_ok=True)
    projects = source_projects(repo_root, dependency_source)
    for project in projects:
        runner(
            (
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--outdir",
                str(output_directory),
                str(project),
            )
        )
    return projects


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dependency-source",
        choices=("submodules", "pypi"),
        required=True,
    )
    args = parser.parse_args(argv)
    build_wheelhouse(args.output, args.dependency_source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
