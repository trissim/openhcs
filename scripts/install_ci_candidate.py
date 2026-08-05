#!/usr/bin/env python3
"""Build and install the CI candidate exclusively through wheel boundaries."""

from __future__ import annotations

import argparse
import subprocess
import sys
from importlib.metadata import version as installed_version
from pathlib import Path

from scripts.validate_local_release_floors import (
    REPO_ROOT,
    discover_local_projects,
    validate,
)


def _build_wheel(project_root: Path, wheel_directory: Path) -> None:
    subprocess.run(
        (
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--outdir",
            str(wheel_directory),
            str(project_root),
        ),
        check=True,
    )


def _root_wheel(wheel_directory: Path) -> Path:
    wheels = tuple(sorted(wheel_directory.glob("openhcs-*.whl")))
    if len(wheels) != 1:
        raise RuntimeError(
            "Expected exactly one OpenHCS candidate wheel, found "
            f"{[wheel.name for wheel in wheels]}"
        )
    return wheels[0]


def build_and_install_candidate(
    *,
    extras: tuple[str, ...],
    dependency_source: str,
    wheel_directory: Path,
    additional_requirements: tuple[str, ...],
    local_project_extras: tuple[str, ...],
) -> None:
    """Install the root wheel against either public or locally built wheels."""

    wheel_directory.mkdir(parents=True, exist_ok=True)
    local_projects = discover_local_projects()
    if dependency_source == "submodules":
        errors = validate()
        if errors:
            raise RuntimeError("\n".join(errors))
        for project in local_projects:
            _build_wheel(project.path.parent, wheel_directory)

    _build_wheel(REPO_ROOT, wheel_directory)
    root_wheel = _root_wheel(wheel_directory)
    extras_suffix = f"[{','.join(extras)}]" if extras else ""
    subprocess.run(
        (
            sys.executable,
            "-m",
            "pip",
            "install",
            "--find-links",
            str(wheel_directory),
            f"{root_wheel}{extras_suffix}",
            *additional_requirements,
        ),
        check=True,
        cwd=wheel_directory,
    )
    subprocess.run(
        (
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--force-reinstall",
            str(root_wheel),
        ),
        check=True,
        cwd=wheel_directory,
    )

    if dependency_source == "submodules":
        if local_project_extras:
            extras_suffix = f"[{','.join(local_project_extras)}]"
            subprocess.run(
                (
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--find-links",
                    str(wheel_directory),
                    *(
                        f"{project.name}{extras_suffix}=={project.version}"
                        for project in local_projects
                    ),
                ),
                check=True,
                cwd=wheel_directory,
            )
        mismatches = tuple(
            f"{project.name}: installed {installed_version(project.name)}, "
            f"candidate {project.version}"
            for project in local_projects
            if installed_version(project.name) != str(project.version)
        )
        if mismatches:
            raise RuntimeError(
                "CI resolved packages other than the built submodule candidates:\n"
                + "\n".join(mismatches)
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extras",
        default="",
        help="Comma-separated OpenHCS extras to install from the candidate wheel.",
    )
    parser.add_argument(
        "--dependency-source",
        choices=("pypi", "submodules"),
        default="pypi",
    )
    parser.add_argument("--wheel-directory", type=Path, required=True)
    parser.add_argument(
        "--requirement",
        action="append",
        default=[],
        help="Additional non-editable requirement to install with the candidate.",
    )
    parser.add_argument(
        "--local-project-extra",
        action="append",
        default=[],
        help="Extra to install from every metadata-discovered local project.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    build_and_install_candidate(
        extras=tuple(part for part in args.extras.split(",") if part),
        dependency_source=args.dependency_source,
        wheel_directory=args.wheel_directory.resolve(),
        additional_requirements=tuple(args.requirement),
        local_project_extras=tuple(args.local_project_extra),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
