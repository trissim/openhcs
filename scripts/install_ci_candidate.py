"""Build and install the CI candidate exclusively through wheel boundaries."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from importlib.metadata import version as installed_version
from pathlib import Path

from packaging.utils import (
    InvalidWheelFilename,
    canonicalize_name,
    parse_wheel_filename,
)

from scripts.validate_local_release_floors import (
    REPO_ROOT,
    ReleaseCandidate,
    discover_local_projects,
    validate_local_candidate_compatibility,
)
from scripts.validate_wheel_deployment import validate_wheel_deployment


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


def _existing_root_wheel(candidate_wheel: Path) -> Path:
    """Validate and return an explicitly supplied OpenHCS candidate wheel."""

    root_wheel = candidate_wheel.resolve()
    if not root_wheel.is_file():
        raise RuntimeError(f"Candidate wheel does not exist: {root_wheel}")
    try:
        distribution_name, _version, _build, _tags = parse_wheel_filename(
            root_wheel.name
        )
    except InvalidWheelFilename as exc:
        raise RuntimeError(f"Candidate is not a valid wheel: {root_wheel}") from exc
    if distribution_name != canonicalize_name("openhcs"):
        raise RuntimeError(f"Candidate wheel is not OpenHCS: {root_wheel}")
    return root_wheel


def build_and_install_candidate(
    *,
    extras: tuple[str, ...],
    dependency_source: str,
    wheel_directory: Path,
    additional_requirements: tuple[str, ...],
    local_project_extras: tuple[str, ...],
    published_wheel_requirements: tuple[str, ...],
    candidate_wheel: Path | None = None,
) -> None:
    """Install the root wheel against either public or locally built wheels."""

    wheel_directory.mkdir(parents=True, exist_ok=True)
    local_projects: tuple[ReleaseCandidate, ...] = ()
    dependency_requirements: tuple[str, ...] = ()
    if dependency_source == "submodules":
        local_projects = discover_local_projects()
        errors = validate_local_candidate_compatibility()
        if errors:
            raise RuntimeError("\n".join(errors))
        for project in local_projects:
            _build_wheel(project.path.parent, wheel_directory)
    elif not published_wheel_requirements:
        raise RuntimeError(
            "PyPI candidate installation requires the readiness job's "
            "metadata-derived wheel requirements."
        )
    else:
        dependency_requirements = published_wheel_requirements

    if candidate_wheel is None:
        _build_wheel(REPO_ROOT, wheel_directory)
        root_wheel = _root_wheel(wheel_directory)
    else:
        root_wheel = _existing_root_wheel(candidate_wheel)
    deployment_errors = validate_wheel_deployment(root_wheel)
    if deployment_errors:
        raise RuntimeError("\n".join(deployment_errors))
    extras_suffix = f"[{','.join(extras)}]" if extras else ""
    subprocess.run(
        (
            sys.executable,
            "-m",
            "pip",
            "install",
            "--find-links",
            str(wheel_directory),
            *dependency_requirements,
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
        "--candidate-wheel",
        type=Path,
        help="Install this existing OpenHCS wheel instead of rebuilding it.",
    )
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
    parser.add_argument(
        "--published-wheel-requirements-json",
        default="[]",
        help=(
            "JSON array of hash-pinned wheel URLs derived by the dependency "
            "readiness job."
        ),
    )
    return parser


def _published_wheel_requirements(value: str) -> tuple[str, ...]:
    """Validate the readiness job's serialized wheel projection."""
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("Published wheel requirements must be valid JSON.") from exc
    if not isinstance(payload, list) or any(
        not isinstance(requirement, str) or not requirement for requirement in payload
    ):
        raise ValueError(
            "Published wheel requirements must be a JSON array of non-empty strings."
        )
    return tuple(payload)


def main() -> int:
    args = _parser().parse_args()
    build_and_install_candidate(
        extras=tuple(part for part in args.extras.split(",") if part),
        dependency_source=args.dependency_source,
        wheel_directory=args.wheel_directory.resolve(),
        additional_requirements=tuple(args.requirement),
        local_project_extras=tuple(args.local_project_extra),
        published_wheel_requirements=_published_wheel_requirements(
            args.published_wheel_requirements_json
        ),
        candidate_wheel=args.candidate_wheel,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
