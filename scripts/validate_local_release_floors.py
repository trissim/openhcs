#!/usr/bin/env python3
"""Validate dependency floors against the versions available in local packages.

Package metadata remains the authority.  This check discovers extracted
packages from ``external/*/pyproject.toml`` and verifies that OpenHCS, plus any
local package-to-package requirements, can be resolved by the candidate
versions in those declarations.  It intentionally knows nothing about package
APIs or feature names.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import time
import tomllib
from typing import Protocol

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from scripts.wait_for_pypi_release import (
    PyPIReleaseProbe,
    positive_number,
    wait_for_release,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ProjectMetadata:
    """The dependency metadata required for release-floor validation."""

    name: str
    version: Version | None
    dependencies: tuple[Requirement, ...]
    path: Path

    @property
    def canonical_name(self) -> str:
        return canonicalize_name(self.name)


@dataclass(frozen=True)
class ReleaseCandidate(ProjectMetadata):
    """A project declaration refined to an exact publishable version."""

    version: Version


@dataclass(frozen=True)
class CandidatePublication:
    """Installer-index visibility for one local package declaration."""

    project: ReleaseCandidate
    probe: PyPIReleaseProbe

    def verified_wheel_requirement(self) -> str:
        """Return the exact public wheel requirement owned by this probe."""
        if not self.probe.available:
            raise RuntimeError(
                f"{self.project.name}=={self.project.version}: {self.probe.detail}"
            )
        if self.probe.wheel_url is None:
            raise RuntimeError(
                f"{self.project.name}=={self.project.version}: "
                "available release probe returned no wheel URL"
            )
        return self.probe.wheel_url


class ReleaseWaiter(Protocol):
    """Callable boundary for waiting on one exact public release."""

    def __call__(
        self,
        project: str,
        version: str,
        *,
        timeout_seconds: float,
        poll_interval_seconds: float,
    ) -> PyPIReleaseProbe: ...


def read_project(path: Path) -> ProjectMetadata:
    """Read a project's authoritative PEP 621 metadata."""
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    project = payload["project"]
    version_text = project.get("version")
    return ProjectMetadata(
        name=project["name"],
        version=Version(version_text) if version_text is not None else None,
        dependencies=tuple(
            Requirement(dependency) for dependency in project.get("dependencies", ())
        ),
        path=path,
    )


def read_release_candidate(path: Path) -> ReleaseCandidate:
    """Refine project metadata that declares an exact candidate version."""
    project = read_project(path)
    version = project.version
    if version is None:
        version = _read_hatch_dynamic_version(path)
    return ReleaseCandidate(
        name=project.name,
        version=version,
        dependencies=project.dependencies,
        path=project.path,
    )


def _read_hatch_dynamic_version(path: Path) -> Version:
    """Resolve a Hatch path-backed version from its authoritative declaration."""

    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    project = payload["project"]
    if "version" not in project.get("dynamic", ()):
        raise ValueError(f"Local candidate has no declared project version: {path}")
    try:
        version_source = payload["tool"]["hatch"]["version"]
        relative_source_path = version_source["path"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Local candidate has an unsupported dynamic version declaration: " f"{path}"
        ) from exc
    if not isinstance(relative_source_path, str):
        raise ValueError(f"Hatch version path must be a string: {path}")

    source_path = path.parent / relative_source_path
    module = ast.parse(
        source_path.read_text(encoding="utf-8"),
        filename=str(source_path),
    )
    for statement in module.body:
        value_node = None
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in statement.targets
        ):
            value_node = statement.value
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == "__version__"
        ):
            value_node = statement.value
        if value_node is None:
            continue
        value = ast.literal_eval(value_node)
        if not isinstance(value, str):
            raise ValueError(f"Hatch __version__ must be a string: {source_path}")
        return Version(value)
    raise ValueError(f"Hatch version source has no literal __version__: {source_path}")


def discover_local_projects(
    repo_root: Path = REPO_ROOT,
) -> tuple[ReleaseCandidate, ...]:
    """Discover candidate releases without a parallel package manifest."""
    return tuple(
        read_release_candidate(path)
        for path in sorted((repo_root / "external").glob("*/pyproject.toml"))
    )


def _requirement_accepts(requirement: Requirement, version: Version) -> bool:
    return not requirement.specifier or requirement.specifier.contains(
        version,
        prereleases=True,
    )


def _requires_candidate_floor(
    requirement: Requirement,
    version: Version,
) -> bool:
    """Return whether a direct requirement excludes older candidate versions."""
    for specifier in requirement.specifier:
        if specifier.operator not in {">=", "~=", "==", "==="}:
            continue
        if specifier.operator == "==" and specifier.version.endswith(".*"):
            continue
        try:
            floor = Version(specifier.version)
        except ValueError:
            continue
        if floor >= version:
            return True
    return False


def validate(repo_root: Path = REPO_ROOT) -> tuple[str, ...]:
    """Return release-floor errors, leaving all project metadata untouched."""
    root_project = read_project(repo_root / "pyproject.toml")
    local_projects = discover_local_projects(repo_root)
    if not local_projects:
        return ("No local candidate projects found under external/*/pyproject.toml",)

    candidates: dict[str, ReleaseCandidate] = {}
    errors: list[str] = []
    for project in local_projects:
        if project.version.is_prerelease:
            errors.append(
                f"Local release candidate {project.name}=={project.version} is a "
                "prerelease; installer-facing dependency floors must use stable "
                "published versions"
            )
        existing = candidates.get(project.canonical_name)
        if existing is not None:
            errors.append(
                f"Duplicate local project {project.name!r}: "
                f"{existing.path} and {project.path}"
            )
            continue
        candidates[project.canonical_name] = project

    root_requirements = {
        canonicalize_name(requirement.name): requirement
        for requirement in root_project.dependencies
    }
    for candidate_name, candidate in sorted(candidates.items()):
        requirement = root_requirements.get(candidate_name)
        if requirement is None:
            errors.append(
                f"OpenHCS has no direct dependency on local candidate "
                f"{candidate.name}=={candidate.version}"
            )
            continue
        if not _requirement_accepts(requirement, candidate.version):
            errors.append(
                f"OpenHCS requirement {requirement} excludes available local candidate "
                f"{candidate.name}=={candidate.version}"
            )
        elif not _requires_candidate_floor(requirement, candidate.version):
            errors.append(
                f"OpenHCS requirement {requirement} does not require local candidate "
                f"floor {candidate.name}>={candidate.version}"
            )

    for project in local_projects:
        for requirement in project.dependencies:
            dependency = candidates.get(canonicalize_name(requirement.name))
            if dependency is None:
                continue
            if not _requirement_accepts(requirement, dependency.version):
                errors.append(
                    f"{project.name}=={project.version} requirement {requirement} "
                    f"excludes available local candidate "
                    f"{dependency.name}=={dependency.version}"
                )

    return tuple(errors)


def wait_for_published_candidates(
    repo_root: Path = REPO_ROOT,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
    waiter: ReleaseWaiter = wait_for_release,
    monotonic: Callable[[], float] = time.monotonic,
) -> tuple[CandidatePublication, ...]:
    """Wait for local candidate versions under one shared time bound."""
    deadline = monotonic() + timeout_seconds
    publications: list[CandidatePublication] = []
    for project in discover_local_projects(repo_root):
        remaining = deadline - monotonic()
        if remaining <= 0:
            probe = PyPIReleaseProbe(
                False,
                "shared candidate-publication deadline expired before this "
                "project could be checked",
            )
        else:
            probe = waiter(
                project.name,
                str(project.version),
                timeout_seconds=remaining,
                poll_interval_seconds=poll_interval_seconds,
            )
        publications.append(CandidatePublication(project=project, probe=probe))
        if not probe.available:
            break
    return tuple(publications)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wait-for-pypi",
        action="store_true",
        help=(
            "Wait until every metadata-discovered local candidate is visible "
            "through PyPI's installer-facing index."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=positive_number,
        default=300.0,
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=positive_number,
        default=5.0,
    )
    parser.add_argument(
        "--wheel-requirements-output",
        type=Path,
        help=(
            "Write the metadata-derived, hash-pinned wheel requirements as JSON "
            "after all candidates become available."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    errors = validate()
    if errors:
        for error in errors:
            print(error)
        return 1
    print("Local candidate versions satisfy all extracted-package requirements.")
    if args.wait_for_pypi:
        publications = wait_for_published_candidates(
            timeout_seconds=args.timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
        )
        for publication in publications:
            project = publication.project
            print(f"{project.name}=={project.version}: " f"{publication.probe.detail}")
        if not publications or not all(
            publication.probe.available for publication in publications
        ):
            return 1
        if args.wheel_requirements_output is not None:
            args.wheel_requirements_output.write_text(
                json.dumps(
                    [
                        publication.verified_wheel_requirement()
                        for publication in publications
                    ],
                    separators=(",", ":"),
                ) + "\n",
                encoding="utf-8",
            )
    elif args.wheel_requirements_output is not None:
        print("--wheel-requirements-output requires --wait-for-pypi")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
