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
from dataclasses import dataclass
from pathlib import Path
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version


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


def read_project(path: Path, *, require_version: bool = True) -> ProjectMetadata:
    """Read a project's authoritative PEP 621 metadata."""
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    project = payload["project"]
    version_text = project.get("version")
    if require_version and version_text is None:
        raise ValueError(f"Local candidate has no literal project.version: {path}")
    return ProjectMetadata(
        name=project["name"],
        version=Version(version_text) if version_text is not None else None,
        dependencies=tuple(
            Requirement(dependency) for dependency in project.get("dependencies", ())
        ),
        path=path,
    )


def discover_local_projects(repo_root: Path = REPO_ROOT) -> tuple[ProjectMetadata, ...]:
    """Discover candidate releases without a parallel package manifest."""
    return tuple(
        read_project(path)
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
    root_project = read_project(repo_root / "pyproject.toml", require_version=False)
    local_projects = discover_local_projects(repo_root)
    if not local_projects:
        return ("No local candidate projects found under external/*/pyproject.toml",)

    candidates: dict[str, ProjectMetadata] = {}
    errors: list[str] = []
    for project in local_projects:
        assert project.version is not None
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
        assert candidate.version is not None
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
        assert project.version is not None
        for requirement in project.dependencies:
            dependency = candidates.get(canonicalize_name(requirement.name))
            if dependency is None:
                continue
            assert dependency.version is not None
            if not _requirement_accepts(requirement, dependency.version):
                errors.append(
                    f"{project.name}=={project.version} requirement {requirement} "
                    f"excludes available local candidate "
                    f"{dependency.name}=={dependency.version}"
                )

    return tuple(errors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    errors = validate()
    if errors:
        for error in errors:
            print(error)
        return 1
    print("Local candidate versions satisfy all extracted-package requirements.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
