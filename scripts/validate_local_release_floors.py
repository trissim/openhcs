#!/usr/bin/env python3
"""Validate dependency compatibility against versions in local packages.

Package metadata remains the authority.  This check discovers extracted
packages from ``external/*/pyproject.toml`` and verifies that OpenHCS, plus any
local package-to-package requirements, can be resolved by the candidate versions
in those declarations. OpenHCS requirements must also exclude the next SemVer
breaking series so an already-published application cannot silently resolve a
future incompatible first-party package. It intentionally knows nothing about
package APIs or feature names.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from packaging.requirements import Requirement
from packaging.utils import NormalizedName, canonicalize_name
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
    def canonical_name(self) -> NormalizedName:
        return canonicalize_name(self.name)


@dataclass(frozen=True)
class ReleaseCandidate(ProjectMetadata):
    """A project declaration refined to an exact publishable version."""

    version: Version


@dataclass(frozen=True)
class CandidateRequirementCompatibility:
    """Compatibility proof between one requirement and its local candidate."""

    requirement: Requirement
    candidate: ReleaseCandidate

    @property
    def accepts_candidate(self) -> bool:
        """Return whether the declared range includes the candidate."""

        specifiers = self.requirement.specifier
        return not specifiers or specifiers.contains(
            self.candidate.version,
            prereleases=True,
        )

    @property
    def requires_candidate_floor(self) -> bool:
        """Return whether the range excludes older candidate versions."""

        for specifier in self.requirement.specifier:
            if specifier.operator not in {">=", "~=", "==", "==="}:
                continue
            if specifier.operator == "==" and specifier.version.endswith(".*"):
                continue
            try:
                floor = Version(specifier.version)
            except ValueError:
                continue
            if floor >= self.candidate.version:
                return True
        return False

    @property
    def breaking_release_boundary(self) -> Version:
        """Return the first version outside the candidate's SemVer series."""

        version = self.candidate.version
        if version.major == 0:
            return Version(f"0.{version.minor + 1}.0")
        return Version(f"{version.major + 1}.0.0")

    @property
    def excludes_next_breaking_series(self) -> bool:
        """Return whether the range excludes the next breaking series."""

        boundary = self.breaking_release_boundary
        for specifier in self.requirement.specifier:
            try:
                declared_version = Version(specifier.version.rstrip(".*"))
            except ValueError:
                continue
            if specifier.operator == "<" and declared_version <= boundary:
                return True
            if specifier.operator == "<=" and declared_version < boundary:
                return True
            if specifier.operator in {"==", "==="}:
                return True
            if specifier.operator == "~=" and not specifier.contains(
                boundary,
                prereleases=True,
            ):
                return True
        return False


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


@dataclass(frozen=True)
class LocalCandidateInventory:
    """One metadata-derived topology shared by source and release proofs."""

    root_project: ProjectMetadata
    projects: tuple[ReleaseCandidate, ...]
    projects_by_name: Mapping[NormalizedName, ReleaseCandidate]
    errors: tuple[str, ...]


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


def _release_source_error(project: ReleaseCandidate) -> str | None:
    """Return source drift from the version-owned release tag, when Git-backed."""
    project_root = project.path.parent
    if not (project_root / ".git").exists():
        return None

    release_tag = f"v{project.version}"
    tag_probe = subprocess.run(
        (
            "git",
            "-C",
            str(project_root),
            "rev-parse",
            "--verify",
            "--quiet",
            f"refs/tags/{release_tag}",
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    if tag_probe.returncode != 0:
        return (
            f"Local release candidate {project.name}=={project.version} has no "
            f"matching {release_tag} tag"
        )

    source_diff = subprocess.run(
        (
            "git",
            "-C",
            str(project_root),
            "diff",
            "--name-only",
            release_tag,
            "--",
            "pyproject.toml",
            "src",
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    if source_diff.returncode != 0:
        detail = source_diff.stderr.strip() or "git diff failed"
        return f"Could not validate {project.name} release source: {detail}"
    changed_paths = tuple(source_diff.stdout.splitlines())
    if changed_paths:
        return (
            f"Local release candidate {project.name}=={project.version} differs "
            f"from {release_tag} in published inputs: {', '.join(changed_paths)}"
        )
    return None


def _candidate_inventory(
    repo_root: Path,
) -> LocalCandidateInventory:
    """Discover one metadata-owned local candidate topology."""

    root_project = read_project(repo_root / "pyproject.toml")
    local_projects = discover_local_projects(repo_root)
    if not local_projects:
        return LocalCandidateInventory(
            root_project=root_project,
            projects=(),
            projects_by_name=MappingProxyType({}),
            errors=(
                "No local candidate projects found under external/*/pyproject.toml",
            ),
        )

    candidates: dict[NormalizedName, ReleaseCandidate] = {}
    errors: list[str] = []
    for project in local_projects:
        existing = candidates.get(project.canonical_name)
        if existing is not None:
            errors.append(
                f"Duplicate local project {project.name!r}: "
                f"{existing.path} and {project.path}"
            )
            continue
        candidates[project.canonical_name] = project

    return LocalCandidateInventory(
        root_project=root_project,
        projects=local_projects,
        projects_by_name=MappingProxyType(candidates),
        errors=tuple(errors),
    )


def _candidate_compatibility_errors(
    inventory: LocalCandidateInventory,
) -> tuple[str, ...]:
    """Validate that current package declarations can resolve together."""

    root_requirements = {
        canonicalize_name(requirement.name): requirement
        for requirement in inventory.root_project.dependencies
    }
    errors: list[str] = []
    for candidate_name, candidate in sorted(inventory.projects_by_name.items()):
        requirement = root_requirements.get(candidate_name)
        if requirement is None:
            errors.append(
                f"OpenHCS has no direct dependency on local candidate "
                f"{candidate.name}=={candidate.version}"
            )
            continue
        compatibility = CandidateRequirementCompatibility(requirement, candidate)
        if not compatibility.accepts_candidate:
            errors.append(
                f"OpenHCS requirement {requirement} excludes available local candidate "
                f"{candidate.name}=={candidate.version}"
            )

    for project in inventory.projects:
        for requirement in project.dependencies:
            dependency = inventory.projects_by_name.get(
                canonicalize_name(requirement.name)
            )
            if dependency is None:
                continue
            compatibility = CandidateRequirementCompatibility(requirement, dependency)
            if not compatibility.accepts_candidate:
                errors.append(
                    f"{project.name}=={project.version} requirement {requirement} "
                    f"excludes available local candidate "
                    f"{dependency.name}=={dependency.version}"
                )

    return tuple(errors)


def validate_local_candidate_compatibility(
    repo_root: Path = REPO_ROOT,
) -> tuple[str, ...]:
    """Return errors that prevent current local package snapshots coexisting."""

    inventory = _candidate_inventory(repo_root)
    return inventory.errors + _candidate_compatibility_errors(inventory)


def validate(repo_root: Path = REPO_ROOT) -> tuple[str, ...]:
    """Return installer-facing release-floor errors without mutating metadata."""

    inventory = _candidate_inventory(repo_root)
    errors = list(inventory.errors)
    errors.extend(_candidate_compatibility_errors(inventory))

    root_requirements = {
        canonicalize_name(requirement.name): requirement
        for requirement in inventory.root_project.dependencies
    }
    for candidate_name, candidate in sorted(inventory.projects_by_name.items()):
        if candidate.version.is_prerelease:
            errors.append(
                f"Local release candidate {candidate.name}=={candidate.version} is a "
                "prerelease; installer-facing dependency floors must use stable "
                "published versions"
            )
        requirement = root_requirements.get(candidate_name)
        if requirement is None:
            continue
        compatibility = CandidateRequirementCompatibility(requirement, candidate)
        if not compatibility.accepts_candidate:
            continue
        if not compatibility.requires_candidate_floor:
            errors.append(
                f"OpenHCS requirement {requirement} does not require local candidate "
                f"floor {candidate.name}>={candidate.version}"
            )
        elif not candidate.version.is_prerelease and not (
            compatibility.excludes_next_breaking_series
        ):
            errors.append(
                f"OpenHCS requirement {requirement} does not exclude next breaking "
                f"series {candidate.name}>="
                f"{compatibility.breaking_release_boundary}"
            )

    for project in inventory.projects:
        source_error = _release_source_error(project)
        if source_error is not None:
            errors.append(source_error)

    return tuple(errors)


def wait_for_published_candidates(
    repo_root: Path = REPO_ROOT,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
    waiter: Callable[..., PyPIReleaseProbe] = wait_for_release,
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
                )
                + "\n",
                encoding="utf-8",
            )
    elif args.wheel_requirements_output is not None:
        print("--wheel-requirements-output requires --wait-for-pypi")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
