#!/usr/bin/env python3
"""Validate OpenHCS external dependency pins against its submodules."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_SUBMODULE_PATHS = (
    Path("external/ObjectState"),
    Path("external/python-introspect"),
    Path("external/metaclass-registry"),
    Path("external/arraybridge"),
    Path("external/pycodify"),
    Path("external/PolyStore"),
    Path("external/pyqt-reactive"),
    Path("external/zmqruntime"),
)

_NORMALIZE_NAME_RE = re.compile(r"[-_.]+")
_EXACT_PIN_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([^\s,;]+)\s*$")


@dataclass(frozen=True)
class ExternalProject:
    name: str
    version: str
    relative_path: Path

    @property
    def path(self) -> Path:
        return PROJECT_ROOT / self.relative_path


def normalize_name(name: str) -> str:
    """Normalize a Python distribution name according to PEP 503."""
    return _NORMALIZE_NAME_RE.sub("-", name).lower()


def load_external_projects() -> tuple[ExternalProject, ...]:
    """Read project names and versions from the pinned submodule checkouts."""
    projects: list[ExternalProject] = []
    missing: list[str] = []

    for relative_path in EXTERNAL_SUBMODULE_PATHS:
        pyproject_path = PROJECT_ROOT / relative_path / "pyproject.toml"
        if not pyproject_path.is_file():
            missing.append(str(relative_path))
            continue

        with pyproject_path.open("rb") as handle:
            project = tomllib.load(handle)["project"]
        projects.append(
            ExternalProject(
                name=project["name"],
                version=project["version"],
                relative_path=relative_path,
            )
        )

    if missing:
        paths = ", ".join(missing)
        raise ValueError(
            f"Uninitialized external submodules: {paths}. "
            "Run 'git submodule update --init --recursive'."
        )

    return tuple(projects)


def load_root_external_pins(
    projects: tuple[ExternalProject, ...],
) -> dict[str, str]:
    """Return exact root dependency pins for the supplied external projects."""
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
        dependencies = tomllib.load(handle)["project"]["dependencies"]

    external_names = {normalize_name(project.name) for project in projects}
    pins: dict[str, str] = {}
    non_exact: list[str] = []

    for dependency in dependencies:
        dependency_name = normalize_name(
            re.split(r"[<>=!~\s\[]", dependency, maxsplit=1)[0]
        )
        if dependency_name not in external_names:
            continue

        match = _EXACT_PIN_RE.fullmatch(dependency)
        if not match:
            non_exact.append(dependency)
            continue
        pins[normalize_name(match.group(1))] = match.group(2)

    if non_exact:
        specs = ", ".join(non_exact)
        raise ValueError(f"External dependencies must use exact pins: {specs}")

    return pins


def validated_external_projects() -> tuple[ExternalProject, ...]:
    """Ensure root pins exactly match the versions in the pinned submodules."""
    projects = load_external_projects()
    pins = load_root_external_pins(projects)
    errors: list[str] = []

    for project in projects:
        name = normalize_name(project.name)
        pinned_version = pins.get(name)
        if pinned_version is None:
            errors.append(f"missing exact dependency pin for {project.name}")
        elif pinned_version != project.version:
            errors.append(
                f"{project.name}: root pins {pinned_version}, "
                f"but {project.relative_path} declares {project.version}"
            )

    if errors:
        raise ValueError("External dependency mismatch:\n- " + "\n- ".join(errors))

    return projects


def pypi_release_errors(
    projects: tuple[ExternalProject, ...], timeout: float = 15.0
) -> list[str]:
    """Return errors for exact dependency releases unavailable from PyPI."""
    errors: list[str] = []
    for project in projects:
        url = (
            f"https://pypi.org/pypi/{quote(project.name)}/{quote(project.version)}/json"
        )
        request = Request(url, headers={"User-Agent": "OpenHCS-release-check/1"})
        try:
            with urlopen(request, timeout=timeout) as response:
                if response.status != 200:
                    errors.append(
                        f"{project.name}=={project.version}: PyPI returned "
                        f"HTTP {response.status}"
                    )
        except HTTPError as exc:
            errors.append(
                f"{project.name}=={project.version}: PyPI returned HTTP {exc.code}"
            )
        except (URLError, TimeoutError) as exc:
            errors.append(
                f"{project.name}=={project.version}: could not query PyPI ({exc})"
            )
    return errors
