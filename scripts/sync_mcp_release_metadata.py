#!/usr/bin/env python3
"""Synchronize client-distribution metadata with ``openhcs.__version__``.

The Python package version and selected desktop capability surface are the
authorities. Dependency-free checks project directly readable package/version
structure. Once the built wheel is installed, the full check imports the
canonical capability registry and validates its required extras.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
import tomllib

from packaging.requirements import Requirement
from packaging.version import Version


REPO_ROOT = Path(__file__).resolve().parents[1]
INIT_PATH = REPO_ROOT / "openhcs" / "__init__.py"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
PLUGIN_MANIFEST_PATH = (
    REPO_ROOT / "packaging" / "codex" / "openhcs" / ".codex-plugin" / "plugin.json"
)
PLUGIN_MCP_PATH = REPO_ROOT / "packaging" / "codex" / "openhcs" / ".mcp.json"
MCPB_ROOT = REPO_ROOT / "packaging" / "mcpb" / "openhcs"
MCPB_MANIFEST_PATH = MCPB_ROOT / "manifest.json"
MCPB_PYPROJECT_PATH = MCPB_ROOT / "pyproject.toml"
REGISTRY_PATH = REPO_ROOT / "server.json"


def read_package_version(path: Path = INIT_PATH) -> Version:
    """Read the literal package version without importing OpenHCS."""
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in statement.targets
        ):
            continue
        if isinstance(statement.value, ast.Constant) and isinstance(
            statement.value.value,
            str,
        ):
            return Version(statement.value.value)
    raise ValueError(f"No literal __version__ assignment found in {path}")


def plugin_semver(package_version: Version) -> str:
    """Project a PEP 440 OpenHCS version into strict SemVer metadata."""
    base = f"{package_version.major}.{package_version.minor}.{package_version.micro}"
    prerelease: list[str] = []
    if package_version.pre is not None:
        label, number = package_version.pre
        prerelease.extend((label, str(number)))
    if package_version.dev is not None:
        prerelease.extend(("dev", str(package_version.dev)))
    if prerelease:
        return f"{base}-{'.'.join(prerelease)}"
    return base


def read_package_name(path: Path = PYPROJECT_PATH) -> str:
    """Read the canonical Python distribution name from project metadata."""
    project = tomllib.loads(path.read_text(encoding="utf-8"))["project"]
    package_name = project["name"]
    if not isinstance(package_name, str) or not package_name:
        raise ValueError(f"Invalid project.name in {path}")
    return package_name


def read_optional_dependency_names(path: Path = PYPROJECT_PATH) -> frozenset[str]:
    """Read the optional-extra names declared by canonical project metadata."""
    project = tomllib.loads(path.read_text(encoding="utf-8"))["project"]
    optional_dependencies = project.get("optional-dependencies", {})
    if not isinstance(optional_dependencies, dict):
        raise ValueError(f"Invalid project.optional-dependencies in {path}")
    return frozenset(optional_dependencies)


def _declared_distribution_requirements() -> tuple[Requirement, ...]:
    """Read the three packaging declarations without assigning semantics."""
    package_name = read_package_name()

    plugin_mcp = json.loads(PLUGIN_MCP_PATH.read_text(encoding="utf-8"))
    plugin_arguments = plugin_mcp["mcpServers"][package_name]["args"]
    plugin_requirement = next(
        argument
        for argument in plugin_arguments
        if isinstance(argument, str) and argument.startswith(f"{package_name}[")
    )

    mcpb_project = tomllib.loads(MCPB_PYPROJECT_PATH.read_text(encoding="utf-8"))[
        "project"
    ]
    mcpb_requirement = next(
        dependency
        for dependency in mcpb_project["dependencies"]
        if Requirement(dependency).name == package_name
    )

    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry_package = next(
        package
        for package in registry["packages"]
        if package["identifier"] == package_name
    )
    registry_requirement = next(
        argument["value"]
        for argument in registry_package["runtimeArguments"]
        if argument.get("name") == "--with"
    )

    return tuple(
        Requirement(requirement)
        for requirement in (
            plugin_requirement,
            mcpb_requirement,
            registry_requirement,
        )
    )


def metadata_package_extras() -> tuple[str, ...]:
    """Validate and return the synchronized packaging-declared extra names."""
    package_name = read_package_name()
    requirements = _declared_distribution_requirements()
    for requirement in requirements:
        if (
            requirement.name != package_name
            or requirement.url is not None
            or requirement.marker is not None
            or len(tuple(requirement.specifier)) != 1
            or next(iter(requirement.specifier)).operator != "=="
        ):
            raise ValueError(
                "Desktop distribution metadata must use one exact local "
                f"{package_name!r} requirement."
            )
    extra_sets = {frozenset(requirement.extras) for requirement in requirements}
    if len(extra_sets) != 1:
        raise ValueError(
            "Desktop distribution metadata declares inconsistent package extras."
        )
    extras = tuple(sorted(next(iter(extra_sets))))
    unknown_extras = set(extras) - read_optional_dependency_names()
    if unknown_extras:
        raise ValueError(
            "Desktop distribution metadata names undeclared package extras: "
            f"{sorted(unknown_extras)}"
        )
    return extras


def desktop_package_requirement(
    package_version: Version,
    *,
    capability_requirements: bool = False,
) -> str:
    """Project the exact desktop requirement from available authorities."""
    if capability_requirements:
        from openhcs.agent.capabilities import (
            CapabilityTransport,
            DesktopLocalCapabilitySurfaceProfile,
            get_capability_registry,
        )

        profile = DesktopLocalCapabilitySurfaceProfile()
        registry = get_capability_registry(
            capability_transport=CapabilityTransport.LOCAL_STDIO,
            capability_surface_profile=profile,
        )
        extras = profile.distribution_extras(registry.capabilities)
    else:
        extras = metadata_package_extras()
    unknown_extras = set(extras) - read_optional_dependency_names()
    if unknown_extras:
        raise ValueError(
            "Desktop capability surface requires undeclared package extras: "
            f"{sorted(unknown_extras)}"
        )
    extras_clause = f"[{','.join(extras)}]" if extras else ""
    return f"{read_package_name()}{extras_clause}=={package_version}"


def _json_text(payload: dict) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def _replace_exactly_once(text: str, pattern: str, replacement: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(
            f"Expected one metadata field matching {pattern!r}, found {count}"
        )
    return updated


def projected_files(
    package_version: Version,
    *,
    capability_requirements: bool = False,
) -> dict[Path, str]:
    """Return the complete version projection without writing it."""
    pep440 = str(package_version)
    semver = plugin_semver(package_version)
    package_name = read_package_name()
    package_requirement = desktop_package_requirement(
        package_version,
        capability_requirements=capability_requirements,
    )

    plugin_manifest = json.loads(PLUGIN_MANIFEST_PATH.read_text(encoding="utf-8"))
    plugin_manifest["version"] = semver

    plugin_mcp = json.loads(PLUGIN_MCP_PATH.read_text(encoding="utf-8"))
    plugin_args = plugin_mcp["mcpServers"]["openhcs"]["args"]
    plugin_args[
        plugin_args.index(
            next(arg for arg in plugin_args if arg.startswith(f"{package_name}["))
        )
    ] = package_requirement

    mcpb_manifest = json.loads(MCPB_MANIFEST_PATH.read_text(encoding="utf-8"))
    mcpb_manifest["version"] = semver

    mcpb_pyproject = MCPB_PYPROJECT_PATH.read_text(encoding="utf-8")
    mcpb_pyproject = _replace_exactly_once(
        mcpb_pyproject,
        r'^version = "[^"]+"$',
        f'version = "{pep440}"',
    )
    mcpb_pyproject = _replace_exactly_once(
        mcpb_pyproject,
        rf'^    "{re.escape(package_name)}\[[^]]+\]==[^"]+",$',
        f'    "{package_requirement}",',
    )

    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry["version"] = pep440
    package = registry["packages"][0]
    package["version"] = pep440
    for argument in package.get("runtimeArguments", []):
        if argument.get("name") == "--with":
            argument["value"] = package_requirement

    return {
        PLUGIN_MANIFEST_PATH: _json_text(plugin_manifest),
        PLUGIN_MCP_PATH: _json_text(plugin_mcp),
        MCPB_MANIFEST_PATH: _json_text(mcpb_manifest),
        MCPB_PYPROJECT_PATH: mcpb_pyproject,
        REGISTRY_PATH: _json_text(registry),
    }


def synchronize(
    *,
    check: bool = False,
    package_version: Version | None = None,
    capability_requirements: bool = False,
) -> tuple[Path, ...]:
    """Write projections, or return drift and leave files untouched in check mode."""
    changed: list[Path] = []
    resolved_version = package_version or read_package_version()
    for path, projected in projected_files(
        resolved_version,
        capability_requirements=capability_requirements,
    ).items():
        current = path.read_text(encoding="utf-8")
        if current == projected:
            continue
        changed.append(path)
        if not check:
            path.write_text(projected, encoding="utf-8")
    return tuple(changed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when generated MCP release metadata is out of sync.",
    )
    parser.add_argument(
        "--expected-version",
        help="Fail unless the package authority matches this release or tag version.",
    )
    parser.add_argument(
        "--capability-requirements",
        action="store_true",
        help=(
            "Import the installed OpenHCS capability registry and validate/project "
            "the selected desktop surface's complete package extras."
        ),
    )
    parser.add_argument(
        "--print-desktop-extras",
        action="store_true",
        help="Print synchronized packaging-declared desktop extras and exit.",
    )
    args = parser.parse_args()
    package_version = read_package_version()
    if args.expected_version is not None and package_version != Version(
        args.expected_version
    ):
        print(
            "Package version does not match expected release version: "
            f"{package_version} != {Version(args.expected_version)}"
        )
        return 1
    if args.print_desktop_extras:
        print(",".join(metadata_package_extras()))
        return 0
    changed = synchronize(
        check=args.check,
        package_version=package_version,
        capability_requirements=args.capability_requirements,
    )
    if args.check and changed:
        for path in changed:
            print(path.relative_to(REPO_ROOT))
        return 1
    for path in changed:
        print(path.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
