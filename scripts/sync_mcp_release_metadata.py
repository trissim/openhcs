#!/usr/bin/env python3
"""Synchronize client-distribution metadata with ``openhcs.__version__``.

The Python package version is the authority. This script projects it into the
Codex plugin, Claude MCPB, and official MCP Registry metadata without importing
OpenHCS or maintaining a second hand-edited version table.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

from packaging.version import Version


REPO_ROOT = Path(__file__).resolve().parents[1]
INIT_PATH = REPO_ROOT / "openhcs" / "__init__.py"
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


def _json_text(payload: dict) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def _replace_exactly_once(text: str, pattern: str, replacement: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(
            f"Expected one metadata field matching {pattern!r}, found {count}"
        )
    return updated


def projected_files(package_version: Version) -> dict[Path, str]:
    """Return the complete version projection without writing it."""
    pep440 = str(package_version)
    semver = plugin_semver(package_version)

    plugin_manifest = json.loads(PLUGIN_MANIFEST_PATH.read_text(encoding="utf-8"))
    plugin_manifest["version"] = semver

    plugin_mcp = json.loads(PLUGIN_MCP_PATH.read_text(encoding="utf-8"))
    plugin_args = plugin_mcp["mcpServers"]["openhcs"]["args"]
    plugin_args[
        plugin_args.index(
            next(arg for arg in plugin_args if arg.startswith("openhcs["))
        )
    ] = f"openhcs[gui,mcp]=={pep440}"

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
        r'^    "openhcs\[gui,mcp\]==[^"]+",$',
        f'    "openhcs[gui,mcp]=={pep440}",',
    )

    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry["version"] = pep440
    package = registry["packages"][0]
    package["version"] = pep440
    for argument in package.get("runtimeArguments", []):
        if argument.get("name") == "--with":
            argument["value"] = f"openhcs[gui,mcp]=={pep440}"

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
) -> tuple[Path, ...]:
    """Write projections, or return drift and leave files untouched in check mode."""
    changed: list[Path] = []
    resolved_version = package_version or read_package_version()
    for path, projected in projected_files(resolved_version).items():
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
    changed = synchronize(check=args.check, package_version=package_version)
    if args.check and changed:
        for path in changed:
            print(path.relative_to(REPO_ROOT))
        return 1
    for path in changed:
        print(path.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
