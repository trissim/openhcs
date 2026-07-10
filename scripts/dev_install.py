#!/usr/bin/env python3
"""Install OpenHCS and its pinned submodules as editable projects.

Usage:
    python scripts/dev_install.py [--extras EXTRAS] [--no-deps]

Options:
    --extras EXTRAS    Comma-separated list of extras to install (e.g., "dev,gui,gpu")
                       Default: "dev,gui"
    --no-deps          Register editable projects without resolving third-party deps
"""

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

from external_dependencies import PROJECT_ROOT, validated_external_projects


def run_command(cmd, check=True):
    """Run a shell command and print output."""
    print(f"\n{'=' * 60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd, check=check, cwd=PROJECT_ROOT)
    return result


def build_install_command(extras: str, no_deps: bool) -> list[str]:
    """Build one resolver transaction containing every local editable."""
    projects = validated_external_projects()
    command = [sys.executable, "-m", "pip", "install"]
    if no_deps:
        command.append("--no-deps")
    for project in projects:
        command.extend(["-e", str(project.path)])

    extras_spec = f"[{extras}]" if extras else ""
    command.extend(["-e", f"{PROJECT_ROOT}{extras_spec}"])
    return command


def _editable_source(distribution_name: str) -> Path:
    # A source-tree *.egg-info can precede the venv's *.dist-info on sys.path.
    # Inspect every matching record and select the actual editable installation.
    found = False
    for distribution in importlib.metadata.distributions(name=distribution_name):
        found = True
        direct_url_text = distribution.read_text("direct_url.json")
        if not direct_url_text:
            continue

        direct_url = json.loads(direct_url_text)
        if not direct_url.get("dir_info", {}).get("editable"):
            continue

        parsed = urlparse(direct_url["url"])
        if parsed.scheme != "file":
            continue
        return Path(unquote(parsed.path)).resolve()

    if not found:
        raise importlib.metadata.PackageNotFoundError(distribution_name)
    raise RuntimeError(f"{distribution_name} has no local editable-install metadata")


def verify_editable_sources() -> None:
    """Verify every distribution points at this worktree's pinned checkout."""
    expected_sources = {
        project.name: project.path.resolve()
        for project in validated_external_projects()
    }
    expected_sources["openhcs"] = PROJECT_ROOT.resolve()

    errors: list[str] = []
    for name, expected_path in expected_sources.items():
        try:
            actual_path = _editable_source(name)
        except (importlib.metadata.PackageNotFoundError, RuntimeError) as exc:
            errors.append(str(exc))
            continue
        if actual_path != expected_path:
            errors.append(f"{name}: expected {expected_path}, found {actual_path}")

    if errors:
        raise RuntimeError(
            "Editable installation verification failed:\n- " + "\n- ".join(errors)
        )


def main():
    parser = argparse.ArgumentParser(description="Install openhcs in development mode")
    parser.add_argument(
        "--extras",
        default="dev,gui",
        help="Comma-separated list of extras to install (default: dev,gui)",
    )
    parser.add_argument(
        "--no-deps",
        action="store_true",
        help="Register editable projects without resolving third-party dependencies",
    )
    args = parser.parse_args()

    projects = validated_external_projects()
    print("Validated main dependency pins:")
    for project in projects:
        print(f"  {project.name}=={project.version} <- {project.relative_path}")

    run_command(build_install_command(args.extras, args.no_deps))
    verify_editable_sources()

    print("\n" + "=" * 60)
    print("Development installation complete!")
    print("=" * 60)
    print("\nOpenHCS and all pinned external modules are installed editable")
    print("from this worktree.")
    print("\nYou can now run openhcs with:")
    print("  openhcs")
    print("\nOr run tests with:")
    print("  pytest")


if __name__ == "__main__":
    main()
