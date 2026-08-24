#!/usr/bin/env python3
"""Render a release-pinned native contract from owning declarations."""

from __future__ import annotations

import argparse
import ast
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from packaging.utils import canonicalize_name
from packaging.version import Version

from openhcs.desktop_installation import (
    DESKTOP_INSTALL_PROFILE,
    DesktopInstallerContract,
    DesktopPackageExtra,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_DECLARATION = REPOSITORY_ROOT / "pyproject.toml"
BRAND_DECLARATION = REPOSITORY_ROOT / "openhcs" / "resources" / "brand.py"


@dataclass(frozen=True, slots=True)
class BrandProductDeclaration:
    """Product identity read from the lightweight brand authority."""

    assignment_name: ClassVar[str] = "BRAND_PRODUCT_NAME"

    product_name: str

    @classmethod
    def from_source(cls, path: Path) -> BrandProductDeclaration:
        """Read the one literal product-name declaration without runtime import."""

        assignments = tuple(
            node
            for node in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == cls.assignment_name
                for target in node.targets
            )
        )
        if len(assignments) != 1:
            raise ValueError(
                f"Expected exactly one {cls.assignment_name} declaration in {path}."
            )
        value = ast.literal_eval(assignments[0].value)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"{cls.assignment_name} must be a non-empty string declaration."
            )
        return cls(product_name=value)


@dataclass(frozen=True, slots=True)
class ProjectInstallDeclarations:
    """Package and entry-point identities derived from project metadata."""

    package_name: str
    entry_point: str
    gui_entry_point: str

    @classmethod
    def from_source(
        cls,
        path: Path,
        package_extras: tuple[DesktopPackageExtra, ...],
    ) -> ProjectInstallDeclarations:
        """Resolve the one install surface declared by project metadata."""

        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        project = payload.get("project")
        if not isinstance(project, dict):
            raise TypeError("Project metadata must define a project table.")
        package_name = project.get("name")
        scripts = project.get("scripts")
        gui_scripts = project.get("gui-scripts")
        optional_dependencies = project.get("optional-dependencies")
        if (
            not isinstance(package_name, str)
            or not isinstance(scripts, dict)
            or not isinstance(gui_scripts, dict)
            or not isinstance(optional_dependencies, dict)
        ):
            raise TypeError("Project installer declarations are incomplete.")
        command_entry_points = tuple(
            name
            for name in scripts
            if isinstance(name, str)
            and canonicalize_name(name) == canonicalize_name(package_name)
        )
        gui_entry_points = tuple(name for name in gui_scripts if isinstance(name, str))
        if len(command_entry_points) != 1 or len(gui_entry_points) != 1:
            raise ValueError(
                "Project metadata must declare one primary command and GUI entry "
                "point."
            )
        missing_extras = tuple(
            extra.value
            for extra in package_extras
            if extra.value not in optional_dependencies
        )
        if missing_extras:
            raise ValueError(
                "Desktop install profile selects unknown project extras: "
                + ", ".join(missing_extras)
            )
        return cls(
            package_name=package_name,
            entry_point=command_entry_points[0],
            gui_entry_point=gui_entry_points[0],
        )


def render_contract(
    output_path: Path,
    version_text: str,
    *,
    project_path: Path = PROJECT_DECLARATION,
    brand_path: Path = BRAND_DECLARATION,
) -> DesktopInstallerContract:
    """Render one exact-version contract from its owning declarations."""

    project = ProjectInstallDeclarations.from_source(
        project_path,
        DESKTOP_INSTALL_PROFILE.package_extras,
    )
    brand = BrandProductDeclaration.from_source(brand_path)
    contract = DESKTOP_INSTALL_PROFILE.project_contract(
        product_name=brand.product_name,
        package_name=project.package_name,
        version=Version(version_text),
        entry_point=project.entry_point,
        gui_entry_point=project.gui_entry_point,
    )
    contract.write(output_path)
    return contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    render_contract(arguments.output, arguments.version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
