"""Shared contract gates for the native platform bootstrap installers."""

from __future__ import annotations

import json
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from openhcs.desktop_installation import (
    DESKTOP_INSTALL_PROFILE,
    DesktopInstallerContract,
    DesktopUvRelease,
)
from scripts.render_installer_contract import render_contract

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = REPOSITORY_ROOT / "pyproject.toml"
INSTALLER_WORKFLOW_PATHS = (
    REPOSITORY_ROOT / ".github" / "workflows" / "integration-tests.yml",
    REPOSITORY_ROOT / ".github" / "workflows" / "publish.yml",
)


def _rendered_contract(
    tmp_path: Path,
    version: str = "0.5.22",
) -> DesktopInstallerContract:
    return render_contract(tmp_path / "installer_contract.json", version)


def test_installer_profile_owns_only_native_install_policy(tmp_path: Path) -> None:
    profile = DESKTOP_INSTALL_PROFILE
    contract = _rendered_contract(tmp_path)
    project = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))["project"]
    requirement = Requirement(contract.package_requirement)
    python_version = Version(profile.python_version)

    assert contract.schema_version == "openhcs.installer.v2"
    assert contract.product_name == "OpenHCS"
    assert requirement.name == project["name"]
    assert requirement.extras == {extra.value for extra in profile.package_extras}
    assert requirement.specifier == SpecifierSet("==0.5.22")
    assert requirement.extras <= project["optional-dependencies"].keys()
    assert contract.binary_only_packages == ",".join(
        package.value for package in profile.binary_only_packages
    )
    assert contract.entry_point == project["name"]
    assert contract.entry_point in project["scripts"]
    assert contract.gui_entry_point in project["gui-scripts"]
    assert (
        project["gui-scripts"][contract.gui_entry_point]
        == "openhcs.pyqt_gui.__main__:main"
    )
    assert python_version in SpecifierSet(project["requires-python"])
    assert contract.uv_release == profile.uv_release


def test_visualization_install_surface_composes_napari_and_fiji() -> None:
    project = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))["project"]
    extras = project["optional-dependencies"]
    parsed_napari_requirements = [
        Requirement(requirement) for requirement in extras["napari"]
    ]
    napari_requirements = {
        requirement.name: requirement for requirement in parsed_napari_requirements
    }
    fiji_requirements = {
        Requirement(requirement).name for requirement in extras["fiji"]
    }

    assert {"napari", "napari-crop"} <= set(napari_requirements)
    napari_requirement = napari_requirements["napari"]
    assert napari_requirement.specifier.contains("0.7.1")
    assert not napari_requirement.specifier.contains("0.6.1")
    assert {
        "napari-roi-manager",
        "openhcs-napari-roi-manager",
    }.isdisjoint(napari_requirements)
    for combined_extra in ("viz", "all"):
        combined_requirements = {
            Requirement(requirement).name for requirement in extras[combined_extra]
        }
        assert set(napari_requirements) | fiji_requirements <= combined_requirements
        assert {
            "napari-roi-manager",
            "openhcs-napari-roi-manager",
        }.isdisjoint(combined_requirements)


def test_core_opencv_dependency_uses_the_shared_headless_runtime() -> None:
    project = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))["project"]
    requirement_names = {
        Requirement(requirement).name for requirement in project["dependencies"]
    }

    assert "opencv-python-headless" in requirement_names
    assert "opencv-python" not in requirement_names


def test_intel_macos_dependencies_select_wheel_backed_release_lines() -> None:
    project = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))["project"]
    intel_environment = {
        "platform_system": "Darwin",
        "platform_machine": "x86_64",
    }
    active_requirements = [
        requirement
        for requirement_text in project["dependencies"]
        if (requirement := Requirement(requirement_text)).marker is None
        or requirement.marker.evaluate(intel_environment)
    ]
    requirements = {
        requirement.name: requirement for requirement in active_requirements
    }

    assert requirements["numba"].specifier.contains("0.62.1")
    assert not requirements["numba"].specifier.contains("0.63")
    assert requirements["opencv-python-headless"].specifier.contains("4.10.0.84")
    assert not requirements["opencv-python-headless"].specifier.contains("4.11.0.86")


def test_pyimagej_install_surfaces_require_managed_java_constraint_api() -> None:
    project = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))["project"]
    extras = project["optional-dependencies"]

    for extra_name in ("fiji", "bioformats", "viz", "all"):
        parsed_requirements = [
            Requirement(requirement_text) for requirement_text in extras[extra_name]
        ]
        requirements = {
            requirement.name: requirement for requirement in parsed_requirements
        }
        scyjava = requirements["scyjava"]
        assert scyjava.specifier.contains("1.12.0")
        assert not scyjava.specifier.contains("1.11.0")


def test_install_profile_projects_release_requirement() -> None:
    selection = DESKTOP_INSTALL_PROFILE.select("openhcs", "0.5.22")

    assert selection.package_requirement == (
        "openhcs[bioformats,cellprofiler-compat,gui,mcp,viz]==0.5.22"
    )
    assert selection.binary_only_argument == (
        "llvmlite,numba,opencv-python,opencv-python-headless"
    )


@pytest.mark.parametrize(
    "package_name",
    [
        "openhcs[gui]>=0.5",
        "openhcs @ https://example.invalid/openhcs.whl",
    ],
)
def test_install_profile_rejects_non_registry_package_identity(
    package_name: str,
) -> None:
    with pytest.raises(ValueError, match="must be unversioned"):
        DESKTOP_INSTALL_PROFILE.select(package_name, "0.5.22")


def test_render_contract_writes_declaration_derived_projection(tmp_path: Path) -> None:
    output_path = tmp_path / "installer_contract.json"

    rendered = render_contract(output_path, "0.5.22")
    loaded = json.loads(output_path.read_text(encoding="utf-8"))

    assert loaded["schema_version"] == rendered.schema_version
    assert loaded["product_name"] == rendered.product_name
    assert loaded["entry_point"] == rendered.entry_point
    assert loaded["gui_entry_point"] == rendered.gui_entry_point
    assert loaded["uv_release"] == {
        "version": rendered.uv_release.version,
        "base_url": rendered.uv_release.base_url,
    }
    assert loaded["package_requirement"] == (
        "openhcs[bioformats,cellprofiler-compat,gui,mcp,viz]==0.5.22"
    )


def test_clean_checkout_workflows_execute_renderer_through_package_namespace() -> None:
    for workflow_path in INSTALLER_WORKFLOW_PATHS:
        source = workflow_path.read_text(encoding="utf-8")

        assert "python -m scripts.render_installer_contract" in source
        assert "python scripts/render_installer_contract.py" not in source


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("python_version", "python3"),
        ("package_extras", ("gui", "gui")),
        ("binary_only_packages", ("not safe",)),
    ],
)
def test_installer_profile_rejects_malformed_policy(
    field: str,
    bad_value: object,
) -> None:
    with pytest.raises(ValueError):
        replace(DESKTOP_INSTALL_PROFILE, **{field: bad_value})


@pytest.mark.parametrize(
    ("version", "base_url"),
    [
        ("latest", "https://astral.sh/uv"),
        ("0.11.28", "http://example.invalid/uv"),
    ],
)
def test_installer_profile_rejects_malformed_uv_release(
    version: str,
    base_url: str,
) -> None:
    with pytest.raises(ValueError):
        DesktopUvRelease(version=version, base_url=base_url)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("product_name", "OpenHCS; rm -rf"),
        ("entry_point", "openhcs-gui && nope"),
        ("gui_entry_point", "openhcs-gui && nope"),
    ],
)
def test_installer_contract_rejects_malformed_command_data(
    tmp_path: Path,
    field: str,
    bad_value: object,
) -> None:
    with pytest.raises(ValueError):
        replace(_rendered_contract(tmp_path), **{field: bad_value})
