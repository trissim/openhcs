"""Shared contract gates for the native platform bootstrap installers."""

from __future__ import annotations

import json
from pathlib import Path
import re
import tomllib

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version
import pytest

from scripts.render_installer_contract import (
    release_requirement,
    render_contract,
    validate_contract,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPOSITORY_ROOT / "openhcs" / "resources" / "installer_contract.json"
PYPROJECT_PATH = REPOSITORY_ROOT / "pyproject.toml"


def _contract() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_installer_contract_queries_published_project_authorities() -> None:
    contract = _contract()
    project = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))["project"]
    requirement = Requirement(contract["package_requirement"])
    python_version = Version(contract["python_version"])

    assert contract["schema_version"] == "openhcs.installer.v2"
    assert contract["product_name"] == "OpenHCS"
    assert requirement.name == project["name"]
    assert requirement.extras == {
        "bioformats",
        "cellprofiler-compat",
        "gui",
        "mcp",
        "viz",
    }
    assert not requirement.specifier
    assert requirement.extras <= project["optional-dependencies"].keys()
    assert contract["entry_point"] == project["name"]
    assert contract["entry_point"] in project["scripts"]
    assert contract["gui_entry_point"] in project["gui-scripts"]
    assert (
        project["gui-scripts"][contract["gui_entry_point"]]
        == "openhcs.pyqt_gui.__main__:main"
    )
    assert python_version in SpecifierSet(project["requires-python"])
    uv_release = contract["uv_release"]
    assert set(uv_release) == {"version", "base_url"}
    assert re.fullmatch(r"\d+\.\d+\.\d+", uv_release["version"])
    assert uv_release["base_url"] == "https://astral.sh/uv"


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


def test_release_requirement_preserves_extras_and_pins_version() -> None:
    assert release_requirement(
        "openhcs[gui,viz,bioformats,mcp,cellprofiler-compat]", "0.5.22"
    ) == ("openhcs[bioformats,cellprofiler-compat,gui,mcp,viz]==0.5.22")
    with pytest.raises(ValueError, match="unversioned PyPI requirement"):
        release_requirement("openhcs[gui]>=0.5", "0.5.22")


def test_render_contract_changes_only_the_package_requirement(tmp_path: Path) -> None:
    source = _contract()
    output_path = tmp_path / "installer_contract.json"

    rendered = render_contract(CONTRACT_PATH, output_path, "0.5.22")
    loaded = json.loads(output_path.read_text(encoding="utf-8"))

    assert loaded == rendered
    assert loaded["package_requirement"] == (
        "openhcs[bioformats,cellprofiler-compat,gui,mcp,viz]==0.5.22"
    )
    assert {
        key: value for key, value in loaded.items() if key != "package_requirement"
    } == {key: value for key, value in source.items() if key != "package_requirement"}


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("schema_version", "openhcs.installer.v1"),
        ("product_name", "OpenHCS; rm -rf"),
        ("python_version", "python3"),
        ("package_requirement", "openhcs @ https://example.invalid/pkg.whl"),
        ("entry_point", "openhcs-gui && nope"),
        ("gui_entry_point", "openhcs-gui && nope"),
        (
            "uv_release",
            {"version": "latest", "base_url": "http://example.invalid/uv"},
        ),
    ],
)
def test_installer_contract_rejects_malformed_command_data(
    field: str,
    bad_value: object,
) -> None:
    contract = _contract()
    contract[field] = bad_value

    with pytest.raises(ValueError):
        validate_contract(contract)
