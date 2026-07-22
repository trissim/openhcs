"""Shared contract gates for the native platform bootstrap installers."""

from __future__ import annotations

import json
from pathlib import Path
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
CONTRACT_PATH = REPOSITORY_ROOT / "packaging" / "installers" / "installer_contract.json"
PYPROJECT_PATH = REPOSITORY_ROOT / "pyproject.toml"


def _contract() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_installer_contract_queries_published_project_authorities() -> None:
    contract = _contract()
    project = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))["project"]
    requirement = Requirement(contract["package_requirement"])
    python_version = Version(contract["python_version"])

    assert contract["schema_version"] == "openhcs.installer.v1"
    assert contract["product_name"] == "OpenHCS"
    assert requirement.name == project["name"]
    assert requirement.extras == {"gui"}
    assert not requirement.specifier
    assert "gui" in project["optional-dependencies"]
    assert contract["entry_point"] in project["scripts"]
    assert python_version in SpecifierSet(project["requires-python"])
    assert contract["uv_installer_urls"] == {
        "windows": "https://astral.sh/uv/install.ps1",
        "macos": "https://astral.sh/uv/install.sh",
    }


def test_release_requirement_preserves_extras_and_pins_version() -> None:
    assert release_requirement("openhcs[gui]", "0.5.22") == ("openhcs[gui]==0.5.22")
    with pytest.raises(ValueError, match="unversioned PyPI requirement"):
        release_requirement("openhcs[gui]>=0.5", "0.5.22")


def test_render_contract_changes_only_the_package_requirement(tmp_path: Path) -> None:
    source = _contract()
    output_path = tmp_path / "installer_contract.json"

    rendered = render_contract(CONTRACT_PATH, output_path, "0.5.22")
    loaded = json.loads(output_path.read_text(encoding="utf-8"))

    assert loaded == rendered
    assert loaded["package_requirement"] == "openhcs[gui]==0.5.22"
    assert {
        key: value for key, value in loaded.items() if key != "package_requirement"
    } == {key: value for key, value in source.items() if key != "package_requirement"}


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("schema_version", "openhcs.installer.v2"),
        ("product_name", "OpenHCS; rm -rf"),
        ("python_version", "python3"),
        ("package_requirement", "openhcs @ https://example.invalid/pkg.whl"),
        ("entry_point", "openhcs-gui && nope"),
        ("uv_installer_urls", {"windows": "http://example.invalid/uv.ps1"}),
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
