#!/usr/bin/env python3
"""Render a release-pinned copy of the declarative installer contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from urllib.parse import urlparse

from packaging.requirements import Requirement
from packaging.version import Version


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = (
    REPOSITORY_ROOT / "packaging" / "installers" / "installer_contract.json"
)
SCHEMA_VERSION = "openhcs.installer.v2"


def validate_contract(contract: object) -> dict[str, object]:
    """Validate the complete cross-platform installer contract."""

    if not isinstance(contract, dict):
        raise ValueError("Installer contract must be a JSON object")
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported installer contract schema_version")

    product_name = contract.get("product_name")
    if not isinstance(product_name, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9 ._-]*", product_name
    ):
        raise ValueError("Installer contract product_name has an invalid format")

    python_version = contract.get("python_version")
    if not isinstance(python_version, str) or not re.fullmatch(
        r"3\.\d+", python_version
    ):
        raise ValueError("Installer contract python_version must be a Python 3 minor")

    package_requirement = contract.get("package_requirement")
    if not isinstance(package_requirement, str):
        raise ValueError("Installer contract package_requirement must be a string")
    requirement = Requirement(package_requirement)
    if requirement.url or requirement.marker:
        raise ValueError(
            "Installer contract package_requirement must be a PyPI requirement"
        )

    entry_point = contract.get("entry_point")
    if not isinstance(entry_point, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", entry_point
    ):
        raise ValueError("Installer contract entry_point has an invalid format")

    uv_release = contract.get("uv_release")
    if not isinstance(uv_release, dict) or set(uv_release) != {
        "version",
        "base_url",
    }:
        raise ValueError(
            "Installer contract uv_release must define version and base_url"
        )
    uv_version = uv_release["version"]
    if not isinstance(uv_version, str) or not re.fullmatch(
        r"\d+\.\d+\.\d+",
        uv_version,
    ):
        raise ValueError("Installer contract uv_release.version must be stable SemVer")
    uv_base_url = uv_release["base_url"]
    parsed_uv_base = urlparse(uv_base_url) if isinstance(uv_base_url, str) else None
    if (
        parsed_uv_base is None
        or parsed_uv_base.scheme != "https"
        or parsed_uv_base.hostname != "astral.sh"
        or parsed_uv_base.path != "/uv"
        or parsed_uv_base.params
        or parsed_uv_base.query
        or parsed_uv_base.fragment
    ):
        raise ValueError(
            "Installer contract uv_release.base_url must be the official "
            "https://astral.sh/uv endpoint"
        )
    return contract


def release_requirement(requirement_text: str, version_text: str) -> str:
    """Pin one unversioned installer requirement to a release version."""

    requirement = Requirement(requirement_text)
    if requirement.url or requirement.marker or requirement.specifier:
        raise ValueError(
            "Installer source requirement must be an unversioned PyPI requirement"
        )
    version = Version(version_text)
    extras = f"[{','.join(sorted(requirement.extras))}]" if requirement.extras else ""
    return f"{requirement.name}{extras}=={version}"


def render_contract(
    source_path: Path,
    output_path: Path,
    version_text: str,
) -> dict[str, object]:
    """Validate and write a release-pinned installer contract."""

    contract = validate_contract(json.loads(source_path.read_text(encoding="utf-8")))
    package_requirement = contract["package_requirement"]
    assert isinstance(package_requirement, str)
    contract["package_requirement"] = release_requirement(
        package_requirement,
        version_text,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(contract, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    render_contract(args.source, args.output, args.version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
