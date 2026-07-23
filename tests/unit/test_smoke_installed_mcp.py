"""Ownership checks for the installed MCP wheel smoke test."""

from pathlib import Path

import pytest

from scripts.smoke_installed_mcp import assert_not_source_checkout_import


def test_smoke_ownership_rejects_source_package_and_root(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"

    with pytest.raises(AssertionError, match="source checkout instead of the wheel"):
        assert_not_source_checkout_import(
            package_path=checkout / "openhcs" / "__init__.py",
            knowledge_root=checkout / "openhcs" / "agent" / "knowledge_base",
            forbidden_root=checkout,
        )

    with pytest.raises(AssertionError, match="Knowledge root resolved"):
        assert_not_source_checkout_import(
            package_path=tmp_path / "site-packages" / "openhcs" / "__init__.py",
            knowledge_root=checkout,
            forbidden_root=checkout,
        )


def test_smoke_ownership_allows_wheel_venv_inside_checkout(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    site_packages = checkout / "test_gui" / "lib" / "python3.12" / "site-packages"

    assert_not_source_checkout_import(
        package_path=site_packages / "openhcs" / "__init__.py",
        knowledge_root=site_packages / "openhcs" / "agent" / "knowledge_base",
        forbidden_root=checkout,
    )
