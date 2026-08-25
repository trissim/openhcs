"""Ownership checks for the installed GUI startup smoke test."""

from pathlib import Path

import pytest

from scripts.smoke_installed_gui import assert_not_source_checkout_import


def test_gui_smoke_rejects_source_package(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"

    with pytest.raises(AssertionError, match="source checkout instead of the wheel"):
        assert_not_source_checkout_import(
            package_path=checkout / "openhcs" / "__init__.py",
            forbidden_root=checkout,
        )


def test_gui_smoke_allows_wheel_venv_inside_checkout(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    site_packages = checkout / "test_gui" / "lib" / "site-packages"

    assert_not_source_checkout_import(
        package_path=site_packages / "openhcs" / "__init__.py",
        forbidden_root=checkout,
    )
