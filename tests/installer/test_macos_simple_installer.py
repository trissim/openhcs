"""Static contract tests for the native macOS installer sources."""

from __future__ import annotations

import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
INSTALLER_ROOT = REPOSITORY_ROOT / "packaging" / "installers"
MACOS_ROOT = INSTALLER_ROOT / "macos"
BOOTSTRAP_PATH = MACOS_ROOT / "install-openhcs.sh"
APPLESCRIPT_PATH = MACOS_ROOT / "Install-OpenHCS.applescript"
BUILD_PATH = MACOS_ROOT / "build-installer.sh"
CONTRACT_PATH = INSTALLER_ROOT / "installer_contract.json"


def _bootstrap() -> str:
    return BOOTSTRAP_PATH.read_text(encoding="utf-8")


def _contract() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_macos_installer_fails_closed_on_validated_shared_contract() -> None:
    source = _bootstrap()
    contract = _contract()

    assert "plutil -extract" in source
    assert "entry_point=$(contract_value entry_point)" in source
    assert "openhcs.installer.v1" in source
    assert "https://astral\\.sh/" in source

    for value in (
        contract["python_version"],
        contract["package_requirement"],
        contract["uv_installer_urls"]["macos"],
    ):
        assert value not in source


def test_macos_installer_uses_uv_without_system_python_or_admin() -> None:
    source = _bootstrap()

    assert "UV_INSTALL_DIR" in source
    assert "UV_PYTHON_INSTALL_DIR" in source
    assert "UV_NO_MODIFY_PATH=1" in source
    assert "UV_NO_CONFIG=1" in source
    assert '"$uv_executable" --no-config python install' in source
    assert '"$uv_executable" --no-config venv --python' in source
    assert '"$uv_executable" --no-config pip install --python' in source
    assert '"$uv_executable" --no-config pip check --python' in source
    assert "sudo" not in source
    assert "/usr/bin/python" not in source


def test_macos_update_switches_only_after_verification() -> None:
    source = _bootstrap()

    verify_position = source.index("pip check --python")
    entry_position = source.index('if [[ ! -x "$installed_entry" ]]')
    state_switch_position = source.index('mv -fh "$current_candidate"')

    assert verify_position < entry_position < state_switch_position
    assert "new_environment" in source
    assert "trap cleanup EXIT HUP INT TERM" in source
    assert "OPENHCS_CPU_ONLY=true" in source
    assert 'ln -s "$new_environment" "$current_candidate"' in source
    assert 'mv -fh "$current_candidate" "$current_environment"' in source
    assert 'readlink "$current_environment"' in source


def test_macos_installer_builds_a_native_app_with_embedded_contract() -> None:
    applescript = APPLESCRIPT_PATH.read_text(encoding="utf-8")
    build = BUILD_PATH.read_text(encoding="utf-8")

    assert "display dialog" in applescript
    assert "progress description" in applescript
    assert "do shell script quoted form of bootstrapPath" in applescript
    assert "administrator privileges" not in applescript
    assert "osacompile" in build
    assert "Contents/Resources/installer_contract.json" in build
    assert "Contents/Resources/install-openhcs.sh" in build
