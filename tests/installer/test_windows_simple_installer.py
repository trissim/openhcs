"""Static contract tests for the native Windows installer sources."""

from __future__ import annotations

import json
from pathlib import Path
import re

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
INSTALLER_ROOT = REPOSITORY_ROOT / "packaging" / "installers"
WINDOWS_ROOT = INSTALLER_ROOT / "windows"
POWERSHELL_PATH = WINDOWS_ROOT / "Install-OpenHCS.ps1"
CMD_PATH = WINDOWS_ROOT / "Install-OpenHCS.cmd"
CONTRACT_PATH = INSTALLER_ROOT / "installer_contract.json"
INTEGRATION_WORKFLOW_PATH = (
    REPOSITORY_ROOT / ".github" / "workflows" / "integration-tests.yml"
)


def _source() -> str:
    return POWERSHELL_PATH.read_text(encoding="utf-8")


def _contract() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_windows_installer_has_stable_double_click_entrypoint() -> None:
    cmd = CMD_PATH.read_text(encoding="utf-8")

    assert "powershell.exe -NoProfile -WindowStyle Hidden" in cmd
    assert '-ExecutionPolicy Bypass -File "%~dp0Install-OpenHCS.ps1"' in cmd
    assert 'start ""' in cmd


def test_windows_installer_fails_closed_on_validated_shared_contract() -> None:
    source = _source()
    contract = _contract()

    assert "Resolve-ContractPath" in source
    assert '"installer_contract.json"' in source
    assert "ConvertFrom-Json" in source
    assert 'Get-RequiredTextProperty $contract "entry_point"' in source
    assert '"openhcs.installer.v1"' in source
    assert "Expected exactly one installer_contract.json" in source
    assert "Uri]::TryCreate" in source
    assert "UriSchemeHttps" in source
    assert "$parsedUrl.IdnHost," in source
    assert '"astral.sh"' in source
    assert '"^3\\.[0-9]+$"' in source

    # Shared semantic values are data, never fallback constants in the script.
    for value in (
        contract["python_version"],
        contract["package_requirement"],
        contract["uv_installer_urls"]["windows"],
    ):
        assert value not in source


def test_windows_installer_uses_uv_as_the_environment_owner() -> None:
    source = _source()

    assert "Invoke-WebRequest" in source
    assert "GetTempPath" in source
    assert "openhcs-uv-installer-$([Guid]::NewGuid()" in source
    assert ".ps1" in source
    assert "-OutFile $temporaryUvInstaller" in source
    assert "-TimeoutSec 120" in source
    assert "Invoke-Expression" not in source
    assert re.search(r'"--no-config", "python", "install"', source)
    assert re.search(r'"--no-config", "venv", "--python"', source)
    assert '"venv", "--clear"' not in source
    assert re.search(r'"--no-config", "pip", "install", "--python"', source)
    assert re.search(r'"--no-config", "pip", "check", "--python"', source)
    assert "$env:UV_INSTALL_DIR" in source
    assert "$env:UV_NO_MODIFY_PATH" in source

    # Contract values remain individual native arguments even when paths contain spaces.
    assert "[string[]]$ArgumentList" in source
    assert "& $FilePath @ArgumentList" in source
    assert '$ErrorActionPreference = "Continue"' in source
    assert "$exitCode = $LASTEXITCODE" in source
    assert source.index('$ErrorActionPreference = "Continue"') < source.index(
        "$exitCode = $LASTEXITCODE"
    )
    assert source.index("$exitCode = $LASTEXITCODE") < source.index(
        "$ErrorActionPreference = $previousErrorActionPreference"
    )
    assert "cmd.exe" not in source
    assert "/c " not in source.lower()


def test_windows_installer_delegates_runtime_to_declared_entrypoint() -> None:
    source = _source()

    assert '"Scripts"' in source
    assert '"$($Contract.EntryPoint).exe"' in source
    assert "WScript.Shell" in source
    assert "CreateShortcut" in source
    assert '$env:OPENHCS_CPU_ONLY = "true"' in source
    assert '"environments"' in source
    assert "Publish-LaunchAdapterAndShortcut" in source
    assert "launcherCandidate" in source
    assert "launcherBackup" in source
    assert "[IO.File]::Replace" in source
    assert "Remove-SupersededEnvironments" in source
    assert source.index('"pip", "check"') < source.index(
        "Publish-LaunchAdapterAndShortcut `"
    )
    assert "openhcs.pyqt_gui" not in source
    assert "python -m" not in source


def test_windows_installer_keeps_ui_responsive_and_failures_visible() -> None:
    source = _source()
    write_log = source[
        source.index("function Write-InstallLog") : source.index(
            "function Invoke-LoggedCommand"
        )
    ]

    assert "System.Windows.Forms" in source
    assert "Start-InstallerWorker" in source
    assert "-EncodedCommand" in source
    assert '"taskkill.exe"' in source
    assert '"/T"' in source
    assert "Cancel install" in source
    assert "installer.log" in source
    assert "bootstrap.log" in source
    assert "if ($Worker)" in write_log
    assert "Write-Host $line" in write_log
    assert "Installation failed. Review the durable log" in source


def test_windows_installer_ci_has_an_absolute_safety_ceiling() -> None:
    workflow = INTEGRATION_WORKFLOW_PATH.read_text(encoding="utf-8")
    smoke_step = workflow[
        workflow.index(
            "      - name: Execute and verify Windows installer"
        ) : workflow.index("      - name: Show Windows installer log on failure")
    ]

    assert "        timeout-minutes: 20" in smoke_step
    assert "Install-OpenHCS.ps1" in smoke_step
    assert '$env:QT_OPENGL = "software"' in smoke_step
    assert smoke_step.index('$env:QT_OPENGL = "software"') < smoke_step.index(
        "python -m scripts.smoke_installed_desktop"
    )


def test_windows_installer_ci_surfaces_detached_viewer_logs_on_failure() -> None:
    workflow = INTEGRATION_WORKFLOW_PATH.read_text(encoding="utf-8")
    failure_step = workflow[
        workflow.index(
            "      - name: Show Windows installer log on failure"
        ) : workflow.index("      - name: Validate macOS installer sources")
    ]

    assert 'Join-Path $HOME ".local\\share\\openhcs\\logs"' in failure_step
    assert 'Filter "*_detached_port_*.log"' in failure_step
    assert "Get-Content -LiteralPath $_.FullName" in failure_step
