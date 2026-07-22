"""Focused filesystem gates for the native installer result verifier."""

from __future__ import annotations

import json
from pathlib import Path
import plistlib
import subprocess

from scripts import smoke_installed_desktop as desktop_smoke


def _write_contract(path: Path, *, entry_point: str = "openhcs") -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "openhcs.installer.v1",
                "product_name": "OpenHCS",
                "package_requirement": "openhcs[bioformats,gui,viz]==0.5.22",
                "entry_point": entry_point,
            }
        ),
        encoding="utf-8",
    )


def _stub_installed_probe(monkeypatch) -> None:
    monkeypatch.setattr(
        desktop_smoke,
        "_installed_distribution_probe",
        lambda *_args: {"version": "0.5.22"},
    )
    monkeypatch.setattr(desktop_smoke, "_smoke_entry_point", lambda *_args: None)


def test_distribution_probe_queries_installed_extra_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    contract_path = tmp_path / "installer_contract.json"
    _write_contract(contract_path)
    contract = desktop_smoke.InstallerSmokeContract.load(contract_path)
    environment = tmp_path / "environment"
    environment.mkdir()
    python_executable = environment / "python"
    python_executable.touch()
    observed: dict[str, object] = {}

    def fake_run_checked(command, *, cwd, environment=None):
        observed["command"] = command
        assert cwd == tmp_path / "environment"
        assert environment is None
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "version": "0.5.22",
                    "location": str(cwd),
                    "entry_points": [{"name": "openhcs", "value": "openhcs.cli:main"}],
                    "selected_extras": sorted(contract.package_requirement.extras),
                    "resolved_requirements": {"metadata-owned": "1.0"},
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(desktop_smoke, "_run_checked", fake_run_checked)

    result = desktop_smoke._installed_distribution_probe(
        python_executable,
        contract,
        environment,
    )

    command = observed["command"]
    assert isinstance(command, list)
    assert command[4:] == [
        "openhcs",
        "openhcs",
        *sorted(contract.package_requirement.extras),
    ]
    assert "Provides-Extra" in command[3]
    assert "installed.requires" in command[3]
    assert result["selected_extras"] == sorted(contract.package_requirement.extras)


def test_windows_smoke_resolves_generated_entry_and_launcher(
    monkeypatch,
    tmp_path: Path,
) -> None:
    contract_path = tmp_path / "installer_contract.json"
    _write_contract(contract_path)
    install_root = tmp_path / "OpenHCS Installed"
    environment = install_root / "environments" / f"env-20260722T120000Z-{'a' * 32}"
    scripts_root = environment / "Scripts"
    scripts_root.mkdir(parents=True)
    (scripts_root / "python.exe").touch()
    (scripts_root / "openhcs.exe").touch()
    launcher_path = install_root / "Launch-OpenHCS.ps1"
    launcher_path.write_text(
        '& (Join-Path $PSScriptRoot "environments\\'
        f'{environment.name}\\Scripts\\openhcs.exe")',
        encoding="utf-8",
    )
    desktop_root = tmp_path / "Desktop"
    desktop_root.mkdir()
    (desktop_root / "OpenHCS.lnk").touch()
    _stub_installed_probe(monkeypatch)

    result = desktop_smoke.smoke_installed_desktop(
        contract_path=contract_path,
        install_root=install_root,
        platform_name="windows",
        home_root=None,
        desktop_root=desktop_root,
    )

    assert result["version"] == "0.5.22"
    assert Path(result["environment"]) == environment.resolve()
    assert Path(result["launcher_path"]) == launcher_path.resolve()


def test_macos_smoke_executes_the_published_app_launcher(
    monkeypatch,
    tmp_path: Path,
) -> None:
    contract_path = tmp_path / "installer_contract.json"
    _write_contract(contract_path)
    home_root = tmp_path / "home"
    install_root = home_root / "Library" / "Application Support" / "OpenHCS"
    environment = install_root / "environments" / "20260722T120000Z-1234"
    bin_root = environment / "bin"
    bin_root.mkdir(parents=True)
    (bin_root / "python").touch()
    (bin_root / "openhcs").touch()
    (install_root / "current").symlink_to(environment)

    launcher_app = home_root / "Applications" / "OpenHCS.app"
    executable = launcher_app / "Contents" / "MacOS" / "launch-openhcs"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/bash\n", encoding="utf-8")
    executable.chmod(0o755)
    with (launcher_app / "Contents" / "Info.plist").open("wb") as stream:
        plistlib.dump({"CFBundleExecutable": executable.name}, stream)
    desktop_root = home_root / "Desktop"
    desktop_root.mkdir()
    (desktop_root / "OpenHCS.app").symlink_to(launcher_app)
    _stub_installed_probe(monkeypatch)
    launched: list[list[str]] = []

    def fake_run_checked(command, *, cwd, environment=None):
        launched.append(command)
        assert cwd == home_root
        assert environment["HOME"] == str(home_root)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="OpenHCS High-Content Screening Platform",
            stderr="",
        )

    monkeypatch.setattr(desktop_smoke, "_run_checked", fake_run_checked)

    result = desktop_smoke.smoke_installed_desktop(
        contract_path=contract_path,
        install_root=install_root,
        platform_name="macos",
        home_root=home_root,
        desktop_root=None,
    )

    assert launched == [[str(executable), "--help"]]
    assert Path(result["launcher_path"]) == launcher_app.resolve()
