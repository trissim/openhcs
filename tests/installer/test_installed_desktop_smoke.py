"""Focused filesystem gates for the native installer result verifier."""

from __future__ import annotations

import json
from pathlib import Path
import plistlib
import subprocess
import sys

import pytest

from scripts import smoke_installed_desktop as desktop_smoke


def _write_contract(path: Path, *, entry_point: str = "openhcs") -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "openhcs.installer.v2",
                "product_name": "OpenHCS",
                "package_requirement": (
                    "openhcs[bioformats,cellprofiler-compat,gui,mcp,viz]==0.5.22"
                ),
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
    monkeypatch.setattr(
        desktop_smoke,
        "_smoke_installed_mcp",
        lambda *_args: {"health_status": "ok"},
    )
    monkeypatch.setattr(
        desktop_smoke,
        "_smoke_installed_demo",
        lambda *_args, **_kwargs: {
            "execution_status": "complete",
            "viewer_observed": True,
            "viewer_type": "napari",
        },
    )
    monkeypatch.setattr(
        desktop_smoke,
        "_smoke_installed_napari",
        lambda *_args: {
            "viewer_type": "napari",
            "qt_platform": "cocoa",
            "layer_count": 1,
            "nonzero_count": 64,
            "closed": True,
        },
    )


def test_checked_command_decodes_child_diagnostics_independently_of_host_locale(
    tmp_path: Path,
) -> None:
    command = [
        sys.executable,
        "-c",
        "import sys; sys.stderr.buffer.write(b'\\x8d'); raise SystemExit(1)",
    ]

    with pytest.raises(AssertionError) as exc_info:
        desktop_smoke._run_checked(command, cwd=tmp_path)

    assert "Command failed with exit code 1" in str(exc_info.value)
    assert "\ufffd" in str(exc_info.value)


def test_checked_command_can_stream_stderr_while_retaining_json_stdout(
    tmp_path: Path,
    capfd,
) -> None:
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "print('live phase', file=sys.stderr, flush=True); "
            'print(\'{"status": "complete"}\')'
        ),
    ]

    completed = desktop_smoke._run_checked(
        command,
        cwd=tmp_path,
        stream_stderr=True,
    )

    captured = capfd.readouterr()
    assert json.loads(completed.stdout) == {"status": "complete"}
    assert completed.stderr is None
    assert captured.err.splitlines() == ["live phase"]


def test_mcp_smoke_uses_installed_python_in_isolated_mode(
    monkeypatch,
    tmp_path: Path,
) -> None:
    installed_python = tmp_path / "installed" / "python"
    installed_python.parent.mkdir()
    installed_python.touch()
    observed: dict[str, object] = {}

    def fake_run_checked(command, *, cwd, environment=None):
        observed.update(command=command, cwd=cwd, environment=environment)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"health_status": "ok"}),
            stderr="",
        )

    monkeypatch.setattr(desktop_smoke, "_run_checked", fake_run_checked)

    payload = desktop_smoke._smoke_installed_mcp(installed_python, tmp_path)

    assert payload == {"health_status": "ok"}
    assert observed["command"] == [
        str(installed_python),
        "-I",
        str(desktop_smoke.INSTALLED_MCP_SMOKE_PATH.resolve()),
        "--forbid-import-root",
        str(desktop_smoke.REPOSITORY_ROOT),
    ]
    assert observed["cwd"] == tmp_path
    smoke_environment = observed["environment"]
    assert isinstance(smoke_environment, dict)
    assert smoke_environment["OPENHCS_CPU_ONLY"] == "true"
    assert smoke_environment["XDG_CACHE_HOME"] == str(
        (tmp_path / "mcp-cache").resolve()
    )
    assert smoke_environment["NUMBA_CACHE_DIR"] == str(
        (tmp_path / "cache" / "numba").resolve()
    )


def test_portable_demo_uses_installed_python_and_real_viewer_contract(
    monkeypatch,
    tmp_path: Path,
) -> None:
    installed_python = tmp_path / "installed" / "python"
    installed_python.parent.mkdir()
    installed_python.touch()
    observed: dict[str, object] = {}

    def fake_run_checked(
        command,
        *,
        cwd,
        environment=None,
        timeout_seconds=120,
        stream_stderr=False,
    ):
        observed.update(
            command=command,
            cwd=cwd,
            environment=environment,
            timeout_seconds=timeout_seconds,
            stream_stderr=stream_stderr,
        )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "execution_status": "complete",
                    "viewer_observed": True,
                    "viewer_type": "napari",
                    "viewer_layer_count": 2,
                    "viewer_nonzero_payload_count": 2,
                }
            ),
            stderr="",
        )

    demo_root = (tmp_path / "installer-smoke-demo").resolve()
    demo_root.mkdir()
    stale_file = demo_root / "stale-output.csv"
    stale_file.write_text("stale", encoding="utf-8")
    monkeypatch.setattr(desktop_smoke, "_run_checked", fake_run_checked)

    payload = desktop_smoke._smoke_installed_demo(
        installed_python,
        tmp_path,
        viewer=True,
    )

    assert payload["viewer_type"] == "napari"
    assert not stale_file.exists()
    assert observed["command"] == [
        str(installed_python),
        "-I",
        "-m",
        "openhcs.mcp.installed_demo",
        "--output-root",
        str(demo_root),
        "--forbid-import-root",
        str(desktop_smoke.REPOSITORY_ROOT),
        "--json",
    ]
    assert observed["cwd"] == tmp_path
    assert observed["timeout_seconds"] is None
    assert observed["stream_stderr"] is True
    demo_environment = observed["environment"]
    assert isinstance(demo_environment, dict)
    assert demo_environment["OPENHCS_AGENT_READ_ROOTS"] == str(demo_root)
    assert demo_environment["OPENHCS_AGENT_WRITE_ROOTS"] == str(demo_root)
    assert demo_environment["NUMBA_CACHE_DIR"] == str(
        (tmp_path / "cache" / "numba").resolve()
    )


def test_portable_demo_headless_mode_preserves_runtime_contract(
    monkeypatch,
    tmp_path: Path,
) -> None:
    installed_python = tmp_path / "installed" / "python"
    installed_python.parent.mkdir()
    installed_python.touch()
    observed: dict[str, object] = {}

    def fake_run_checked(
        command,
        *,
        cwd,
        environment=None,
        timeout_seconds=120,
        stream_stderr=False,
    ):
        observed.update(
            command=command,
            cwd=cwd,
            environment=environment,
            timeout_seconds=timeout_seconds,
            stream_stderr=stream_stderr,
        )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "execution_status": "complete",
                    "viewer_observed": False,
                    "viewer_type": None,
                    "viewer_port": None,
                    "viewer_layer_count": 0,
                    "viewer_nonzero_payload_count": 0,
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(desktop_smoke, "_run_checked", fake_run_checked)

    payload = desktop_smoke._smoke_installed_demo(
        installed_python,
        tmp_path,
        viewer=False,
    )

    assert payload["execution_status"] == "complete"
    assert "--no-viewer" in observed["command"]
    assert observed["command"][-1] == "--json"
    assert observed["timeout_seconds"] is None
    assert observed["stream_stderr"] is True


def test_native_napari_smoke_uses_installed_python_without_offscreen_qt(
    monkeypatch,
    tmp_path: Path,
) -> None:
    installed_python = tmp_path / "installed" / "python"
    installed_python.parent.mkdir()
    installed_python.touch()
    observed: dict[str, object] = {}
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    def fake_run_checked(command, *, cwd, environment=None):
        observed.update(command=command, cwd=cwd, environment=environment)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "viewer_type": "napari",
                    "qt_platform": "cocoa",
                    "layer_count": 1,
                    "layer_name": "OpenHCS installer smoke",
                    "shape": [8, 8],
                    "nonzero_count": 64,
                    "closed": True,
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(desktop_smoke, "_run_checked", fake_run_checked)

    payload = desktop_smoke._smoke_installed_napari(installed_python, tmp_path)

    command = observed["command"]
    assert command[:3] == [str(installed_python), "-I", "-c"]
    assert "napari.Viewer(show=False)" in command[3]
    assert "viewer.add_image" in command[3]
    assert "np.count_nonzero" in command[3]
    assert "viewer.close()" in command[3]
    environment = observed["environment"]
    assert isinstance(environment, dict)
    assert "QT_QPA_PLATFORM" not in environment
    assert payload["qt_platform"] == "cocoa"


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
    environment = install_root / "env-12ab34cd"
    scripts_root = environment / "Scripts"
    scripts_root.mkdir(parents=True)
    (scripts_root / "python.exe").touch()
    (scripts_root / "openhcs.exe").touch()
    launcher_path = install_root / "Launch-OpenHCS.ps1"
    launcher_path.write_text(
        '$environmentName = (Get-Content -LiteralPath '
        '(Join-Path $PSScriptRoot "current-environment") -Raw).Trim()\n'
        '& (Join-Path $PSScriptRoot '
        '"$environmentName\\Scripts\\openhcs.exe")',
        encoding="utf-8",
    )
    (install_root / "OpenHCS.exe").touch()
    (install_root / "current-environment").write_text(
        environment.name,
        encoding="utf-8",
    )
    desktop_root = tmp_path / "Desktop"
    desktop_root.mkdir()
    (desktop_root / "OpenHCS.lnk").touch()
    _stub_installed_probe(monkeypatch)
    viewer_modes: list[bool] = []

    def fake_demo(*_args, viewer):
        viewer_modes.append(viewer)
        return {
            "execution_status": "complete",
            "viewer_observed": True,
            "viewer_type": "napari",
        }

    monkeypatch.setattr(desktop_smoke, "_smoke_installed_demo", fake_demo)
    monkeypatch.setattr(
        desktop_smoke,
        "_smoke_installed_napari",
        lambda *_args: pytest.fail("Windows must keep its full viewer demo"),
    )

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
    assert viewer_modes == [True]
    assert "napari" not in result


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
    viewer_modes: list[bool] = []
    native_napari_calls: list[tuple[Path, Path]] = []

    def fake_demo(*_args, viewer):
        viewer_modes.append(viewer)
        return {
            "execution_status": "complete",
            "viewer_observed": False,
            "viewer_type": None,
        }

    def fake_napari(python_executable, installed_root):
        native_napari_calls.append((python_executable, installed_root))
        return {
            "viewer_type": "napari",
            "qt_platform": "cocoa",
            "layer_count": 1,
            "nonzero_count": 64,
            "closed": True,
        }

    monkeypatch.setattr(desktop_smoke, "_smoke_installed_demo", fake_demo)
    monkeypatch.setattr(desktop_smoke, "_smoke_installed_napari", fake_napari)

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
    assert viewer_modes == [False]
    assert native_napari_calls == [(bin_root / "python", install_root.resolve())]
    assert result["napari"]["qt_platform"] == "cocoa"
