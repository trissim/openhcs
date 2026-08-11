from __future__ import annotations

import json
import os
import plistlib
import subprocess
import struct
import sys
from pathlib import Path

import pytest

from openhcs.agent.runtime_platform import AgentRuntimePlatformKey
from openhcs.desktop_deployment import (
    DesktopDeploymentAuthority,
    DesktopDeploymentContext,
    DesktopDeploymentError,
    MacOSDesktopDeployment,
    WindowsDesktopDeployment,
)
from openhcs.mcp.bootstrap import (
    MCP_INSTALLATION_POINTER_ENVIRONMENT_VARIABLE,
    MCP_STABLE_LAUNCH_COMMAND_ENVIRONMENT_VARIABLE,
)
from openhcs.resources.brand import BrandAsset, brand_asset_path
from openhcs.utils.environment import OpenHCSProcessEnvironment


def test_desktop_deployment_import_does_not_load_agent_dto_graph() -> None:
    """Keep post-install shortcut publication outside agent schema startup."""

    checkout = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
                "-c",
                (
                    f"import sys; sys.path.insert(0, {str(checkout)!r}); "
                    "import openhcs.desktop_deployment; "
                "assert 'openhcs.agent.dto.common' not in sys.modules; "
                "assert 'python_introspect' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert completed.returncode == 0, completed.stderr


def _context(tmp_path: Path, *, pointer_name: str) -> DesktopDeploymentContext:
    install_root = tmp_path / "OpenHCS"
    environment_root = install_root / "environments" / "env-current"
    environment_root.mkdir(parents=True)
    uv_executable = install_root / "bootstrap" / "uv" / "uv"
    uv_executable.parent.mkdir(parents=True, exist_ok=True)
    uv_executable.touch()
    return DesktopDeploymentContext.from_runtime(
        install_root / pointer_name,
        environment_root=environment_root,
        home=tmp_path / "home",
        environment={"OPENHCS_UV_EXECUTABLE": str(uv_executable)},
    )


def test_desktop_deployment_platforms_use_the_registered_host_axis() -> None:
    assert (
        DesktopDeploymentAuthority.strategy_type_for_enum_member(
            AgentRuntimePlatformKey.WINDOWS
        )
        is WindowsDesktopDeployment
    )
    assert (
        DesktopDeploymentAuthority.strategy_type_for_enum_member(
            AgentRuntimePlatformKey.MACOS
        )
        is MacOSDesktopDeployment
    )


def test_context_rejects_pointer_outside_installer_layout(tmp_path: Path) -> None:
    environment_root = tmp_path / "OpenHCS" / "environments" / "env-current"
    environment_root.mkdir(parents=True)
    uv_executable = tmp_path / "uv"
    uv_executable.touch()

    with pytest.raises(DesktopDeploymentError, match="does not belong"):
        DesktopDeploymentContext.from_runtime(
            tmp_path / "elsewhere" / "current",
            environment_root=environment_root,
            environment={"OPENHCS_UV_EXECUTABLE": str(uv_executable)},
        )


@pytest.mark.parametrize(
    "environment_relative",
    (Path("env-current"), Path("environments") / "env-current"),
)
def test_context_accepts_current_and_legacy_installer_layouts(
    tmp_path: Path,
    environment_relative: Path,
) -> None:
    install_root = tmp_path / "OpenHCS"
    environment_root = install_root / environment_relative
    environment_root.mkdir(parents=True)
    uv_executable = install_root / "bootstrap" / "uv" / "uv.exe"
    uv_executable.parent.mkdir(parents=True)
    uv_executable.touch()

    context = DesktopDeploymentContext.from_runtime(
        install_root / "Launch-OpenHCS.ps1",
        environment_root=environment_root,
        environment={"OPENHCS_UV_EXECUTABLE": str(uv_executable)},
    )

    assert context.install_root == install_root.resolve()
    assert context.environment_root == environment_root.resolve()
    assert context.numba_cache_path == (install_root / "cache" / "numba").resolve()


def test_windows_mcp_launcher_reads_atomic_current_environment_pointer(
    tmp_path: Path,
) -> None:
    context = _windows_context(tmp_path, "env-12ab34cd")
    powershell = tmp_path / "Windows" / "powershell.exe"

    source = WindowsDesktopDeployment.mcp_launcher_source(
        context,
        powershell_executable=powershell,
    )

    assert "current-environment" in source
    assert context.environment_root.name not in source
    assert str(context.environment_root.parent) in source
    assert 'Join-Path $environmentRoot "Scripts\\openhcs.exe"' in source
    assert '"environments"' not in source
    assert "GetDirectoryName($environmentRoot)" in source
    assert "StringComparison]::OrdinalIgnoreCase" in source
    assert str(context.uv_executable) in source
    assert MCP_INSTALLATION_POINTER_ENVIRONMENT_VARIABLE in source
    assert MCP_STABLE_LAUNCH_COMMAND_ENVIRONMENT_VARIABLE in source
    assert OpenHCSProcessEnvironment.numba_cache_key in source
    assert (
        str(OpenHCSProcessEnvironment.numba_cache_path(context.install_root)) in source
    )
    stable_command_line = next(
        line
        for line in source.splitlines()
        if MCP_STABLE_LAUNCH_COMMAND_ENVIRONMENT_VARIABLE in line
    )
    encoded = stable_command_line.split("=", maxsplit=1)[1].strip().strip("'")
    assert json.loads(encoded) == [
        str(powershell),
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(context.installation_pointer),
        "mcp",
    ]


def test_windows_native_launcher_forwards_to_declared_module_without_entry_shim(
    tmp_path: Path,
) -> None:
    context = _windows_context(
        tmp_path,
        "env-20260805T120000Z-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )
    source = WindowsDesktopDeployment.native_launcher_source(
        context,
        powershell_executable=Path(
            "C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
        ),
    )

    assert "__OPENHCS_" not in source
    assert '"OPENHCS_STARTUP_HANDOFF_EVENT"' in source
    assert 'startInfo.FileName = pythonExecutable;' in source
    assert 'private const string GuiModule = "openhcs.pyqt_gui.__main__";' in source
    assert 'Path.Combine(scripts, "pythonw.exe")' in source
    assert 'Path.Combine(scripts, "python.exe")' in source
    assert "ModuleArguments(arguments)" in source
    assert "openhcs-gui.exe" not in source
    assert "startInfo.CreateNoWindow = true;" in source
    assert '"current-environment"' in source
    assert 'EnvironmentContainerRelativePath =\n        "";' in source
    assert "Directory.GetParent(environmentRoot)" in source
    assert "StringComparison.OrdinalIgnoreCase" in source
    assert '"OPENHCS_UV_EXECUTABLE"' in source
    assert '"OPENHCS_MCP_INSTALLATION_POINTER"' in source
    assert '"NUMBA_CACHE_DIR"' in source
    assert (
        str(OpenHCSProcessEnvironment.numba_cache_path(context.install_root)) in source
    )


def test_windows_powershell_failure_preserves_process_diagnostics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=1,
            stdout="compiler stdout",
            stderr="compiler stderr",
        ),
    )

    with pytest.raises(
        DesktopDeploymentError,
        match="compiler stdout\\ncompiler stderr",
    ):
        WindowsDesktopDeployment._run_powershell(
            tmp_path / "powershell.exe",
            ["-File", "compile.ps1"],
        )


def _gui_subsystem_fixture() -> bytes:
    content = bytearray(256)
    struct.pack_into("<I", content, 0x3C, 0x80)
    content[0x80:0x84] = b"PE\0\0"
    struct.pack_into("<H", content, 0x80 + 24 + 68, 2)
    return bytes(content)


def test_windows_native_launcher_compilation_uses_powershell_5_contract(
    monkeypatch,
    tmp_path: Path,
) -> None:
    deployment = WindowsDesktopDeployment()
    output_path = tmp_path / "OpenHCS.exe"
    icon_path = tmp_path / "OpenHCS.ico"
    icon_path.write_bytes(b"icon")
    scripts: list[str] = []

    def run_powershell(_executable: Path, arguments: list[str]):
        script_path = Path(arguments[arguments.index("-File") + 1])
        scripts.append(script_path.read_text(encoding="utf-8"))
        output_path.write_bytes(_gui_subsystem_fixture())
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(deployment, "_run_powershell", run_powershell)

    deployment._compile_native_launcher(
        powershell_executable=tmp_path / "powershell.exe",
        source="internal static class OpenHCSLauncher {}",
        icon_path=icon_path,
        output_path=output_path,
    )

    assert len(scripts) == 1
    assert "System.CodeDom.Compiler.CompilerParameters" in scripts[0]
    assert "-CompilerParameters $compilerParameters" in scripts[0]
    assert "-CompilerOptions" not in scripts[0]
    assert "/target:winexe" in scripts[0]
    assert "/win32icon:" in scripts[0]


def test_windows_native_launcher_leaves_startup_presentation_to_qt(
    tmp_path: Path,
) -> None:
    """The stable forwarder waits invisibly for the one Qt startup surface."""

    source = WindowsDesktopDeployment.native_launcher_source(
        _windows_context(
            tmp_path,
            "env-20260805T120000Z-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        ),
        powershell_executable=Path(r"C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe"),
    )

    assert "class StartupWindow" not in source
    assert "Application.Run" not in source
    assert "handoffEvent.WaitOne(100)" in source
    assert "process.HasExited" in source
    assert "MessageBox.Show" in source


def _windows_context(tmp_path: Path, environment_name: str) -> DesktopDeploymentContext:
    install_root = tmp_path / "OpenHCS"
    environment_root = install_root / environment_name
    scripts = environment_root / "Scripts"
    scripts.mkdir(parents=True)
    (scripts / "openhcs.exe").write_bytes(b"command")
    (scripts / "openhcs-gui.exe").write_bytes(_gui_subsystem_fixture())
    uv_executable = install_root / "bootstrap" / "uv" / "uv.exe"
    uv_executable.parent.mkdir(parents=True, exist_ok=True)
    uv_executable.write_bytes(b"uv")
    return DesktopDeploymentContext.from_runtime(
        install_root / "Launch-OpenHCS.ps1",
        environment_root=environment_root,
        home=tmp_path / "home",
        environment={"OPENHCS_UV_EXECUTABLE": str(uv_executable)},
    )


def test_windows_refresh_rejects_console_gui_entry_point(
    monkeypatch,
    tmp_path: Path,
) -> None:
    context = _windows_context(
        tmp_path,
        "env-20260805T120000Z-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )
    gui_executable = context.environment_root / "Scripts" / "openhcs-gui.exe"
    console_executable = bytearray(_gui_subsystem_fixture())
    struct.pack_into("<H", console_executable, 0x80 + 24 + 68, 3)
    gui_executable.write_bytes(console_executable)
    deployment = WindowsDesktopDeployment()
    monkeypatch.setattr(
        deployment,
        "_powershell_executable",
        lambda _environment: tmp_path / "powershell.exe",
    )

    with pytest.raises(
        DesktopDeploymentError,
        match="installed GUI entry point is not a GUI-subsystem executable",
    ):
        deployment.refresh(context)


def test_windows_refresh_publishes_stable_gui_launcher_and_reuses_its_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    first_context = _windows_context(
        tmp_path,
        "env-20260805T120000Z-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )
    desktop = tmp_path / "Desktop"
    deployment = WindowsDesktopDeployment()
    compile_calls: list[Path] = []
    shortcut_targets: list[Path] = []
    powershell = tmp_path / "Windows" / "powershell.exe"
    powershell.parent.mkdir()
    powershell.write_bytes(b"powershell")

    monkeypatch.setattr(deployment, "_powershell_executable", lambda _env: powershell)
    monkeypatch.setattr(deployment, "_desktop_directory", lambda _powershell: desktop)
    monkeypatch.setattr(deployment, "_notify_shortcut_published", lambda _path: None)

    def compile_launcher(**kwargs) -> None:
        compile_calls.append(kwargs["output_path"])
        kwargs["output_path"].write_bytes(_gui_subsystem_fixture())

    def create_shortcut(**kwargs) -> None:
        shortcut_targets.append(kwargs["target_path"])
        kwargs["shortcut_path"].write_text(
            str(kwargs["target_path"]),
            encoding="utf-8",
        )

    monkeypatch.setattr(deployment, "_compile_native_launcher", compile_launcher)
    monkeypatch.setattr(deployment, "_create_shortcut", create_shortcut)

    first_report = deployment.refresh(first_context)
    stable_launcher = first_context.install_root / "OpenHCS.exe"

    assert Path(first_report.application_path or "") == stable_launcher
    assert Path(first_report.restart_executable) == stable_launcher
    assert stable_launcher.read_bytes() == _gui_subsystem_fixture()
    assert shortcut_targets == [stable_launcher]
    assert (first_context.install_root / "current-environment").read_text(
        encoding="utf-8"
    ) == first_context.environment_root.name
    assert "current-environment" in first_context.installation_pointer.read_text(
        encoding="utf-8-sig"
    )

    second_context = _windows_context(
        tmp_path,
        "env-20260805T121500Z-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    )
    second_report = deployment.refresh(second_context)

    assert Path(second_report.application_path or "") == stable_launcher
    assert Path(second_report.restart_executable) == stable_launcher
    assert len(compile_calls) == 1
    assert shortcut_targets == [stable_launcher, stable_launcher]
    assert (first_context.install_root / "current-environment").read_text(
        encoding="utf-8"
    ) == second_context.environment_root.name


def test_macos_refresh_rewrites_launcher_icon_and_deleted_desktop_link(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path, pointer_name="current")
    entry_point = context.environment_root / "bin" / "openhcs"
    entry_point.parent.mkdir()
    entry_point.write_text("#!/bin/sh\n", encoding="utf-8")
    entry_point.chmod(0o755)
    context.installation_pointer.symlink_to(context.environment_root)
    deployment = MacOSDesktopDeployment()

    first = deployment.refresh(context)
    launcher = Path(first.launcher_path)
    application = Path(first.application_path or "")
    assert Path(first.restart_executable) == (
        application / "Contents" / "MacOS" / "launch-openhcs"
    )
    desktop_link = Path(first.desktop_shortcut_path)
    icon = application / "Contents" / "Resources" / "OpenHCS.icns"
    assert launcher.is_file() and os.access(launcher, os.X_OK)
    assert str(first.restart_executable) in launcher.read_text(encoding="utf-8")
    assert desktop_link.is_symlink()
    assert desktop_link.resolve() == application.resolve()
    assert icon.read_bytes() == brand_asset_path(BrandAsset.MACOS_ICON).read_bytes()
    with (application / "Contents" / "Info.plist").open("rb") as stream:
        plist = plistlib.load(stream)
    assert plist["CFBundleIconFile"] == "OpenHCS.icns"

    launcher.write_text("stale launcher", encoding="utf-8")
    icon.write_bytes(b"stale icon")
    desktop_link.unlink()
    context.installation_pointer.unlink()

    second = deployment.refresh(context)

    assert Path(second.launcher_path).read_text(encoding="utf-8").startswith(
        "#!/bin/bash"
    )
    assert icon.read_bytes() == brand_asset_path(BrandAsset.MACOS_ICON).read_bytes()
    assert desktop_link.is_symlink()
    assert desktop_link.resolve() == application.resolve()
    assert context.installation_pointer.is_symlink()
    assert context.installation_pointer.resolve() == context.environment_root
