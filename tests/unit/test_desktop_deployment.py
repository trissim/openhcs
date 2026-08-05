from __future__ import annotations

import json
import os
import plistlib
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


def _context(tmp_path: Path, *, pointer_name: str) -> DesktopDeploymentContext:
    install_root = tmp_path / "OpenHCS"
    environment_root = install_root / "environments" / "env-current"
    environment_root.mkdir(parents=True)
    uv_executable = install_root / "bootstrap" / "uv" / "uv"
    uv_executable.parent.mkdir(parents=True)
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


def test_windows_launcher_source_tracks_current_environment_and_mcp_pointer(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path, pointer_name="Launch-OpenHCS.ps1")
    powershell = tmp_path / "Windows" / "powershell.exe"

    source = WindowsDesktopDeployment.launcher_source(
        context,
        powershell_executable=powershell,
    )

    assert str(context.environment_root / "Scripts" / "openhcs.exe") in source
    assert str(context.uv_executable) in source
    assert MCP_INSTALLATION_POINTER_ENVIRONMENT_VARIABLE in source
    assert MCP_STABLE_LAUNCH_COMMAND_ENVIRONMENT_VARIABLE in source
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
    desktop_link = Path(first.desktop_shortcut_path)
    icon = application / "Contents" / "Resources" / "OpenHCS.icns"
    assert launcher.is_file() and os.access(launcher, os.X_OK)
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
