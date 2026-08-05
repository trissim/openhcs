"""Refresh installer-managed launchers, shortcuts, and application icons."""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import shlex
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from importlib.metadata import distribution
from pathlib import Path
from typing import ClassVar
from uuid import uuid4

from metaclass_registry import AutoRegisterMeta
from packaging.utils import canonicalize_name

from openhcs import __version__ as OPENHCS_VERSION
from openhcs.agent.runtime_platform import AgentRuntimePlatformKey
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.mcp.bootstrap import (
    MCP_INSTALLATION_POINTER_ENVIRONMENT_VARIABLE,
    MCP_STABLE_LAUNCH_COMMAND_ENVIRONMENT_VARIABLE,
)
from openhcs.resources.brand import (
    BRAND_PRODUCT_NAME,
    BrandAsset,
    brand_asset_path,
)


_UV_EXECUTABLE_ENVIRONMENT_VARIABLE = "OPENHCS_UV_EXECUTABLE"
_DISTRIBUTION_NAME = "openhcs"


class DesktopDeploymentError(RuntimeError):
    """Raised when an installer-managed desktop shell cannot be refreshed."""


@dataclass(frozen=True, slots=True)
class DesktopApplicationIdentity:
    """Brand and entry points projected from the installed distribution."""

    product_name: str
    command_entry_point: str
    gui_entry_point: str

    @classmethod
    def installed(cls) -> "DesktopApplicationIdentity":
        installed_distribution = distribution(_DISTRIBUTION_NAME)
        distribution_key = canonicalize_name(installed_distribution.metadata["Name"])
        command_entry_points = tuple(
            entry_point
            for entry_point in installed_distribution.entry_points
            if entry_point.group == "console_scripts"
            and canonicalize_name(entry_point.name) == distribution_key
        )
        gui_entry_points = tuple(
            entry_point
            for entry_point in installed_distribution.entry_points
            if entry_point.group == "gui_scripts"
        )
        if len(command_entry_points) != 1 or len(gui_entry_points) != 1:
            raise DesktopDeploymentError(
                "The installed OpenHCS distribution does not declare exactly one "
                "primary command and GUI entry point."
            )
        return cls(
            product_name=BRAND_PRODUCT_NAME,
            command_entry_point=command_entry_points[0].name,
            gui_entry_point=gui_entry_points[0].name,
        )


@dataclass(frozen=True, slots=True)
class DesktopDeploymentContext:
    """Validated installed environment and installer-owned stable pointer."""

    environment_root: Path
    install_root: Path
    installation_pointer: Path
    home: Path
    uv_executable: Path
    application: DesktopApplicationIdentity

    @classmethod
    def from_runtime(
        cls,
        installation_pointer: Path,
        *,
        environment_root: Path | None = None,
        home: Path | None = None,
        environment: dict[str, str] | None = None,
    ) -> "DesktopDeploymentContext":
        """Resolve one native-installer layout from its stable pointer."""

        values = os.environ if environment is None else environment
        pointer = installation_pointer.expanduser()
        if not pointer.is_absolute():
            raise DesktopDeploymentError(
                "The installer-managed desktop pointer must be an absolute path."
            )
        resolved_environment = (
            Path(sys.prefix) if environment_root is None else environment_root
        ).resolve()
        environments_root = resolved_environment.parent
        if environments_root.name != "environments":
            raise DesktopDeploymentError(
                "The running OpenHCS environment is not inside the native "
                "installer's environments directory."
            )
        install_root = environments_root.parent.resolve()
        if pointer.parent.resolve(strict=False) != install_root:
            raise DesktopDeploymentError(
                "The installer-managed desktop pointer does not belong to the "
                "running OpenHCS installation."
            )
        raw_uv_executable = values.get(_UV_EXECUTABLE_ENVIRONMENT_VARIABLE)
        if not raw_uv_executable:
            raise DesktopDeploymentError(
                "The native OpenHCS launcher did not identify its managed uv "
                "executable. Re-run the official installer to repair this install."
            )
        uv_executable = Path(raw_uv_executable).expanduser()
        if not uv_executable.is_absolute():
            raise DesktopDeploymentError(
                "The managed uv executable path must be absolute."
            )
        return cls(
            environment_root=resolved_environment,
            install_root=install_root,
            installation_pointer=pointer,
            home=(Path.home() if home is None else home).resolve(),
            uv_executable=uv_executable.resolve(strict=False),
            application=DesktopApplicationIdentity.installed(),
        )


@dataclass(frozen=True, slots=True)
class DesktopDeploymentReport:
    """Paths refreshed by one platform deployment authority."""

    platform: AgentRuntimePlatformKey
    launcher_path: str
    desktop_shortcut_path: str
    application_path: str | None


@dataclass(frozen=True, slots=True)
class _Publication:
    candidate: Path
    target: Path
    backup: Path


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _remove_path(path: Path) -> None:
    if not _path_exists(path):
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


class _AtomicPathPublication:
    """Publish prepared filesystem projections as one rollback-safe unit."""

    def __init__(self, *pairs: tuple[Path, Path]) -> None:
        transaction_id = uuid4().hex
        self._publications = tuple(
            _Publication(
                candidate=candidate,
                target=target,
                backup=target.with_name(f".{target.name}.backup-{transaction_id}"),
            )
            for candidate, target in pairs
        )

    def publish(self) -> None:
        published: list[_Publication] = []
        backed_up: list[_Publication] = []
        try:
            for publication in self._publications:
                if not _path_exists(publication.candidate):
                    raise DesktopDeploymentError(
                        "Desktop deployment candidate is missing: "
                        f"{publication.candidate}"
                    )
                _remove_path(publication.backup)
                if _path_exists(publication.target):
                    os.replace(publication.target, publication.backup)
                    backed_up.append(publication)
                os.replace(publication.candidate, publication.target)
                published.append(publication)
        except Exception:
            for publication in reversed(published):
                _remove_path(publication.target)
            for publication in reversed(backed_up):
                if _path_exists(publication.backup):
                    os.replace(publication.backup, publication.target)
            raise
        finally:
            for publication in self._publications:
                _remove_path(publication.candidate)
        for publication in backed_up:
            _remove_path(publication.backup)


def _candidate_path(target: Path) -> Path:
    return target.with_name(f".{target.stem}.candidate-{uuid4().hex}{target.suffix}")


def _powershell_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class DesktopDeploymentAuthority(
    EnumKeyedStrategyMixin[AgentRuntimePlatformKey],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered owner of one platform's installed desktop projections."""

    __enum_member_attr__ = "platform_key"
    platform_key: ClassVar[AgentRuntimePlatformKey]

    @classmethod
    def current(cls) -> "DesktopDeploymentAuthority":
        platform_key = AgentRuntimePlatformKey.current()
        try:
            return cls.for_enum_member(platform_key)
        except KeyError as exc:
            raise DesktopDeploymentError(
                "Native desktop deployment refresh is supported only for the "
                "Windows and macOS installers."
            ) from exc

    @abstractmethod
    def refresh(
        self,
        context: DesktopDeploymentContext,
    ) -> DesktopDeploymentReport:
        """Atomically refresh the platform launcher and visual shell."""


class WindowsDesktopDeployment(DesktopDeploymentAuthority):
    """Windows PowerShell launcher and Shell Link projection."""

    platform_key = AgentRuntimePlatformKey.WINDOWS

    @staticmethod
    def _powershell_executable(environment: dict[str, str]) -> Path:
        windows_root = environment.get("SystemRoot") or environment.get("WINDIR")
        if not windows_root:
            raise DesktopDeploymentError(
                "Windows did not identify its system directory."
            )
        executable = (
            Path(windows_root)
            / "System32"
            / "WindowsPowerShell"
            / "v1.0"
            / "powershell.exe"
        )
        if not executable.is_file():
            raise DesktopDeploymentError(
                f"Windows PowerShell is unavailable: {executable}"
            )
        return executable

    @staticmethod
    def launcher_source(
        context: DesktopDeploymentContext,
        *,
        powershell_executable: Path,
    ) -> str:
        """Render the stable launcher from the installed environment identity."""

        stable_command = json.dumps(
            [
                str(powershell_executable),
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(context.installation_pointer),
                "mcp",
            ],
            separators=(",", ":"),
        )
        entry_point = context.environment_root / "Scripts" / (
            f"{context.application.command_entry_point}.exe"
        )
        return "\n".join(
            (
                '$env:OPENHCS_CPU_ONLY = "true"',
                (
                    f"$env:{_UV_EXECUTABLE_ENVIRONMENT_VARIABLE} = "
                    f"{_powershell_literal(str(context.uv_executable))}"
                ),
                (
                    f"$env:{MCP_STABLE_LAUNCH_COMMAND_ENVIRONMENT_VARIABLE} = "
                    f"{_powershell_literal(stable_command)}"
                ),
                (
                    f"$env:{MCP_INSTALLATION_POINTER_ENVIRONMENT_VARIABLE} = "
                    f"{_powershell_literal(str(context.installation_pointer))}"
                ),
                f"& {_powershell_literal(str(entry_point))} @args",
                "exit $LASTEXITCODE",
                "",
            )
        )

    @staticmethod
    def _run_powershell(
        powershell_executable: Path,
        arguments: list[str],
    ) -> subprocess.CompletedProcess[str]:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return subprocess.run(
            [str(powershell_executable), *arguments],
            check=True,
            capture_output=True,
            text=True,
            creationflags=creationflags,
        )

    def _desktop_directory(self, powershell_executable: Path) -> Path:
        result = self._run_powershell(
            powershell_executable,
            [
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "[Environment]::GetFolderPath('DesktopDirectory')",
            ],
        )
        desktop = Path(result.stdout.strip())
        if not desktop.is_absolute():
            raise DesktopDeploymentError(
                "Windows did not provide an absolute user Desktop directory."
            )
        return desktop

    def _create_shortcut(
        self,
        *,
        powershell_executable: Path,
        shortcut_path: Path,
        target_path: Path,
        working_directory: Path,
        icon_path: Path,
        product_name: str,
    ) -> None:
        script_path = shortcut_path.with_name(
            f".{shortcut_path.stem}.create-{uuid4().hex}.ps1"
        )
        script_path.write_text(
            """param(
    [Parameter(Mandatory = $true)][string]$ShortcutPath,
    [Parameter(Mandatory = $true)][string]$TargetPath,
    [Parameter(Mandatory = $true)][string]$WorkingDirectory,
    [Parameter(Mandatory = $true)][string]$IconPath,
    [Parameter(Mandatory = $true)][string]$ProductName
)
$ErrorActionPreference = "Stop"
$shell = New-Object -ComObject WScript.Shell
try {
    $shortcut = $shell.CreateShortcut($ShortcutPath)
    try {
        $shortcut.TargetPath = $TargetPath
        $shortcut.Arguments = ""
        $shortcut.WorkingDirectory = $WorkingDirectory
        $shortcut.Description = "Launch $ProductName"
        $shortcut.IconLocation = "$IconPath,0"
        $shortcut.Save()
    }
    finally {
        if ($null -ne $shortcut) {
            [Runtime.InteropServices.Marshal]::FinalReleaseComObject($shortcut) |
                Out-Null
        }
    }
}
finally {
    [Runtime.InteropServices.Marshal]::FinalReleaseComObject($shell) | Out-Null
}
""",
            encoding="utf-8",
        )
        try:
            self._run_powershell(
                powershell_executable,
                [
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(script_path),
                    "-ShortcutPath",
                    str(shortcut_path),
                    "-TargetPath",
                    str(target_path),
                    "-WorkingDirectory",
                    str(working_directory),
                    "-IconPath",
                    str(icon_path),
                    "-ProductName",
                    product_name,
                ],
            )
        finally:
            script_path.unlink(missing_ok=True)

    @staticmethod
    def _notify_shortcut_published(shortcut_path: Path) -> None:
        import ctypes

        shortcut_created = 0x00000002
        shortcut_updated = 0x00002000
        path_unicode = 0x0005
        flush_notification = 0x1000
        ctypes.windll.shell32.SHChangeNotify(
            shortcut_created | shortcut_updated,
            path_unicode | flush_notification,
            ctypes.c_wchar_p(str(shortcut_path)),
            None,
        )

    def refresh(
        self,
        context: DesktopDeploymentContext,
    ) -> DesktopDeploymentReport:
        powershell_executable = self._powershell_executable(os.environ)
        expected_pointer_name = (
            f"Launch-{context.application.product_name.replace(' ', '-')}.ps1"
        )
        if context.installation_pointer.name != expected_pointer_name:
            raise DesktopDeploymentError(
                "The Windows installation pointer is not the stable OpenHCS launcher."
            )
        gui_executable = context.environment_root / "Scripts" / (
            f"{context.application.gui_entry_point}.exe"
        )
        entry_executable = context.environment_root / "Scripts" / (
            f"{context.application.command_entry_point}.exe"
        )
        icon_path = brand_asset_path(BrandAsset.WINDOWS_ICON)
        for required_path in (gui_executable, entry_executable, icon_path):
            if not required_path.is_file():
                raise DesktopDeploymentError(
                    f"Installed desktop resource is unavailable: {required_path}"
                )

        desktop_directory = self._desktop_directory(powershell_executable)
        desktop_directory.mkdir(parents=True, exist_ok=True)
        shortcut_path = desktop_directory / (
            f"{context.application.product_name}.lnk"
        )
        launcher_candidate = _candidate_path(context.installation_pointer)
        shortcut_candidate = _candidate_path(shortcut_path)
        launcher_candidate.write_text(
            self.launcher_source(
                context,
                powershell_executable=powershell_executable,
            ),
            encoding="utf-8-sig",
        )
        self._create_shortcut(
            powershell_executable=powershell_executable,
            shortcut_path=shortcut_candidate,
            target_path=gui_executable,
            working_directory=context.install_root,
            icon_path=icon_path,
            product_name=context.application.product_name,
        )
        _AtomicPathPublication(
            (launcher_candidate, context.installation_pointer),
            (shortcut_candidate, shortcut_path),
        ).publish()
        self._notify_shortcut_published(shortcut_path)
        return DesktopDeploymentReport(
            platform=self.platform_key,
            launcher_path=str(context.installation_pointer),
            desktop_shortcut_path=str(shortcut_path),
            application_path=str(gui_executable),
        )


class MacOSDesktopDeployment(DesktopDeploymentAuthority):
    """macOS environment launcher, app bundle, and Desktop link projection."""

    platform_key = AgentRuntimePlatformKey.MACOS

    @staticmethod
    def environment_launcher_source(context: DesktopDeploymentContext) -> str:
        stable_launcher = context.installation_pointer / "launch-openhcs.sh"
        stable_command = json.dumps(
            [str(stable_launcher), "mcp"], separators=(",", ":")
        )
        entry_point = (
            context.environment_root
            / "bin"
            / context.application.command_entry_point
        )
        return "\n".join(
            (
                "#!/bin/bash",
                "set -euo pipefail",
                "export OPENHCS_CPU_ONLY=true",
                (
                    f"export {_UV_EXECUTABLE_ENVIRONMENT_VARIABLE}="
                    f"{shlex.quote(str(context.uv_executable))}"
                ),
                (
                    f"export {MCP_STABLE_LAUNCH_COMMAND_ENVIRONMENT_VARIABLE}="
                    f"{shlex.quote(stable_command)}"
                ),
                (
                    f"export {MCP_INSTALLATION_POINTER_ENVIRONMENT_VARIABLE}="
                    f"{shlex.quote(str(context.installation_pointer))}"
                ),
                f"exec {shlex.quote(str(entry_point))} \"$@\"",
                "",
            )
        )

    @staticmethod
    def _prepare_application(
        candidate: Path,
        *,
        context: DesktopDeploymentContext,
    ) -> None:
        executable_directory = candidate / "Contents" / "MacOS"
        resources_directory = candidate / "Contents" / "Resources"
        executable_directory.mkdir(parents=True)
        resources_directory.mkdir(parents=True)
        shutil.copyfile(
            brand_asset_path(BrandAsset.MACOS_ICON),
            resources_directory / "OpenHCS.icns",
        )
        with (candidate / "Contents" / "Info.plist").open("wb") as stream:
            plistlib.dump(
                {
                    "CFBundleDisplayName": context.application.product_name,
                    "CFBundleExecutable": "launch-openhcs",
                    "CFBundleIdentifier": "org.openhcs.desktop",
                    "CFBundleIconFile": "OpenHCS.icns",
                    "CFBundleName": context.application.product_name,
                    "CFBundlePackageType": "APPL",
                    "CFBundleShortVersionString": OPENHCS_VERSION,
                    "CFBundleVersion": OPENHCS_VERSION,
                },
                stream,
                sort_keys=True,
            )
        app_launcher = executable_directory / "launch-openhcs"
        app_launcher.write_text(
            "\n".join(
                (
                    "#!/bin/bash",
                    "set -euo pipefail",
                    (
                        f"exec {shlex.quote(str(context.installation_pointer / 'launch-openhcs.sh'))} "
                        '"$@"'
                    ),
                    "",
                )
            ),
            encoding="utf-8",
        )
        app_launcher.chmod(0o755)

    def refresh(
        self,
        context: DesktopDeploymentContext,
    ) -> DesktopDeploymentReport:
        if context.installation_pointer.name != "current":
            raise DesktopDeploymentError(
                "The macOS installation pointer is not the current environment link."
            )
        entry_point = (
            context.environment_root
            / "bin"
            / context.application.command_entry_point
        )
        icon_path = brand_asset_path(BrandAsset.MACOS_ICON)
        for required_path in (entry_point, icon_path):
            if not required_path.is_file():
                raise DesktopDeploymentError(
                    f"Installed desktop resource is unavailable: {required_path}"
                )

        applications_directory = context.home / "Applications"
        desktop_directory = context.home / "Desktop"
        applications_directory.mkdir(parents=True, exist_ok=True)
        desktop_directory.mkdir(parents=True, exist_ok=True)
        application_path = applications_directory / (
            f"{context.application.product_name}.app"
        )
        desktop_link = desktop_directory / (
            f"{context.application.product_name}.app"
        )
        if _path_exists(desktop_link) and not desktop_link.is_symlink():
            raise DesktopDeploymentError(
                "Refusing to replace a non-link Desktop item: " f"{desktop_link}"
            )

        environment_launcher = context.environment_root / "launch-openhcs.sh"
        launcher_candidate = _candidate_path(environment_launcher)
        application_candidate = _candidate_path(application_path)
        desktop_candidate = _candidate_path(desktop_link)
        pointer_candidate = _candidate_path(context.installation_pointer)
        launcher_candidate.write_text(
            self.environment_launcher_source(context),
            encoding="utf-8",
        )
        launcher_candidate.chmod(0o755)
        self._prepare_application(
            application_candidate,
            context=context,
        )
        desktop_candidate.symlink_to(application_path)
        pointer_candidate.symlink_to(context.environment_root)
        _AtomicPathPublication(
            (launcher_candidate, environment_launcher),
            (application_candidate, application_path),
            (desktop_candidate, desktop_link),
            (pointer_candidate, context.installation_pointer),
        ).publish()
        os.utime(application_path, None)
        return DesktopDeploymentReport(
            platform=self.platform_key,
            launcher_path=str(environment_launcher),
            desktop_shortcut_path=str(desktop_link),
            application_path=str(application_path),
        )


def refresh_installer_managed_desktop(
    installation_pointer: Path,
) -> DesktopDeploymentReport:
    """Refresh the current native installation through its platform authority."""

    context = DesktopDeploymentContext.from_runtime(installation_pointer)
    return DesktopDeploymentAuthority.current().refresh(context)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--installation-pointer",
        type=Path,
        default=None,
        help=(
            "Absolute native-installer pointer. Defaults to the pointer exported "
            "by the stable OpenHCS launcher."
        ),
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    pointer = arguments.installation_pointer
    if pointer is None:
        raw_pointer = os.environ.get(
            MCP_INSTALLATION_POINTER_ENVIRONMENT_VARIABLE
        )
        if raw_pointer is None:
            raise DesktopDeploymentError(
                "OpenHCS was not launched through a native installer-managed "
                "desktop integration."
            )
        pointer = Path(raw_pointer)
    report = refresh_installer_managed_desktop(pointer)
    if arguments.json:
        print(json.dumps(asdict(report), sort_keys=True))
    else:
        print(f"Refreshed OpenHCS desktop integration: {report.launcher_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
