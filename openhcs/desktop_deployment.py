"""Refresh installer-managed launchers, shortcuts, and application icons."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import plistlib
import shlex
import shutil
import struct
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from importlib.metadata import distribution
from importlib.resources import files
from pathlib import Path, PureWindowsPath
from typing import ClassVar
from uuid import uuid4

from metaclass_registry import AutoRegisterMeta
from packaging.utils import canonicalize_name

from openhcs import __version__ as OPENHCS_VERSION
from openhcs.agent.runtime_platform import (
    AgentRuntimePlatformAuthority,
    AgentRuntimePlatformKey,
)
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
from openhcs.utils.environment import OpenHCSProcessEnvironment

_UV_EXECUTABLE_ENVIRONMENT_VARIABLE = "OPENHCS_UV_EXECUTABLE"
DESKTOP_RESTART_EXECUTABLE_ENVIRONMENT_VARIABLE = "OPENHCS_DESKTOP_RESTART_EXECUTABLE"
_DISTRIBUTION_NAME = "openhcs"
_WINDOWS_GUI_PE_SUBSYSTEM = 2


class DesktopDeploymentError(RuntimeError):
    """Raised when an installer-managed desktop shell cannot be refreshed."""


@dataclass(frozen=True, slots=True)
class DesktopApplicationIdentity:
    """Brand and entry points projected from the installed distribution."""

    product_name: str
    command_entry_point: str
    gui_entry_point: str
    gui_module: str

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
            gui_module=gui_entry_points[0].module,
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
        install_root = pointer.parent.resolve(strict=False)
        try:
            resolved_environment.relative_to(install_root)
        except ValueError as exc:
            raise DesktopDeploymentError(
                "The installer-managed desktop pointer does not belong to the "
                "running OpenHCS installation."
            ) from exc
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
    restart_executable: str
    deferred_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DesktopEnvironmentCandidate:
    """Platform-owned paths for one unpublished managed environment."""

    root: Path
    python_executable: Path

    @classmethod
    def under(
        cls,
        parent: Path,
        python_relative_path: Path,
        *,
        transaction_id: str | None = None,
    ) -> "DesktopEnvironmentCandidate":
        identifier = transaction_id or uuid4().hex[:8]
        if len(identifier) != 8 or any(
            character not in "0123456789abcdef" for character in identifier
        ):
            raise DesktopDeploymentError(
                "Desktop update transaction identifiers must be eight lowercase "
                "hexadecimal characters."
            )
        root = parent / f"env-{identifier}"
        return cls(
            root=root,
            python_executable=root / python_relative_path,
        )


@dataclass(frozen=True, slots=True)
class _Publication:
    candidate: Path
    target: Path
    backup: Path


@dataclass(frozen=True, slots=True)
class _WindowsLauncherFingerprint:
    """Source/icon inputs and realized native-launcher identity."""

    inputs_sha256: str
    executable_sha256: str

    @classmethod
    def read(cls, path: Path) -> "_WindowsLauncherFingerprint":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("Windows launcher fingerprint must be a JSON object.")
        return cls(**payload)

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), sort_keys=True), encoding="utf-8")


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _remove_path(path: Path) -> None:
    if not _path_exists(path):
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _discard_transaction_path(path: Path) -> None:
    """Best-effort cleanup that cannot invalidate a completed publication."""

    try:
        _remove_path(path)
    except OSError:
        pass


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
                _discard_transaction_path(publication.candidate)
        for publication in backed_up:
            _discard_transaction_path(publication.backup)


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

    @classmethod
    def numba_cache_path(cls) -> Path:
        """Return this platform's user-local compiled-code cache."""

        platform_authority = AgentRuntimePlatformAuthority.for_enum_member(
            cls.platform_key
        )
        return OpenHCSProcessEnvironment.numba_cache_path(platform_authority)

    @abstractmethod
    def refresh(
        self,
        context: DesktopDeploymentContext,
    ) -> DesktopDeploymentReport:
        """Atomically refresh the platform launcher and visual shell."""

    @abstractmethod
    def update_candidate(
        self,
        context: DesktopDeploymentContext,
        *,
        transaction_id: str | None = None,
    ) -> DesktopEnvironmentCandidate:
        """Declare an unpublished environment path for one update transaction."""


class WindowsDesktopDeployment(DesktopDeploymentAuthority):
    """Windows native GUI, stable MCP launcher, and Shell Link projection."""

    platform_key = AgentRuntimePlatformKey.WINDOWS
    _application_launcher_name = "OpenHCS.exe"
    _current_environment_pointer_name = "current-environment"
    _launcher_fingerprint_name = "OpenHCS-launcher-fingerprint.json"
    _sharing_violation_codes = frozenset((32, 33))

    def update_candidate(
        self,
        context: DesktopDeploymentContext,
        *,
        transaction_id: str | None = None,
    ) -> DesktopEnvironmentCandidate:
        return DesktopEnvironmentCandidate.under(
            context.environment_root.parent,
            Path("Scripts") / "python.exe",
            transaction_id=transaction_id,
        )

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
    def _stable_mcp_command(
        context: DesktopDeploymentContext,
        *,
        powershell_executable: Path,
    ) -> str:
        return json.dumps(
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

    @classmethod
    def mcp_launcher_source(
        cls,
        context: DesktopDeploymentContext,
        *,
        powershell_executable: Path,
    ) -> str:
        """Render the stable MCP launcher from the current pointer authority."""

        stable_command = cls._stable_mcp_command(
            context,
            powershell_executable=powershell_executable,
        )
        current_pointer = context.install_root / cls._current_environment_pointer_name
        return "\n".join(
            (
                '$ErrorActionPreference = "Stop"',
                (
                    "$environmentName = "
                    f"(Get-Content -LiteralPath {_powershell_literal(str(current_pointer))} "
                    "-Raw).Trim()"
                ),
                (
                    "$environmentContainer = "
                    f"{_powershell_literal(str(context.environment_root.parent))}"
                ),
                (
                    "$environmentRoot = [IO.Path]::GetFullPath("
                    "(Join-Path $environmentContainer $environmentName))"
                ),
                (
                    "$expectedEnvironmentParent = "
                    "[IO.Path]::GetFullPath($environmentContainer).TrimEnd('\\', '/')"
                ),
                (
                    "$actualEnvironmentParent = "
                    "[IO.Path]::GetDirectoryName($environmentRoot).TrimEnd('\\', '/')"
                ),
                (
                    "if (-not [string]::Equals($actualEnvironmentParent, "
                    "$expectedEnvironmentParent, "
                    "[StringComparison]::OrdinalIgnoreCase)) {"
                ),
                (
                    '    throw "The installed current-environment pointer is invalid. '
                    'Re-run the official OpenHCS installer to repair it."'
                ),
                "}",
                (
                    "$entryPoint = Join-Path $environmentRoot "
                    f'"Scripts\\{context.application.command_entry_point}.exe"'
                ),
                "if (-not (Test-Path -LiteralPath $entryPoint -PathType Leaf)) {",
                '    throw "The current OpenHCS command entry point is unavailable."',
                "}",
                f'$env:{OpenHCSProcessEnvironment.cpu_only_key} = "true"',
                (
                    f"$env:{OpenHCSProcessEnvironment.numba_cache_key} = "
                    f"{_powershell_literal(str(cls.numba_cache_path()))}"
                ),
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
                "& $entryPoint @args",
                "exit $LASTEXITCODE",
                "",
            )
        )

    @classmethod
    def native_launcher_source(
        cls,
        context: DesktopDeploymentContext,
        *,
        powershell_executable: Path,
    ) -> str:
        """Render the packaged WinExe from desktop and IPC authorities."""

        from openhcs.gui_startup import STARTUP_HANDOFF_EVENT_ENVIRONMENT

        source = (files("openhcs.resources.windows") / "OpenHCSLauncher.cs").read_text(
            encoding="utf-8"
        )
        environment_container_relative = context.environment_root.parent.relative_to(
            context.install_root
        )
        uv_relative = context.uv_executable.relative_to(context.install_root)
        values = {
            "__OPENHCS_PRODUCT_NAME__": context.application.product_name,
            "__OPENHCS_CURRENT_ENVIRONMENT_POINTER_NAME__": (
                cls._current_environment_pointer_name
            ),
            "__OPENHCS_MCP_LAUNCHER_NAME__": context.installation_pointer.name,
            "__OPENHCS_ENVIRONMENT_CONTAINER_RELATIVE_PATH__": (
                str(PureWindowsPath(*environment_container_relative.parts))
                if environment_container_relative.parts
                else ""
            ),
            "__OPENHCS_GUI_MODULE__": context.application.gui_module,
            "__OPENHCS_UV_RELATIVE_PATH__": str(PureWindowsPath(*uv_relative.parts)),
            "__OPENHCS_CPU_ONLY_ENVIRONMENT__": (
                OpenHCSProcessEnvironment.cpu_only_key
            ),
            "__OPENHCS_NUMBA_CACHE_ENVIRONMENT__": (
                OpenHCSProcessEnvironment.numba_cache_key
            ),
            "__OPENHCS_NUMBA_CACHE_PATH__": str(cls.numba_cache_path()),
            "__OPENHCS_UV_ENVIRONMENT__": _UV_EXECUTABLE_ENVIRONMENT_VARIABLE,
            "__OPENHCS_RESTART_EXECUTABLE_ENVIRONMENT__": (
                DESKTOP_RESTART_EXECUTABLE_ENVIRONMENT_VARIABLE
            ),
            "__OPENHCS_MCP_INSTALLATION_POINTER_ENVIRONMENT__": (
                MCP_INSTALLATION_POINTER_ENVIRONMENT_VARIABLE
            ),
            "__OPENHCS_MCP_STABLE_COMMAND_ENVIRONMENT__": (
                MCP_STABLE_LAUNCH_COMMAND_ENVIRONMENT_VARIABLE
            ),
            "__OPENHCS_STARTUP_HANDOFF_EVENT__": (STARTUP_HANDOFF_EVENT_ENVIRONMENT),
            "__OPENHCS_STABLE_MCP_COMMAND_JSON__": cls._stable_mcp_command(
                context,
                powershell_executable=powershell_executable,
            ),
        }
        for placeholder, value in values.items():
            if source.count(placeholder) != 1:
                raise DesktopDeploymentError(
                    "The packaged Windows launcher has an invalid projection token: "
                    f"{placeholder}"
                )
            source = source.replace(placeholder, json.dumps(value))
        return source

    @staticmethod
    def _sha256(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    @classmethod
    def _launcher_inputs_sha256(cls, source: str, icon_path: Path) -> str:
        digest = hashlib.sha256()
        digest.update(source.encode("utf-8"))
        digest.update(b"\0")
        digest.update(icon_path.read_bytes())
        return digest.hexdigest()

    @staticmethod
    def _pe_subsystem(executable: Path) -> int:
        content = executable.read_bytes()
        if len(content) < 64:
            raise DesktopDeploymentError(
                f"Windows launcher is not a valid PE executable: {executable}"
            )
        pe_offset = struct.unpack_from("<I", content, 0x3C)[0]
        subsystem_offset = pe_offset + 24 + 68
        if (
            subsystem_offset + 2 > len(content)
            or content[pe_offset : pe_offset + 4] != b"PE\0\0"
        ):
            raise DesktopDeploymentError(
                f"Windows launcher is not a valid PE executable: {executable}"
            )
        return struct.unpack_from("<H", content, subsystem_offset)[0]

    @classmethod
    def _launcher_is_current(
        cls,
        *,
        launcher_path: Path,
        fingerprint_path: Path,
        inputs_sha256: str,
    ) -> bool:
        try:
            fingerprint = _WindowsLauncherFingerprint.read(fingerprint_path)
            return (
                launcher_path.is_file()
                and fingerprint.inputs_sha256 == inputs_sha256
                and fingerprint.executable_sha256
                == cls._sha256(launcher_path.read_bytes())
                and cls._pe_subsystem(launcher_path) == _WINDOWS_GUI_PE_SUBSYSTEM
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return False

    @classmethod
    def _publish_native_launcher(
        cls,
        *pairs: tuple[Path, Path],
        launcher_path: Path,
    ) -> tuple[str, ...]:
        """Publish a launcher refresh or defer it while the stable copy is live."""

        if not pairs:
            return ()
        try:
            _AtomicPathPublication(*pairs).publish()
        except OSError as error:
            if not (
                getattr(error, "winerror", None) in cls._sharing_violation_codes
                and launcher_path.is_file()
                and cls._pe_subsystem(launcher_path) == _WINDOWS_GUI_PE_SUBSYSTEM
            ):
                raise
            return tuple(str(target) for _candidate, target in pairs)
        return ()

    def _compile_native_launcher(
        self,
        *,
        powershell_executable: Path,
        source: str,
        icon_path: Path,
        output_path: Path,
    ) -> None:
        source_path = output_path.with_name(
            f".{output_path.stem}.source-{uuid4().hex}.cs"
        )
        script_path = output_path.with_name(
            f".{output_path.stem}.compile-{uuid4().hex}.ps1"
        )
        source_path.write_text(source, encoding="utf-8")
        script_path.write_text(
            """param(
    [Parameter(Mandatory = $true)][string]$SourcePath,
    [Parameter(Mandatory = $true)][string]$IconPath,
    [Parameter(Mandatory = $true)][string]$OutputPath
)
$ErrorActionPreference = "Stop"
$references = @("System.dll", "System.Drawing.dll", "System.Windows.Forms.dll")
$compilerParameters = New-Object System.CodeDom.Compiler.CompilerParameters
$compilerParameters.GenerateExecutable = $true
$compilerParameters.GenerateInMemory = $false
$compilerParameters.OutputAssembly = $OutputPath
$compilerParameters.CompilerOptions = (
    '/optimize+ /target:winexe /win32icon:"{0}"' -f $IconPath
)
foreach ($reference in $references) {
    [void]$compilerParameters.ReferencedAssemblies.Add($reference)
}
Add-Type -Path $SourcePath -CompilerParameters $compilerParameters
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
                    "-SourcePath",
                    str(source_path),
                    "-IconPath",
                    str(icon_path),
                    "-OutputPath",
                    str(output_path),
                ],
            )
            if self._pe_subsystem(output_path) != _WINDOWS_GUI_PE_SUBSYSTEM:
                raise DesktopDeploymentError(
                    "The compiled OpenHCS launcher is not a GUI-subsystem executable."
                )
        except Exception:
            output_path.unlink(missing_ok=True)
            raise
        finally:
            source_path.unlink(missing_ok=True)
            script_path.unlink(missing_ok=True)

    @staticmethod
    def _run_powershell(
        powershell_executable: Path,
        arguments: list[str],
    ) -> subprocess.CompletedProcess[str]:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        completed = subprocess.run(
            [str(powershell_executable), *arguments],
            check=False,
            capture_output=True,
            text=True,
            creationflags=creationflags,
        )
        if completed.returncode:
            diagnostic = "\n".join(
                output.strip()
                for output in (completed.stdout, completed.stderr)
                if output.strip()
            )
            raise DesktopDeploymentError(
                "Windows PowerShell desktop deployment failed with exit code "
                f"{completed.returncode}.\n{diagnostic}"
            )
        return completed

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
        gui_executable = (
            context.environment_root
            / "Scripts"
            / (f"{context.application.gui_entry_point}.exe")
        )
        entry_executable = (
            context.environment_root
            / "Scripts"
            / (f"{context.application.command_entry_point}.exe")
        )
        icon_path = brand_asset_path(BrandAsset.WINDOWS_ICON)
        for required_path in (gui_executable, entry_executable, icon_path):
            if not required_path.is_file():
                raise DesktopDeploymentError(
                    f"Installed desktop resource is unavailable: {required_path}"
                )
        if self._pe_subsystem(gui_executable) != _WINDOWS_GUI_PE_SUBSYSTEM:
            raise DesktopDeploymentError(
                "The installed GUI entry point is not a GUI-subsystem executable: "
                f"{gui_executable}"
            )

        desktop_directory = self._desktop_directory(powershell_executable)
        desktop_directory.mkdir(parents=True, exist_ok=True)
        shortcut_path = desktop_directory / (f"{context.application.product_name}.lnk")
        application_launcher_path = (
            context.install_root / self._application_launcher_name
        )
        current_environment_pointer = (
            context.install_root / self._current_environment_pointer_name
        )
        launcher_fingerprint_path = (
            context.install_root / self._launcher_fingerprint_name
        )
        launcher_candidate = _candidate_path(context.installation_pointer)
        shortcut_candidate = _candidate_path(shortcut_path)
        current_pointer_candidate = _candidate_path(current_environment_pointer)
        candidates = [
            launcher_candidate,
            shortcut_candidate,
            current_pointer_candidate,
        ]
        try:
            launcher_candidate.write_text(
                self.mcp_launcher_source(
                    context,
                    powershell_executable=powershell_executable,
                ),
                encoding="utf-8-sig",
            )
            current_pointer_candidate.write_text(
                context.environment_root.name,
                encoding="utf-8",
            )

            native_source = self.native_launcher_source(
                context,
                powershell_executable=powershell_executable,
            )
            inputs_sha256 = self._launcher_inputs_sha256(native_source, icon_path)
            native_launcher_pairs: list[tuple[Path, Path]] = []
            if not self._launcher_is_current(
                launcher_path=application_launcher_path,
                fingerprint_path=launcher_fingerprint_path,
                inputs_sha256=inputs_sha256,
            ):
                application_launcher_candidate = _candidate_path(
                    application_launcher_path
                )
                fingerprint_candidate = _candidate_path(launcher_fingerprint_path)
                candidates.extend(
                    (application_launcher_candidate, fingerprint_candidate)
                )
                self._compile_native_launcher(
                    powershell_executable=powershell_executable,
                    source=native_source,
                    icon_path=icon_path,
                    output_path=application_launcher_candidate,
                )
                _WindowsLauncherFingerprint(
                    inputs_sha256=inputs_sha256,
                    executable_sha256=self._sha256(
                        application_launcher_candidate.read_bytes()
                    ),
                ).write(fingerprint_candidate)
                native_launcher_pairs.extend(
                    (
                        (application_launcher_candidate, application_launcher_path),
                        (fingerprint_candidate, launcher_fingerprint_path),
                    )
                )
            deferred_paths = self._publish_native_launcher(
                *native_launcher_pairs,
                launcher_path=application_launcher_path,
            )
            self._create_shortcut(
                powershell_executable=powershell_executable,
                shortcut_path=shortcut_candidate,
                target_path=application_launcher_path,
                working_directory=context.install_root,
                icon_path=icon_path,
                product_name=context.application.product_name,
            )
            _AtomicPathPublication(
                (launcher_candidate, context.installation_pointer),
                (shortcut_candidate, shortcut_path),
                (current_pointer_candidate, current_environment_pointer),
            ).publish()
        finally:
            for candidate in candidates:
                _remove_path(candidate)
        return DesktopDeploymentReport(
            platform=self.platform_key,
            launcher_path=str(context.installation_pointer),
            desktop_shortcut_path=str(shortcut_path),
            application_path=str(application_launcher_path),
            restart_executable=str(application_launcher_path),
            deferred_paths=deferred_paths,
        )


class MacOSDesktopDeployment(DesktopDeploymentAuthority):
    """macOS environment launcher, app bundle, and Desktop link projection."""

    platform_key = AgentRuntimePlatformKey.MACOS

    def update_candidate(
        self,
        context: DesktopDeploymentContext,
        *,
        transaction_id: str | None = None,
    ) -> DesktopEnvironmentCandidate:
        return DesktopEnvironmentCandidate.under(
            context.environment_root.parent,
            Path("bin") / "python",
            transaction_id=transaction_id,
        )

    @staticmethod
    def application_path(context: DesktopDeploymentContext) -> Path:
        return (
            context.home / "Applications" / (f"{context.application.product_name}.app")
        )

    @classmethod
    def application_launcher_path(cls, context: DesktopDeploymentContext) -> Path:
        return cls.application_path(context) / "Contents" / "MacOS" / "launch-openhcs"

    @classmethod
    def environment_launcher_source(cls, context: DesktopDeploymentContext) -> str:
        stable_launcher = context.installation_pointer / "launch-openhcs.sh"
        stable_command = json.dumps(
            [str(stable_launcher), "mcp"], separators=(",", ":")
        )
        entry_point = (
            context.environment_root / "bin" / context.application.command_entry_point
        )
        return "\n".join(
            (
                "#!/bin/bash",
                "set -euo pipefail",
                "export OPENHCS_CPU_ONLY=true",
                (
                    f"export {OpenHCSProcessEnvironment.numba_cache_key}="
                    f"{shlex.quote(str(cls.numba_cache_path()))}"
                ),
                (
                    f"export {_UV_EXECUTABLE_ENVIRONMENT_VARIABLE}="
                    f"{shlex.quote(str(context.uv_executable))}"
                ),
                (
                    f"export {DESKTOP_RESTART_EXECUTABLE_ENVIRONMENT_VARIABLE}="
                    f"{shlex.quote(str(cls.application_launcher_path(context)))}"
                ),
                (
                    f"export {MCP_STABLE_LAUNCH_COMMAND_ENVIRONMENT_VARIABLE}="
                    f"{shlex.quote(stable_command)}"
                ),
                (
                    f"export {MCP_INSTALLATION_POINTER_ENVIRONMENT_VARIABLE}="
                    f"{shlex.quote(str(context.installation_pointer))}"
                ),
                f'exec {shlex.quote(str(entry_point))} "$@"',
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
            context.environment_root / "bin" / context.application.command_entry_point
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
        application_path = self.application_path(context)
        desktop_link = desktop_directory / (f"{context.application.product_name}.app")
        if _path_exists(desktop_link) and not desktop_link.is_symlink():
            raise DesktopDeploymentError(
                "Refusing to replace a non-link Desktop item: " f"{desktop_link}"
            )

        environment_launcher = context.environment_root / "launch-openhcs.sh"
        launcher_candidate = _candidate_path(environment_launcher)
        application_candidate = _candidate_path(application_path)
        desktop_candidate = _candidate_path(desktop_link)
        pointer_candidate = _candidate_path(context.installation_pointer)
        candidates = (
            launcher_candidate,
            application_candidate,
            desktop_candidate,
            pointer_candidate,
        )
        try:
            launcher_candidate.write_text(
                self.environment_launcher_source(context),
                encoding="utf-8",
            )
            launcher_candidate.chmod(0o755)
            self._prepare_application(
                application_candidate,
                context=context,
            )
            os.utime(application_candidate, None)
            desktop_candidate.symlink_to(application_path)
            pointer_candidate.symlink_to(context.environment_root)
            _AtomicPathPublication(
                (launcher_candidate, environment_launcher),
                (application_candidate, application_path),
                (desktop_candidate, desktop_link),
                (pointer_candidate, context.installation_pointer),
            ).publish()
        finally:
            for candidate in candidates:
                _discard_transaction_path(candidate)
        return DesktopDeploymentReport(
            platform=self.platform_key,
            launcher_path=str(environment_launcher),
            desktop_shortcut_path=str(desktop_link),
            application_path=str(application_path),
            restart_executable=str(self.application_launcher_path(context)),
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
        raw_pointer = os.environ.get(MCP_INSTALLATION_POINTER_ENVIRONMENT_VARIABLE)
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
