"""Nominal host-platform authorities for headless agent runtime paths."""

from __future__ import annotations

import os
import sys
import tempfile
from abc import ABC
from enum import Enum
from pathlib import Path
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

from openhcs.core.registry_strategies import EnumKeyedStrategyMixin


class AgentRuntimePlatformKey(str, Enum):
    """Closed host-platform axis used by agent runtime authorities."""

    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"
    POSIX = "posix"
    OTHER = "other"

    @classmethod
    def current(cls) -> "AgentRuntimePlatformKey":
        """Return the nominal platform key for the running interpreter."""
        if sys.platform.startswith("win"):
            return cls.WINDOWS
        if sys.platform == "darwin":
            return cls.MACOS
        if sys.platform.startswith("linux"):
            return cls.LINUX
        if os.name == "posix":
            return cls.POSIX
        return cls.OTHER


class AgentRuntimePlatformAuthority(
    EnumKeyedStrategyMixin[AgentRuntimePlatformKey],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered authority for platform-owned agent filesystem semantics."""

    __enum_member_attr__ = "platform_key"
    platform_key: ClassVar[AgentRuntimePlatformKey]

    @classmethod
    def current(cls) -> "AgentRuntimePlatformAuthority":
        """Instantiate the authority declared for the current platform."""
        return cls.for_enum_member(AgentRuntimePlatformKey.current())

    @staticmethod
    def resolved_path(path: str | Path) -> Path:
        """Resolve environment and user markers without requiring existence."""
        expanded = os.path.expandvars(os.fspath(path))
        return Path(expanded).expanduser().resolve(strict=False)

    @staticmethod
    def process_exists(pid: int) -> bool:
        """Return whether a local process id is live on this host."""
        import psutil

        return psutil.pid_exists(pid)

    @staticmethod
    def temporary_root() -> Path:
        """Return the platform temporary directory."""
        return Path(tempfile.gettempdir()).resolve(strict=False)

    def current_user_id(self) -> int | None:
        """Return the POSIX user id when the platform defines one."""
        return None

    def supports_posix_permissions(self) -> bool:
        """Return whether descriptor ownership and mode checks are meaningful."""
        return False

    def application_data_root(self, application_name: str) -> Path:
        """Return the user-scoped persistent data root for an application."""
        return (Path.home() / ".local" / "share" / application_name.casefold()).resolve(
            strict=False
        )

    def _application_runtime_candidates(
        self,
        application_name: str,
        component: Path,
    ) -> tuple[Path, ...]:
        del application_name, component
        return ()

    def application_runtime_dirs(
        self,
        application_name: str,
        component: str | Path,
    ) -> tuple[Path, ...]:
        """Return ordered user-scoped runtime directories for a component."""
        component_path = Path(component)
        candidates = list(
            self._application_runtime_candidates(
                application_name,
                component_path,
            )
        )
        temporary_name_parts = (
            application_name.casefold(),
            *component_path.parts,
        )
        temporary_name = "-".join(temporary_name_parts)
        user_id = self.current_user_id()
        if user_id is not None:
            temporary_name = f"{temporary_name}-{user_id}"
        candidates.append(self.temporary_root() / temporary_name)
        return tuple(dict.fromkeys(candidates))


class OtherAgentRuntimePlatformAuthority(AgentRuntimePlatformAuthority):
    """Portable conservative defaults for unknown host platforms."""

    platform_key = AgentRuntimePlatformKey.OTHER


class PosixAgentRuntimePlatformMixin:
    """Reusable POSIX runtime ownership and XDG path semantics."""

    def current_user_id(self) -> int | None:
        getuid = getattr(os, "getuid", None)
        return getuid() if getuid is not None else None

    def supports_posix_permissions(self) -> bool:
        return self.current_user_id() is not None

    def application_data_root(self, application_name: str) -> Path:
        configured = os.environ.get("XDG_DATA_HOME")
        data_home = (
            self.resolved_path(configured)
            if configured
            else (Path.home() / ".local" / "share").resolve(strict=False)
        )
        return data_home / application_name.casefold()

    def _application_runtime_candidates(
        self,
        application_name: str,
        component: Path,
    ) -> tuple[Path, ...]:
        configured = os.environ.get("XDG_RUNTIME_DIR")
        if not configured:
            return ()
        return (
            self.resolved_path(configured) / application_name.casefold() / component,
        )


class PosixAgentRuntimePlatformAuthority(
    PosixAgentRuntimePlatformMixin,
    AgentRuntimePlatformAuthority,
):
    """Registered authority for otherwise-unclassified POSIX hosts."""

    platform_key = AgentRuntimePlatformKey.POSIX


class LinuxAgentRuntimePlatformAuthority(
    PosixAgentRuntimePlatformMixin,
    AgentRuntimePlatformAuthority,
):
    """Linux runtime paths, including the standard per-user runtime root."""

    platform_key = AgentRuntimePlatformKey.LINUX

    def _application_runtime_candidates(
        self,
        application_name: str,
        component: Path,
    ) -> tuple[Path, ...]:
        candidates = list(
            super()._application_runtime_candidates(application_name, component)
        )
        user_id = self.current_user_id()
        if user_id is not None:
            candidates.append(
                Path("/run/user")
                / str(user_id)
                / application_name.casefold()
                / component
            )
        return tuple(candidates)


class MacOSAgentRuntimePlatformAuthority(
    PosixAgentRuntimePlatformMixin,
    AgentRuntimePlatformAuthority,
):
    """macOS application-data convention plus POSIX runtime ownership."""

    platform_key = AgentRuntimePlatformKey.MACOS

    def application_data_root(self, application_name: str) -> Path:
        return (
            Path.home() / "Library" / "Application Support" / application_name
        ).resolve(strict=False)


class WindowsAgentRuntimePlatformAuthority(AgentRuntimePlatformAuthority):
    """Windows user-data and local runtime path conventions."""

    platform_key = AgentRuntimePlatformKey.WINDOWS

    def _local_data_home(self) -> Path:
        configured = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if configured:
            return self.resolved_path(configured)
        return (Path.home() / "AppData" / "Local").resolve(strict=False)

    def application_data_root(self, application_name: str) -> Path:
        return self._local_data_home() / application_name

    def _application_runtime_candidates(
        self,
        application_name: str,
        component: Path,
    ) -> tuple[Path, ...]:
        return (self._local_data_home() / application_name / "runtime" / component,)
