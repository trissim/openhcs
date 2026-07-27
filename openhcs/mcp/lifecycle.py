"""Local MCP process-generation and reconnect semantics.

The MCP client owns a local stdio server's process and initialized transport.
Consequently an active server cannot transparently replace itself after its
source or installed environment changes. This module owns the recoverable
boundary: detect process-generation drift and identify the stable command that
the client can reconnect through to select the current installation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import errno
from enum import Enum
from hashlib import sha256
import json
import os
from pathlib import Path
import stat

from openhcs.mcp.bootstrap import (
    MCP_INSTALLATION_POINTER_ENVIRONMENT_VARIABLE,
    MCP_STABLE_LAUNCH_COMMAND_ENVIRONMENT_VARIABLE,
)

_MAX_INSTALLATION_POINTER_BYTES = 64 * 1024


def _same_pointer_version(
    before: os.stat_result,
    after: os.stat_result,
) -> bool:
    """Compare only fields that identify the pointer content generation."""
    return (
        before.st_mode,
        before.st_dev,
        before.st_ino,
        before.st_mtime_ns,
        before.st_size,
    ) == (
        after.st_mode,
        after.st_dev,
        after.st_ino,
        after.st_mtime_ns,
        after.st_size,
    )


class McpLifecycleConfigurationError(ValueError):
    """The launcher supplied an invalid local MCP lifecycle contract."""


class McpProcessRecoveryReason(str, Enum):
    """Why the running MCP process must be replaced."""

    CURRENT = "current"
    SOURCE_CHANGED = "source_changed"
    INSTALLATION_CHANGED = "installation_changed"
    SOURCE_AND_INSTALLATION_CHANGED = "source_and_installation_changed"

    @classmethod
    def from_drift(
        cls,
        *,
        source_changed: bool,
        installation_changed: bool,
    ) -> "McpProcessRecoveryReason":
        if source_changed and installation_changed:
            return cls.SOURCE_AND_INSTALLATION_CHANGED
        if source_changed:
            return cls.SOURCE_CHANGED
        if installation_changed:
            return cls.INSTALLATION_CHANGED
        return cls.CURRENT


class McpReconnectOwner(str, Enum):
    """Protocol participant responsible for opening a new stdio session."""

    CLIENT = "mcp_client"


@dataclass(frozen=True, slots=True)
class McpInstallationPointerSnapshot:
    """Identity of the installer-owned current-environment pointer."""

    exists: bool
    kind: str | None
    device: int | None
    inode: int | None
    mtime_ns: int | None
    size: int | None
    content_sha256: str | None

    @classmethod
    def from_path(cls, pointer_path: Path) -> "McpInstallationPointerSnapshot":
        for _attempt in range(2):
            try:
                stat_result = pointer_path.lstat()
                if stat.S_ISLNK(stat_result.st_mode):
                    kind = "symlink"
                    content = os.readlink(pointer_path).encode(
                        encoding="utf-8",
                        errors="surrogateescape",
                    )
                elif stat.S_ISREG(stat_result.st_mode):
                    kind = "file"
                    with pointer_path.open("rb") as pointer_file:
                        content = pointer_file.read(_MAX_INSTALLATION_POINTER_BYTES + 1)
                    if len(content) > _MAX_INSTALLATION_POINTER_BYTES:
                        raise McpLifecycleConfigurationError(
                            "The MCP installation pointer exceeds "
                            f"{_MAX_INSTALLATION_POINTER_BYTES} bytes: {pointer_path}"
                        )
                else:
                    kind = "other"
                    content = b""
                if not _same_pointer_version(stat_result, pointer_path.lstat()):
                    continue
            except OSError as exc:
                if exc.errno not in (errno.ENOENT, errno.EINVAL, errno.ESTALE):
                    raise
                continue

            return cls(
                exists=True,
                kind=kind,
                device=stat_result.st_dev,
                inode=stat_result.st_ino,
                mtime_ns=stat_result.st_mtime_ns,
                size=stat_result.st_size,
                content_sha256=sha256(content).hexdigest(),
            )

        return cls(
            exists=False,
            kind=None,
            device=None,
            inode=None,
            mtime_ns=None,
            size=None,
            content_sha256=None,
        )


@dataclass(frozen=True, slots=True)
class McpProcessRecoveryStatus:
    """One current recovery decision for the running local MCP process."""

    reason: McpProcessRecoveryReason
    installation_pointer_path: str | None
    installation_pointer_changed_since_import: bool
    installation_pointer_available: bool | None
    restart_required: bool
    restart_command: tuple[str, ...]
    restart_command_is_stable: bool
    reconnect_required: bool
    reconnect_owner: McpReconnectOwner | None
    retry_after_reconnect: bool
    automatic_recovery_on_reconnect: bool
    hint: str | None


def _validated_absolute_path(raw_path: str, *, variable_name: str) -> Path:
    if not raw_path or "\x00" in raw_path:
        raise McpLifecycleConfigurationError(
            f"{variable_name} must contain one non-empty absolute path."
        )
    path = Path(raw_path)
    if not path.is_absolute():
        raise McpLifecycleConfigurationError(
            f"{variable_name} must contain an absolute path: {raw_path!r}"
        )
    return path


def _stable_launch_command(
    environment: Mapping[str, str],
) -> tuple[str, ...] | None:
    encoded_command = environment.get(MCP_STABLE_LAUNCH_COMMAND_ENVIRONMENT_VARIABLE)
    if encoded_command is None:
        return None
    try:
        decoded_command = json.loads(encoded_command)
    except json.JSONDecodeError as exc:
        raise McpLifecycleConfigurationError(
            f"{MCP_STABLE_LAUNCH_COMMAND_ENVIRONMENT_VARIABLE} must be a JSON array."
        ) from exc
    if (
        not isinstance(decoded_command, list)
        or not decoded_command
        or any(
            not isinstance(argument, str) or not argument or "\x00" in argument
            for argument in decoded_command
        )
    ):
        raise McpLifecycleConfigurationError(
            f"{MCP_STABLE_LAUNCH_COMMAND_ENVIRONMENT_VARIABLE} must be a "
            "non-empty JSON array of non-empty strings."
        )
    _validated_absolute_path(
        decoded_command[0],
        variable_name=MCP_STABLE_LAUNCH_COMMAND_ENVIRONMENT_VARIABLE,
    )
    return tuple(decoded_command)


@dataclass(frozen=True, slots=True)
class McpProcessLifecycle:
    """Imported process generation and client-owned recovery authority."""

    fallback_restart_command: tuple[str, ...]
    stable_launch_command: tuple[str, ...] | None
    installation_pointer_path: Path | None
    installation_pointer_at_import: McpInstallationPointerSnapshot | None

    @classmethod
    def from_environment(
        cls,
        *,
        fallback_restart_command: Sequence[str],
        environment: Mapping[str, str] | None = None,
    ) -> "McpProcessLifecycle":
        current_environment = os.environ if environment is None else environment
        normalized_fallback = tuple(fallback_restart_command)
        if not normalized_fallback or any(
            not isinstance(argument, str) or not argument or "\x00" in argument
            for argument in normalized_fallback
        ):
            raise McpLifecycleConfigurationError(
                "fallback_restart_command must contain non-empty strings."
            )
        stable_command = _stable_launch_command(current_environment)
        raw_pointer_path = current_environment.get(
            MCP_INSTALLATION_POINTER_ENVIRONMENT_VARIABLE
        )
        pointer_path = (
            None
            if raw_pointer_path is None
            else _validated_absolute_path(
                raw_pointer_path,
                variable_name=MCP_INSTALLATION_POINTER_ENVIRONMENT_VARIABLE,
            )
        )
        pointer_snapshot = (
            None
            if pointer_path is None
            else McpInstallationPointerSnapshot.from_path(pointer_path)
        )
        return cls(
            fallback_restart_command=normalized_fallback,
            stable_launch_command=stable_command,
            installation_pointer_path=pointer_path,
            installation_pointer_at_import=pointer_snapshot,
        )

    def recovery_status(
        self,
        *,
        source_changed: bool,
    ) -> McpProcessRecoveryStatus:
        """Return reconnect semantics for current source/installation drift."""
        current_pointer_snapshot = (
            None
            if self.installation_pointer_path is None
            else McpInstallationPointerSnapshot.from_path(
                self.installation_pointer_path
            )
        )
        installation_changed = (
            self.installation_pointer_at_import is not None
            and current_pointer_snapshot != self.installation_pointer_at_import
        )
        reason = McpProcessRecoveryReason.from_drift(
            source_changed=source_changed,
            installation_changed=installation_changed,
        )
        restart_required = reason is not McpProcessRecoveryReason.CURRENT
        command_is_stable = self.stable_launch_command is not None
        restart_command = (
            self.stable_launch_command
            if command_is_stable
            else self.fallback_restart_command
        )
        pointer_available = (
            None
            if current_pointer_snapshot is None
            else current_pointer_snapshot.exists
        )
        automatic_recovery = (
            restart_required and command_is_stable and pointer_available is not False
        )
        hint = None
        if restart_required:
            command_description = (
                "The stable launcher will select the current OpenHCS environment."
                if command_is_stable
                else "The reported command belongs to this process environment."
            )
            hint = (
                "Restart the MCP client/server process using restart_command. "
                "The MCP client owns this stdio process and must close it, launch "
                "the command, complete a new MCP initialize handshake, and only "
                "then retry the blocked operation. "
                f"{command_description}"
            )
        return McpProcessRecoveryStatus(
            reason=reason,
            installation_pointer_path=(
                None
                if self.installation_pointer_path is None
                else str(self.installation_pointer_path)
            ),
            installation_pointer_changed_since_import=installation_changed,
            installation_pointer_available=pointer_available,
            restart_required=restart_required,
            restart_command=restart_command if restart_required else (),
            restart_command_is_stable=command_is_stable,
            reconnect_required=restart_required,
            reconnect_owner=(McpReconnectOwner.CLIENT if restart_required else None),
            retry_after_reconnect=restart_required,
            automatic_recovery_on_reconnect=automatic_recovery,
            hint=hint,
        )
