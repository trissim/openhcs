"""Register the installed OpenHCS MCP server with local agent clients.

The native installers provide one stable launcher command.  This module projects
that command into each detected client's authoritative local configuration
surface without exposing the installer's versioned environment layout.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

import tomlkit
from metaclass_registry import AutoRegisterMeta
from tomlkit.items import InlineTable

from openhcs.agent.runtime_platform import AgentRuntimePlatformKey
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin

CLIENT_REGISTRATION_SCHEMA_VERSION = "openhcs.mcp.client-registration.v1"
OPENHCS_MCP_SERVER_NAME = "openhcs"
_CONFIG_MODE = 0o600


class ClientRegistrationStatus(str, Enum):
    """Stable installer-facing result status."""

    REGISTERED = "registered"
    UPDATED = "updated"
    UNCHANGED = "unchanged"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class McpLauncherSpec:
    """Stable stdio launcher supplied by a native installer."""

    command: str
    arguments: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        command_path = Path(self.command).expanduser()
        if not command_path.is_absolute():
            raise ValueError("MCP launcher command must be an absolute path.")
        if not all(isinstance(argument, str) for argument in self.arguments):
            raise TypeError("MCP launcher arguments must all be strings.")
        object.__setattr__(self, "command", os.fspath(command_path))

    def stdio_server_entry(self) -> dict[str, Any]:
        """Return the shared strict JSON representation for stdio clients."""
        return {
            "command": self.command,
            "args": list(self.arguments),
        }


ProcessRunner = Callable[..., subprocess.CompletedProcess[str]]
ExecutableResolver = Callable[[str], str | None]


@dataclass(frozen=True, slots=True)
class ClientRegistrationEnvironment:
    """Injectable host boundary used by registered client targets."""

    home: Path
    environ: Mapping[str, str]
    platform_key: AgentRuntimePlatformKey
    executable_resolver: ExecutableResolver
    process_runner: ProcessRunner

    @classmethod
    def current(cls) -> "ClientRegistrationEnvironment":
        """Capture the current user's client-registration environment."""
        return cls(
            home=Path.home(),
            environ=dict(os.environ),
            platform_key=AgentRuntimePlatformKey.current(),
            executable_resolver=shutil.which,
            process_runner=subprocess.run,
        )

    def executable(self, candidates: Sequence[str]) -> str | None:
        """Resolve the first usable executable from a client's candidates."""
        for candidate in candidates:
            resolved = self.executable_resolver(candidate)
            if resolved:
                return resolved
        return None


class ClaudeDesktopPlatformSemanticsABC(
    EnumKeyedStrategyMixin[AgentRuntimePlatformKey],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered owner of Claude Desktop paths for one supported platform."""

    __enum_member_attr__ = "platform_key"
    platform_key: ClassVar[AgentRuntimePlatformKey]

    @abstractmethod
    def config_path(self, environment: ClientRegistrationEnvironment) -> Path:
        """Return the documented per-user Claude Desktop configuration path."""
        raise NotImplementedError

    @abstractmethod
    def installation_paths(
        self,
        environment: ClientRegistrationEnvironment,
    ) -> tuple[Path, ...]:
        """Return stable desktop-app paths, excluding Claude Code CLI paths."""
        raise NotImplementedError

    def desktop_app_installed(
        self,
        environment: ClientRegistrationEnvironment,
    ) -> bool:
        """Return whether a declared desktop application path exists."""
        return any(path.exists() for path in self.installation_paths(environment))


class WindowsClaudeDesktopPlatformSemantics(ClaudeDesktopPlatformSemanticsABC):
    """Claude Desktop configuration and installation paths on Windows."""

    platform_key = AgentRuntimePlatformKey.WINDOWS

    def config_path(self, environment: ClientRegistrationEnvironment) -> Path:
        app_data = environment.environ.get("APPDATA")
        if app_data:
            return Path(app_data) / "Claude" / "claude_desktop_config.json"
        return (
            environment.home
            / "AppData"
            / "Roaming"
            / "Claude"
            / "claude_desktop_config.json"
        )

    def installation_paths(
        self,
        environment: ClientRegistrationEnvironment,
    ) -> tuple[Path, ...]:
        candidates: list[Path] = []
        local_app_data = environment.environ.get("LOCALAPPDATA")
        if local_app_data:
            local_root = Path(local_app_data)
            candidates.extend(
                (
                    local_root / "AnthropicClaude" / "Claude.exe",
                    local_root / "Programs" / "Claude" / "Claude.exe",
                )
            )
        program_files = environment.environ.get("PROGRAMFILES")
        if program_files:
            candidates.append(Path(program_files) / "Claude" / "Claude.exe")
        program_files_x86 = environment.environ.get("PROGRAMFILES(X86)")
        if program_files_x86:
            candidates.append(Path(program_files_x86) / "Claude" / "Claude.exe")
        return tuple(candidates)

    def desktop_app_installed(
        self,
        environment: ClientRegistrationEnvironment,
    ) -> bool:
        if super().desktop_app_installed(environment):
            return True
        local_app_data = environment.environ.get("LOCALAPPDATA")
        if not local_app_data:
            return False
        packages_root = Path(local_app_data) / "Packages"
        return packages_root.is_dir() and any(
            package_path.is_dir() for package_path in packages_root.glob("Claude_*")
        )


class MacOSClaudeDesktopPlatformSemantics(ClaudeDesktopPlatformSemanticsABC):
    """Claude Desktop configuration and installation paths on macOS."""

    platform_key = AgentRuntimePlatformKey.MACOS

    def config_path(self, environment: ClientRegistrationEnvironment) -> Path:
        return (
            environment.home
            / "Library"
            / "Application Support"
            / "Claude"
            / "claude_desktop_config.json"
        )

    def installation_paths(
        self,
        environment: ClientRegistrationEnvironment,
    ) -> tuple[Path, ...]:
        return (
            Path("/Applications/Claude.app"),
            environment.home / "Applications" / "Claude.app",
        )


@dataclass(frozen=True, slots=True)
class ClientConfigMutation:
    """Leaf-owned configuration mutation without duplicated client identity."""

    status: ClientRegistrationStatus
    config_path: str | None
    backup_path: str | None
    message: str


@dataclass(frozen=True, slots=True)
class ClientRegistrationResult:
    """Structured result for one explicitly requested or detected target."""

    target_id: str
    display_name: str
    status: str
    required: bool
    detected: bool
    config_path: str | None
    backup_path: str | None
    message: str

    @classmethod
    def from_mutation(
        cls,
        target: type["McpClientRegistrationTarget"],
        mutation: ClientConfigMutation,
        *,
        required: bool,
        detected: bool,
    ) -> "ClientRegistrationResult":
        """Attach the owning target identity to a leaf mutation."""
        return cls(
            target_id=target.require_target_id(),
            display_name=target.display_name,
            status=mutation.status.value,
            required=required,
            detected=detected,
            config_path=mutation.config_path,
            backup_path=mutation.backup_path,
            message=mutation.message,
        )

    @classmethod
    def failure(
        cls,
        *,
        target_id: str,
        display_name: str,
        required: bool,
        detected: bool,
        message: str,
        config_path: Path | None = None,
    ) -> "ClientRegistrationResult":
        """Construct a failure result at an orchestration boundary."""
        return cls(
            target_id=target_id,
            display_name=display_name,
            status=ClientRegistrationStatus.FAILED.value,
            required=required,
            detected=detected,
            config_path=os.fspath(config_path) if config_path is not None else None,
            backup_path=None,
            message=message,
        )


@dataclass(frozen=True, slots=True)
class ClientRegistrationReport:
    """Installer-facing aggregate with explicit partial-success semantics."""

    schema_version: str
    ok: bool
    required_ok: bool
    results: tuple[ClientRegistrationResult, ...]

    @classmethod
    def from_results(
        cls,
        results: Sequence[ClientRegistrationResult],
    ) -> "ClientRegistrationReport":
        """Summarize all attempted registrations without erasing successes."""
        frozen_results = tuple(results)
        failed = tuple(
            result
            for result in frozen_results
            if result.status == ClientRegistrationStatus.FAILED.value
        )
        return cls(
            schema_version=CLIENT_REGISTRATION_SCHEMA_VERSION,
            ok=not failed,
            required_ok=not any(result.required for result in failed),
            results=frozen_results,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-native installer payload."""
        return {
            "schema_version": self.schema_version,
            "ok": self.ok,
            "required_ok": self.required_ok,
            "results": [asdict(result) for result in self.results],
        }


class ClientConfigFormatError(ValueError):
    """A client configuration cannot be safely updated."""


class McpClientRegistrationTarget(ABC, metaclass=AutoRegisterMeta):
    """True owner for local MCP client detection and configuration."""

    __registry__: ClassVar[dict[str, type["McpClientRegistrationTarget"]]] = {}
    __registry_key__ = "target_id"
    __skip_if_no_key__ = True
    __registry_name__ = "MCP client registration target"

    target_id: ClassVar[str | None] = None
    display_name: ClassVar[str] = ""

    @classmethod
    def require_target_id(cls) -> str:
        """Return the declared stable target identity."""
        if cls.target_id is None:
            raise ValueError(f"{cls.__name__} must declare target_id.")
        return cls.target_id

    @classmethod
    def registered_targets(
        cls,
    ) -> tuple[type["McpClientRegistrationTarget"], ...]:
        """Return all concrete client targets in declaration order."""
        return tuple(cls.__registry__.values())

    @classmethod
    def target_for_id(
        cls,
        target_id: str,
    ) -> type["McpClientRegistrationTarget"] | None:
        """Resolve a target through the root registry."""
        return cls.__registry__.get(target_id)

    @classmethod
    @abstractmethod
    def detected(cls, environment: ClientRegistrationEnvironment) -> bool:
        """Return whether this local client is installed or configured."""

    @classmethod
    @abstractmethod
    def register(
        cls,
        launcher: McpLauncherSpec,
        environment: ClientRegistrationEnvironment,
    ) -> ClientConfigMutation:
        """Project the stable launcher into this client's owned surface."""

    @classmethod
    def diagnostic_config_path(
        cls,
        environment: ClientRegistrationEnvironment,
    ) -> Path | None:
        """Return a useful path for failure diagnostics when one exists."""
        del environment
        return None


class CodexClientRegistrationTarget(McpClientRegistrationTarget):
    """ChatGPT desktop and Codex shared TOML configuration."""

    target_id = "codex"
    display_name = "ChatGPT desktop and OpenAI Codex"
    executable_candidates = ("codex",)

    @classmethod
    def config_path(cls, environment: ClientRegistrationEnvironment) -> Path:
        """Return the shared ChatGPT desktop and Codex configuration path."""
        configured_home = environment.environ.get("CODEX_HOME")
        codex_home = (
            Path(configured_home).expanduser()
            if configured_home
            else environment.home / ".codex"
        )
        return codex_home / "config.toml"

    @classmethod
    def diagnostic_config_path(
        cls,
        environment: ClientRegistrationEnvironment,
    ) -> Path:
        return cls.config_path(environment)

    @classmethod
    def detected(cls, environment: ClientRegistrationEnvironment) -> bool:
        config_path = cls.config_path(environment)
        return (
            config_path.exists()
            or config_path.parent.exists()
            or environment.executable(cls.executable_candidates) is not None
        )

    @classmethod
    def register(
        cls,
        launcher: McpLauncherSpec,
        environment: ClientRegistrationEnvironment,
    ) -> ClientConfigMutation:
        return _update_codex_toml(cls.config_path(environment), launcher)


class ClaudeDesktopClientRegistrationTarget(McpClientRegistrationTarget):
    """Claude Desktop strict JSON configuration."""

    target_id = "claude-desktop"
    display_name = "Claude Desktop"

    @staticmethod
    def _platform_semantics(
        environment: ClientRegistrationEnvironment,
    ) -> ClaudeDesktopPlatformSemanticsABC | None:
        try:
            return ClaudeDesktopPlatformSemanticsABC.for_enum_member(
                environment.platform_key
            )
        except KeyError:
            return None

    @classmethod
    def config_path(
        cls,
        environment: ClientRegistrationEnvironment,
    ) -> Path | None:
        """Return the documented per-user Claude Desktop configuration."""
        semantics = cls._platform_semantics(environment)
        return None if semantics is None else semantics.config_path(environment)

    @classmethod
    def diagnostic_config_path(
        cls,
        environment: ClientRegistrationEnvironment,
    ) -> Path | None:
        return cls.config_path(environment)

    @classmethod
    def installation_paths(
        cls,
        environment: ClientRegistrationEnvironment,
    ) -> tuple[Path, ...]:
        """Return stable desktop-app paths, never Claude Code CLI paths."""
        semantics = cls._platform_semantics(environment)
        return () if semantics is None else semantics.installation_paths(environment)

    @classmethod
    def desktop_app_installed(
        cls,
        environment: ClientRegistrationEnvironment,
    ) -> bool:
        """Detect a desktop app, including the documented Windows MSIX form."""
        semantics = cls._platform_semantics(environment)
        return (
            False if semantics is None else semantics.desktop_app_installed(environment)
        )

    @classmethod
    def detected(cls, environment: ClientRegistrationEnvironment) -> bool:
        config_path = cls.config_path(environment)
        return (
            config_path is not None and config_path.exists()
        ) or cls.desktop_app_installed(environment)

    @classmethod
    def register(
        cls,
        launcher: McpLauncherSpec,
        environment: ClientRegistrationEnvironment,
    ) -> ClientConfigMutation:
        config_path = cls.config_path(environment)
        if config_path is None:
            raise RuntimeError(
                "Claude Desktop local MCP configuration is supported on "
                "Windows and macOS."
            )
        return _update_json_mcp_servers(config_path, launcher)


class HomeJsonMcpClientRegistrationMixin:
    """Shared leaf hooks for documented home-relative strict JSON clients."""

    config_relative_path: ClassVar[Path]
    executable_candidates: ClassVar[tuple[str, ...]]

    @classmethod
    def config_path(cls, environment: ClientRegistrationEnvironment) -> Path:
        return environment.home / cls.config_relative_path

    @classmethod
    def diagnostic_config_path(
        cls,
        environment: ClientRegistrationEnvironment,
    ) -> Path:
        return cls.config_path(environment)

    @classmethod
    def detected(cls, environment: ClientRegistrationEnvironment) -> bool:
        config_path = cls.config_path(environment)
        return (
            config_path.exists()
            or config_path.parent.exists()
            or environment.executable(cls.executable_candidates) is not None
        )

    @classmethod
    def register(
        cls,
        launcher: McpLauncherSpec,
        environment: ClientRegistrationEnvironment,
    ) -> ClientConfigMutation:
        return _update_json_mcp_servers(cls.config_path(environment), launcher)


class CursorClientRegistrationTarget(
    HomeJsonMcpClientRegistrationMixin,
    McpClientRegistrationTarget,
):
    """Cursor's documented per-user strict JSON MCP configuration."""

    target_id = "cursor"
    display_name = "Cursor"
    config_relative_path = Path(".cursor/mcp.json")
    executable_candidates = ("cursor", "Cursor")


class GeminiCliClientRegistrationTarget(
    HomeJsonMcpClientRegistrationMixin,
    McpClientRegistrationTarget,
):
    """Gemini CLI's documented user settings MCP configuration."""

    target_id = "gemini-cli"
    display_name = "Gemini CLI"
    config_relative_path = Path(".gemini/settings.json")
    executable_candidates = ("gemini",)


class WindsurfClientRegistrationTarget(
    HomeJsonMcpClientRegistrationMixin,
    McpClientRegistrationTarget,
):
    """Windsurf Cascade's documented per-user MCP configuration."""

    target_id = "windsurf"
    display_name = "Windsurf"
    config_relative_path = Path(".codeium/windsurf/mcp_config.json")
    executable_candidates = ("windsurf", "Windsurf")


class VsCodeClientRegistrationTarget(McpClientRegistrationTarget):
    """VS Code user-profile registration through its public CLI contract."""

    target_id = "vscode"
    display_name = "Visual Studio Code"
    executable_candidates = ("code", "code-insiders")

    @classmethod
    def executable(cls, environment: ClientRegistrationEnvironment) -> str | None:
        return environment.executable(cls.executable_candidates)

    @classmethod
    def detected(cls, environment: ClientRegistrationEnvironment) -> bool:
        return cls.executable(environment) is not None

    @classmethod
    def register(
        cls,
        launcher: McpLauncherSpec,
        environment: ClientRegistrationEnvironment,
    ) -> ClientConfigMutation:
        executable = cls.executable(environment)
        if executable is None:
            raise RuntimeError("No usable VS Code command-line executable was found.")
        server_definition = {
            "name": OPENHCS_MCP_SERVER_NAME,
            **launcher.stdio_server_entry(),
        }
        completed = environment.process_runner(
            [
                executable,
                "--add-mcp",
                json.dumps(server_definition, separators=(",", ":")),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(
                f"VS Code --add-mcp exited with {completed.returncode}{suffix}"
            )
        return ClientConfigMutation(
            status=ClientRegistrationStatus.REGISTERED,
            config_path=None,
            backup_path=None,
            message=f"Registered OpenHCS through {executable} --add-mcp.",
        )


def _atomic_write_bytes(path: Path, payload: bytes, *, mode: int) -> None:
    """Atomically replace one user configuration file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_config_write(
    path: Path,
    rendered: str,
    *,
    original: bytes | None,
) -> str | None:
    """Back up an existing config, then atomically publish its replacement."""
    mode = _CONFIG_MODE
    backup_path: Path | None = None
    if original is not None:
        mode = path.stat().st_mode & 0o777
        backup_path = path.with_name(f"{path.name}.openhcs.bak")
        _atomic_write_bytes(backup_path, original, mode=mode)
    _atomic_write_bytes(path, rendered.encode("utf-8"), mode=mode)
    return os.fspath(backup_path) if backup_path is not None else None


def _read_config(path: Path) -> tuple[str, bytes | None]:
    if not path.exists():
        return "", None
    original = path.read_bytes()
    try:
        return original.decode("utf-8"), original
    except UnicodeDecodeError as exc:
        raise ClientConfigFormatError(
            f"{path} is not valid UTF-8; it was left unchanged."
        ) from exc


def _toml_server_matches(existing: Any, launcher: McpLauncherSpec) -> bool:
    if not isinstance(existing, Mapping):
        return False
    if set(existing) != {"command", "args"}:
        return False
    arguments = existing.get("args")
    return (
        existing.get("command") == launcher.command
        and isinstance(arguments, list)
        and list(arguments) == list(launcher.arguments)
    )


def _update_codex_toml(
    path: Path,
    launcher: McpLauncherSpec,
) -> ClientConfigMutation:
    source, original = _read_config(path)
    try:
        document = tomlkit.parse(source)
    except Exception as exc:
        raise ClientConfigFormatError(
            f"{path} is not valid TOML; it was left unchanged: {exc}"
        ) from exc

    servers = document.get("mcp_servers")
    if servers is None:
        servers = tomlkit.table()
        document["mcp_servers"] = servers
    elif not isinstance(servers, Mapping):
        raise ClientConfigFormatError(
            f"{path} has a non-table mcp_servers value; it was left unchanged."
        )

    if _toml_server_matches(servers.get(OPENHCS_MCP_SERVER_NAME), launcher):
        return ClientConfigMutation(
            status=ClientRegistrationStatus.UNCHANGED,
            config_path=os.fspath(path),
            backup_path=None,
            message="OpenHCS is already registered with ChatGPT desktop and Codex.",
        )

    server = (
        tomlkit.inline_table() if isinstance(servers, InlineTable) else tomlkit.table()
    )
    server.add("command", launcher.command)
    server.add("args", list(launcher.arguments))
    servers[OPENHCS_MCP_SERVER_NAME] = server
    rendered = tomlkit.dumps(document)
    backup_path = _atomic_config_write(path, rendered, original=original)
    return ClientConfigMutation(
        status=(
            ClientRegistrationStatus.UPDATED
            if original is not None
            else ClientRegistrationStatus.REGISTERED
        ),
        config_path=os.fspath(path),
        backup_path=backup_path,
        message=(
            "Registered OpenHCS in the shared ChatGPT desktop and Codex "
            "MCP configuration."
        ),
    )


def _update_json_mcp_servers(
    path: Path,
    launcher: McpLauncherSpec,
) -> ClientConfigMutation:
    source, original = _read_config(path)
    try:
        document = json.loads(source) if source else {}
    except json.JSONDecodeError as exc:
        raise ClientConfigFormatError(
            f"{path} is not valid JSON; it was left unchanged: {exc}"
        ) from exc
    if not isinstance(document, dict):
        raise ClientConfigFormatError(
            f"{path} must contain a JSON object; it was left unchanged."
        )
    servers = document.get("mcpServers")
    if servers is None:
        servers = {}
        document["mcpServers"] = servers
    elif not isinstance(servers, dict):
        raise ClientConfigFormatError(
            f"{path} has a non-object mcpServers value; it was left unchanged."
        )

    desired = launcher.stdio_server_entry()
    if servers.get(OPENHCS_MCP_SERVER_NAME) == desired:
        return ClientConfigMutation(
            status=ClientRegistrationStatus.UNCHANGED,
            config_path=os.fspath(path),
            backup_path=None,
            message="OpenHCS is already registered with this client.",
        )

    servers[OPENHCS_MCP_SERVER_NAME] = desired
    rendered = f"{json.dumps(document, indent=2, ensure_ascii=False)}\n"
    backup_path = _atomic_config_write(path, rendered, original=original)
    return ClientConfigMutation(
        status=(
            ClientRegistrationStatus.UPDATED
            if original is not None
            else ClientRegistrationStatus.REGISTERED
        ),
        config_path=os.fspath(path),
        backup_path=backup_path,
        message="Registered OpenHCS in the client's MCP configuration.",
    )


def register_mcp_clients(
    launcher: McpLauncherSpec,
    *,
    required_target_ids: Sequence[str] = (),
    register_detected: bool = False,
    environment: ClientRegistrationEnvironment | None = None,
) -> ClientRegistrationReport:
    """Register required and detected clients through their nominal owners."""
    host = environment or ClientRegistrationEnvironment.current()
    results: list[ClientRegistrationResult] = []
    attempted_ids: set[str] = set()

    for target_id in dict.fromkeys(required_target_ids):
        attempted_ids.add(target_id)
        target = McpClientRegistrationTarget.target_for_id(target_id)
        if target is None:
            results.append(
                ClientRegistrationResult.failure(
                    target_id=target_id,
                    display_name=target_id,
                    required=True,
                    detected=False,
                    message=f"Unknown MCP client target: {target_id}",
                )
            )
            continue
        detected = False
        try:
            detected = target.detected(host)
            mutation = target.register(launcher, host)
            results.append(
                ClientRegistrationResult.from_mutation(
                    target,
                    mutation,
                    required=True,
                    detected=detected,
                )
            )
        except Exception as exc:
            results.append(
                ClientRegistrationResult.failure(
                    target_id=target.require_target_id(),
                    display_name=target.display_name,
                    required=True,
                    detected=detected,
                    config_path=target.diagnostic_config_path(host),
                    message=str(exc),
                )
            )

    if register_detected:
        for target in McpClientRegistrationTarget.registered_targets():
            target_id = target.require_target_id()
            if target_id in attempted_ids:
                continue
            try:
                detected = target.detected(host)
            except Exception as exc:
                results.append(
                    ClientRegistrationResult.failure(
                        target_id=target_id,
                        display_name=target.display_name,
                        required=False,
                        detected=False,
                        config_path=target.diagnostic_config_path(host),
                        message=f"Client detection failed: {exc}",
                    )
                )
                continue
            if not detected:
                continue
            try:
                mutation = target.register(launcher, host)
                results.append(
                    ClientRegistrationResult.from_mutation(
                        target,
                        mutation,
                        required=False,
                        detected=True,
                    )
                )
            except Exception as exc:
                results.append(
                    ClientRegistrationResult.failure(
                        target_id=target_id,
                        display_name=target.display_name,
                        required=False,
                        detected=True,
                        config_path=target.diagnostic_config_path(host),
                        message=str(exc),
                    )
                )

    return ClientRegistrationReport.from_results(results)


def _arguments_from_json(value: str) -> tuple[str, ...]:
    try:
        arguments = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON array: {exc}") from exc
    if not isinstance(arguments, list) or not all(
        isinstance(argument, str) for argument in arguments
    ):
        raise argparse.ArgumentTypeError("--args-json must be a JSON array of strings")
    return tuple(arguments)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Register the installed OpenHCS MCP server with local clients."
    )
    parser.add_argument(
        "--command",
        required=True,
        help="Absolute path to the installer's stable OpenHCS MCP launcher.",
    )
    launcher_arguments = parser.add_mutually_exclusive_group()
    launcher_arguments.add_argument(
        "--args-json",
        type=_arguments_from_json,
        default=None,
        help="JSON array of arguments passed to the stable launcher.",
    )
    launcher_arguments.add_argument(
        "--launcher-argument",
        action="append",
        default=None,
        help=(
            "One argument passed to the stable launcher; may be repeated. "
            "Use --launcher-argument=VALUE when VALUE begins with a dash."
        ),
    )
    parser.add_argument(
        "--register",
        action="append",
        default=[],
        metavar="TARGET",
        help="Required client target ID; may be repeated.",
    )
    parser.add_argument(
        "--register-detected",
        action="store_true",
        help="Also register clients detected on this host.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the structured installer-facing JSON report.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the native-installer-facing client registration command."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    if not arguments.register and not arguments.register_detected:
        parser.error("request at least one --register target or --register-detected")
    try:
        launcher = McpLauncherSpec(
            command=arguments.command,
            arguments=(
                arguments.args_json
                if arguments.args_json is not None
                else tuple(arguments.launcher_argument or ())
            ),
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    report = register_mcp_clients(
        launcher,
        required_target_ids=arguments.register,
        register_detected=arguments.register_detected,
    )
    payload = report.as_dict()
    if arguments.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        for result in report.results:
            print(
                f"{result.display_name}: {result.status} - {result.message}",
                file=(
                    sys.stderr
                    if result.status == ClientRegistrationStatus.FAILED.value
                    else sys.stdout
                ),
            )
    return 0 if report.required_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
