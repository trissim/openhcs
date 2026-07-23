"""MCP development client for the active OpenHCS checkout."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shlex
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TextIO

from openhcs.serialization.json import to_jsonable as to_jsonable
from openhcs.agent.dto.common import JsonObject
from openhcs.agent.capabilities import (
    FullLocalCapabilitySurfaceProfile,
    LocalCapabilitySurfaceProfile,
)
from openhcs.constants.constants import AllComponents as AllComponents
from openhcs.mcp.dev_client_core import (
    DEFAULT_CALL_TIMEOUT_SECONDS,
    DEFAULT_REGISTRY_DISCOVERY_TIMEOUT_SECONDS as DEFAULT_REGISTRY_DISCOVERY_TIMEOUT_SECONDS,
    McpDevCliUsageError,
    McpDevClientPhase as McpDevClientPhase,
    McpDevServerSpec,
    McpDevStdioSession,
    McpDevToolBatchResponse,
    McpDevToolCall,
    McpDevToolListResponse,
    McpDevToolResult as McpDevToolResult,
    McpDevTransportFailure as McpDevTransportFailure,
    WorkflowPollRowState as WorkflowPollRowState,
    WorkflowPollSummaryStatus as WorkflowPollSummaryStatus,
    WorkflowStatePollPolicy as WorkflowStatePollPolicy,
    _command_failed,
    captured_server_stderr_tail,
    parse_json_object,
    plate_manager_state_surface_tool_arguments as plate_manager_state_surface_tool_arguments,
    require_json_object_payload,
    workflow_poll_has_reached_terminal_state as workflow_poll_has_reached_terminal_state,
    workflow_poll_summary_result as workflow_poll_summary_result,
    workflow_poll_terminal_status as workflow_poll_terminal_status,
)
from openhcs.mcp.dev_client_commanding import (
    GeneratedMcpDevCommandProfile as GeneratedMcpDevCommandProfile,
    McpDevCommandSpec,
)
from openhcs.mcp import dev_client_commands as dev_client_commands
from openhcs.mcp.dev_client_rendering import (
    DEFAULT_CODE_DOCUMENT_MAX_CHARS as DEFAULT_CODE_DOCUMENT_MAX_CHARS,
)

_DECLARATION_MODULES = (dev_client_commands,)

__all__ = (
    "AllComponents",
    "DEFAULT_CODE_DOCUMENT_MAX_CHARS",
    "DEFAULT_REGISTRY_DISCOVERY_TIMEOUT_SECONDS",
    "GeneratedMcpDevCommandProfile",
    "McpDevCliUsageError",
    "McpDevClientPhase",
    "McpDevClient",
    "McpDevCommandExecution",
    "McpDevCommandSpec",
    "McpDevServerSpec",
    "McpDevToolCall",
    "McpDevToolResult",
    "McpDevTransportFailure",
    "WorkflowPollRowState",
    "WorkflowPollSummaryStatus",
    "WorkflowStatePollPolicy",
    "_build_parser",
    "_calls_from_args",
    "_command_failed",
    "main",
    "parse_json_object",
    "plate_manager_state_surface_tool_arguments",
    "to_jsonable",
    "workflow_poll_has_reached_terminal_state",
    "workflow_poll_summary_result",
    "workflow_poll_terminal_status",
)


@dataclass(frozen=True, slots=True)
class McpDevCommandExecution:
    """One canonical dev-client command executed on a persistent MCP session."""

    argv: tuple[str, ...]
    payload: JsonObject
    rendered_output: str
    returncode: int
    server_stderr_tail: str | None


class McpDevClient:
    """Synchronous command client backed by one initialized MCP stdio session."""

    def __init__(
        self,
        python_executable: str = sys.executable,
        *,
        surface_profile: LocalCapabilitySurfaceProfile | None = None,
        initialize_timeout_seconds: float = DEFAULT_REGISTRY_DISCOVERY_TIMEOUT_SECONDS,
        server_stderr: TextIO | None = None,
    ) -> None:
        self.server_spec = McpDevServerSpec(
            python_executable,
            surface_profile=(
                FullLocalCapabilitySurfaceProfile()
                if surface_profile is None
                else surface_profile
            ),
        )
        self.initialize_timeout_seconds = initialize_timeout_seconds
        self._parser = _build_parser()
        self._runner = asyncio.Runner()
        self._owns_server_stderr = server_stderr is None
        self._server_stderr = (
            tempfile.TemporaryFile(
                mode="w+",
                encoding="utf-8",
                errors="replace",
            )
            if server_stderr is None
            else server_stderr
        )
        self._session = McpDevStdioSession(self.server_spec, self._server_stderr)
        self._session_started = False
        self._closed = False

    async def _start(self) -> None:
        await self._session.__aenter__()
        self._session_started = True
        await self._session.initialize(
            timeout_seconds=self.initialize_timeout_seconds,
        )

    def start(self) -> "McpDevClient":
        """Start and initialize the owned stdio session."""
        if self._session_started or self._closed:
            raise RuntimeError("MCP dev client cannot be started in its current state.")
        try:
            self._runner.run(self._start())
        except BaseException:
            self.close()
            raise
        return self

    def __enter__(self) -> "McpDevClient":
        return self.start()

    async def _execute_session(
        self,
        command_spec: McpDevCommandSpec,
        args: argparse.Namespace,
        timeout_seconds: float | None,
    ) -> McpDevToolBatchResponse | McpDevToolListResponse:
        command = command_spec.run_session(self._session, args)
        if timeout_seconds is None:
            return await command
        return await asyncio.wait_for(command, timeout=timeout_seconds)

    def execute(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: float | None = None,
    ) -> McpDevCommandExecution:
        """Parse and execute one command through the canonical command registry."""
        if not self._session_started or self._closed:
            raise RuntimeError("MCP dev client is not active.")
        normalized_argv = tuple(argv)
        args = self._parser.parse_args(normalized_argv)
        command_spec = McpDevCommandSpec.for_name(args.command)
        try:
            response = self._runner.run(
                self._execute_session(command_spec, args, timeout_seconds)
            )
        except McpDevCliUsageError:
            raise
        except Exception as exc:
            response = command_spec.transport_failure_response(
                self.server_spec,
                command_spec.execution_phase,
                exc,
                server_stderr_tail=captured_server_stderr_tail(self._server_stderr),
            )
        payload = require_json_object_payload(to_jsonable(response))
        returncode = 0 if args.allow_error_payloads else int(_command_failed(payload))
        return McpDevCommandExecution(
            argv=normalized_argv,
            payload=payload,
            rendered_output=command_spec.render_response(payload, args),
            returncode=returncode,
            server_stderr_tail=captured_server_stderr_tail(self._server_stderr),
        )

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._session_started:
                self._runner.run(self._session.__aexit__(None, None, None))
        finally:
            self._session_started = False
            self._runner.close()
            if self._owns_server_stderr:
                self._server_stderr.close()
            self._closed = True

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()


def _build_parser() -> argparse.ArgumentParser:
    root_options = argparse.ArgumentParser(add_help=False)
    _add_common_options(root_options, suppress_defaults=False)
    command_options = argparse.ArgumentParser(
        add_help=False,
        argument_default=argparse.SUPPRESS,
    )
    _add_common_options(command_options, suppress_defaults=True)

    parser = argparse.ArgumentParser(
        description=(
            "Launch a fresh OpenHCS MCP stdio server from this checkout and "
            "call development tools without relying on Codex MCP process reuse."
        ),
        parents=[root_options],
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    for command_spec in McpDevCommandSpec.all_specs():
        command_spec.register_parser(subparsers, command_options)
    shell_parser = subparsers.add_parser(
        "shell",
        aliases=("session", "batch"),
        help="Run multiple dev-client commands through one persistent MCP session.",
        parents=[command_options],
    )
    shell_parser.set_defaults(command="shell")
    shell_parser.add_argument(
        "--command",
        dest="command_lines",
        action="append",
        help="Execute one command line; repeat for a non-interactive batch.",
    )
    shell_parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the session after the first usage, transport, or tool error.",
    )
    shell_parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Do not print the interactive shell prompt.",
    )
    return parser


def _add_common_options(
    parser: argparse.ArgumentParser,
    *,
    suppress_defaults: bool,
) -> None:
    if suppress_defaults:
        parser.add_argument(
            "--python",
            help="Python executable used to launch `-m openhcs.mcp`.",
        )
        parser.add_argument(
            "--timeout-seconds",
            type=float,
            help="Client-side timeout for initialize, list, and each tool call.",
        )
        parser.add_argument(
            "--allow-error-payloads",
            action="store_true",
            help="Exit zero even when a tool returns MCP or OpenHCS agent errors.",
        )
        parser.add_argument(
            "--surface",
            choices=LocalCapabilitySurfaceProfile.names(),
            help="MCP capability surface used by the fresh server.",
        )
        return

    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch `-m openhcs.mcp`.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_CALL_TIMEOUT_SECONDS,
        help="Client-side timeout for initialize, list, and each tool call.",
    )
    parser.add_argument(
        "--allow-error-payloads",
        action="store_true",
        default=False,
        help="Exit zero even when a tool returns MCP or OpenHCS agent errors.",
    )
    parser.add_argument(
        "--surface",
        choices=LocalCapabilitySurfaceProfile.names(),
        default=FullLocalCapabilitySurfaceProfile.name,
        help="MCP capability surface used by the fresh server (default: full).",
    )


def _calls_from_args(args: argparse.Namespace) -> tuple[McpDevToolCall, ...]:
    return McpDevCommandSpec.for_name(args.command).calls_from_args(args)


def write_stdout(text: str) -> bool:
    """Write CLI output without traceback when downstream pipes close early."""
    try:
        sys.stdout.write(text)
        sys.stdout.write("\n")
        sys.stdout.flush()
    except BrokenPipeError:
        sys.stdout = open(os.devnull, "w")
        return False
    return True


async def _run_async(
    args: argparse.Namespace,
) -> McpDevToolBatchResponse | McpDevToolListResponse:
    server_spec = McpDevServerSpec(
        args.python,
        surface_profile=LocalCapabilitySurfaceProfile.for_name(args.surface),
    )
    return await McpDevCommandSpec.for_name(args.command).run(server_spec, args)


def _persistent_command_argv(
    command_line: str,
    args: argparse.Namespace,
) -> tuple[str, ...]:
    command_argv = tuple(shlex.split(command_line))
    if command_argv[:1] and command_argv[0] in {"shell", "session", "batch"}:
        raise McpDevCliUsageError(
            "A persistent shell cannot start another shell session."
        )
    prefix = ["--timeout-seconds", str(args.timeout_seconds)]
    if args.allow_error_payloads:
        prefix.append("--allow-error-payloads")
    return (*prefix, *command_argv)


def _run_persistent_shell(
    args: argparse.Namespace,
    *,
    stdin=None,
    stdout=None,
    stderr=None,
) -> int:
    """Run line-oriented commands through one initialized current-source server."""
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr
    command_lines = args.command_lines
    interactive = command_lines is None and stdin.isatty()
    aggregate_returncode = 0
    profile = LocalCapabilitySurfaceProfile.for_name(args.surface)

    try:
        with McpDevClient(
            args.python,
            surface_profile=profile,
            initialize_timeout_seconds=max(
                args.timeout_seconds,
                DEFAULT_REGISTRY_DISCOVERY_TIMEOUT_SECONDS,
            ),
        ) as client:
            while True:
                if command_lines is None:
                    if interactive and not args.no_prompt:
                        stdout.write("openhcs-mcp-dev> ")
                        stdout.flush()
                    command_line = stdin.readline()
                    if command_line == "":
                        break
                else:
                    if not command_lines:
                        break
                    command_line = command_lines.pop(0)

                stripped_line = command_line.strip()
                if not stripped_line or stripped_line.startswith("#"):
                    continue
                if stripped_line in {"exit", "quit"}:
                    break
                try:
                    command_argv = _persistent_command_argv(stripped_line, args)
                    execution = client.execute(command_argv)
                except (McpDevCliUsageError, ValueError) as exc:
                    stderr.write(f"error: {exc}\n")
                    aggregate_returncode = 2
                except SystemExit as exc:
                    aggregate_returncode = max(
                        aggregate_returncode,
                        int(exc.code) if isinstance(exc.code, int) else 2,
                    )
                else:
                    stdout.write(execution.rendered_output)
                    stdout.write("\n")
                    stdout.flush()
                    aggregate_returncode = max(
                        aggregate_returncode,
                        execution.returncode,
                    )
                if args.stop_on_error and aggregate_returncode:
                    break
    except Exception as exc:
        stderr.write(f"MCP persistent session failed: {exc}\n")
        return 1
    return aggregate_returncode


def main(argv: Sequence[str] | None = None) -> int:
    disabled_level = logging.root.manager.disable
    registry_logger = logging.getLogger("metaclass_registry.cache")
    registry_level = registry_logger.level
    try:
        if os.getenv("OPENHCS_MCP_DEV_CLIENT_VERBOSE") is None:
            logging.disable(logging.WARNING)
        else:
            registry_logger.setLevel(logging.WARNING)
        parser = _build_parser()
        args = parser.parse_args(argv)
        if args.command == "shell":
            return _run_persistent_shell(args)
        command_spec = McpDevCommandSpec.for_name(args.command)
        try:
            payload = require_json_object_payload(
                to_jsonable(asyncio.run(_run_async(args)))
            )
        except McpDevCliUsageError as exc:
            parser.error(str(exc))
        if not write_stdout(command_spec.render_response(payload, args)):
            return 0
        if args.allow_error_payloads:
            return 0
        return 1 if _command_failed(payload) else 0
    finally:
        logging.disable(disabled_level)
        registry_logger.setLevel(registry_level)


if __name__ == "__main__":
    raise SystemExit(main())
