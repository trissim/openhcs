"""Fresh-process MCP development client for the active OpenHCS checkout."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from collections.abc import Sequence

if os.getenv("OPENHCS_MCP_DEV_CLIENT_VERBOSE") is None:
    logging.disable(logging.WARNING)
else:
    logging.getLogger("metaclass_registry.cache").setLevel(logging.WARNING)

from openhcs.agent.serialization import to_jsonable as to_jsonable
from openhcs.constants.constants import AllComponents as AllComponents
from openhcs.mcp import dev_client_renderers as dev_client_renderers
from openhcs.mcp.dev_client_core import (
    DEFAULT_CALL_TIMEOUT_SECONDS,
    DEFAULT_REGISTRY_DISCOVERY_TIMEOUT_SECONDS as DEFAULT_REGISTRY_DISCOVERY_TIMEOUT_SECONDS,
    McpDevCliUsageError,
    McpDevClientPhase as McpDevClientPhase,
    McpDevServerSpec,
    McpDevToolBatchResponse,
    McpDevToolCall,
    McpDevToolListResponse,
    McpDevToolResult as McpDevToolResult,
    McpDevTransportFailure as McpDevTransportFailure,
    WorkflowPollRowState as WorkflowPollRowState,
    WorkflowPollSummaryStatus as WorkflowPollSummaryStatus,
    WorkflowStatePollPolicy as WorkflowStatePollPolicy,
    _command_failed,
    call_execute_source_with_submission as call_execute_source_with_submission,
    call_fresh_mcp_server as call_fresh_mcp_server,
    call_selected_workflow_with_state_poll as call_selected_workflow_with_state_poll,
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

_DECLARATION_MODULES = (dev_client_renderers, dev_client_commands)

__all__ = (
    "AllComponents",
    "DEFAULT_CODE_DOCUMENT_MAX_CHARS",
    "DEFAULT_REGISTRY_DISCOVERY_TIMEOUT_SECONDS",
    "GeneratedMcpDevCommandProfile",
    "McpDevCliUsageError",
    "McpDevClientPhase",
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
    "call_execute_source_with_submission",
    "call_fresh_mcp_server",
    "call_selected_workflow_with_state_poll",
    "main",
    "parse_json_object",
    "plate_manager_state_surface_tool_arguments",
    "to_jsonable",
    "workflow_poll_has_reached_terminal_state",
    "workflow_poll_summary_result",
    "workflow_poll_terminal_status",
)


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
    server_spec = McpDevServerSpec(args.python)
    return await McpDevCommandSpec.for_name(args.command).run(server_spec, args)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    command_spec = McpDevCommandSpec.for_name(args.command)
    try:
        payload = require_json_object_payload(to_jsonable(asyncio.run(_run_async(args))))
    except McpDevCliUsageError as exc:
        parser.error(str(exc))
    if not write_stdout(command_spec.render_response(payload, args)):
        return 0
    if args.allow_error_payloads:
        return 0
    return 1 if _command_failed(payload) else 0


if __name__ == "__main__":
    raise SystemExit(main())
