"""Fresh-process MCP development client for the active OpenHCS checkout."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
import json
import sys
from typing import cast

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, TextContent


JsonPrimitive = str | int | float | bool | None
JsonValue = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]

DEFAULT_CALL_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class McpDevToolCall:
    """One MCP tool invocation issued against a fresh stdio server."""

    name: str
    arguments: JsonObject


@dataclass(frozen=True, slots=True)
class McpDevServerSpec:
    """Command used to launch the active checkout MCP server."""

    python_executable: str
    module_name: str = "openhcs.mcp"

    def parameters(self) -> StdioServerParameters:
        return StdioServerParameters(
            command=self.python_executable,
            args=("-m", self.module_name),
        )


def parse_json_object(argument_text: str) -> JsonObject:
    """Parse a JSON object for MCP tool arguments."""
    value = cast(JsonValue, json.loads(argument_text))
    if not isinstance(value, dict):
        raise ValueError("MCP tool arguments must be a JSON object.")
    return value


def _payload_from_text(text: str) -> JsonValue:
    try:
        return cast(JsonValue, json.loads(text))
    except json.JSONDecodeError:
        return {"text": text}


def _content_payloads(result: CallToolResult) -> list[JsonValue]:
    payloads: list[JsonValue] = []
    for content in result.content:
        if not isinstance(content, TextContent):
            raise RuntimeError(
                "OpenHCS MCP dev client only supports text tool responses; "
                f"received {type(content).__name__}."
            )
        payloads.append(_payload_from_text(content.text))
    return payloads


def _tool_result_payload(tool_name: str, result: CallToolResult) -> JsonObject:
    return {
        "tool": tool_name,
        "mcp_error": bool(result.isError),
        "payloads": _content_payloads(result),
    }


def _contains_agent_error(value: JsonValue) -> bool:
    if isinstance(value, dict):
        errors = value.get("errors")
        if isinstance(errors, list) and len(errors) > 0:
            return True
        return any(_contains_agent_error(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_agent_error(child) for child in value)
    return False


def _command_failed(payload: JsonObject) -> bool:
    results = payload.get("results")
    if isinstance(results, list):
        for result in results:
            if isinstance(result, dict):
                if result.get("mcp_error") is True:
                    return True
                if _contains_agent_error(result):
                    return True
    return False


async def call_fresh_mcp_server(
    server_spec: McpDevServerSpec,
    calls: Sequence[McpDevToolCall],
    timeout_seconds: float,
) -> JsonObject:
    """Start a fresh MCP server, issue calls, and return JSON-ready results."""
    async with stdio_client(server_spec.parameters()) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await asyncio.wait_for(session.initialize(), timeout=timeout_seconds)
            results: list[JsonValue] = []
            for call in calls:
                result = await asyncio.wait_for(
                    session.call_tool(call.name, call.arguments),
                    timeout=timeout_seconds,
                )
                results.append(_tool_result_payload(call.name, result))
            return {
                "server": {
                    "command": server_spec.python_executable,
                    "module": server_spec.module_name,
                },
                "results": results,
            }


async def list_fresh_mcp_tools(
    server_spec: McpDevServerSpec,
    timeout_seconds: float,
) -> JsonObject:
    """Start a fresh MCP server and return registered tool metadata."""
    async with stdio_client(server_spec.parameters()) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await asyncio.wait_for(session.initialize(), timeout=timeout_seconds)
            result = await asyncio.wait_for(
                session.list_tools(),
                timeout=timeout_seconds,
            )
            tools: list[JsonValue] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": cast(JsonValue, tool.inputSchema),
                }
                for tool in result.tools
            ]
            return {
                "server": {
                    "command": server_spec.python_executable,
                    "module": server_spec.module_name,
                },
                "tool_count": len(tools),
                "tools": tools,
            }


def _health_calls() -> tuple[McpDevToolCall, ...]:
    return (McpDevToolCall("openhcs_health_check", {}),)


def _ui_smoke_calls() -> tuple[McpDevToolCall, ...]:
    return (
        McpDevToolCall("openhcs_health_check", {}),
        McpDevToolCall("openhcs_ui_bridge_status", {}),
        McpDevToolCall("openhcs_ui_list_bridges", {}),
        McpDevToolCall("openhcs_ui_list_windows", {}),
    )


def _viewer_payload_arguments(args: argparse.Namespace) -> JsonObject:
    return {
        "port": args.port,
        "include_array_values": args.include_array_values,
        "include_shape_payloads": args.include_shape_payloads,
        "max_array_elements": args.max_array_elements,
        "max_shape_payloads": args.max_shape_payloads,
        "timeout_ms": args.control_timeout_ms,
    }


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
    subparsers.add_parser(
        "health",
        help="Call openhcs_health_check.",
        parents=[command_options],
    )
    subparsers.add_parser(
        "tools",
        help="List current-source MCP tools.",
        parents=[command_options],
    )

    call_parser = subparsers.add_parser(
        "call",
        help="Call one MCP tool.",
        parents=[command_options],
    )
    call_parser.add_argument("tool_name")
    call_parser.add_argument(
        "--arguments",
        default="{}",
        help="JSON object passed as the MCP tool arguments.",
    )

    workflow_parser = subparsers.add_parser(
        "selected-workflow",
        help="Run a selected UI plate workflow through the MCP UI bridge.",
        parents=[command_options],
    )
    workflow_parser.add_argument(
        "workflow",
        choices=("init_plate", "compile_plate", "run_plate"),
    )

    widget_parser = subparsers.add_parser(
        "widget-tree",
        help="Read a UI window's generic clickable widget tree.",
        parents=[command_options],
    )
    widget_parser.add_argument("window_id")
    widget_parser.add_argument("--maximum-text-length", type=int, default=120)
    widget_parser.add_argument("--timeout-ms", type=int, default=750)

    snapshot_parser = subparsers.add_parser(
        "window-snapshot",
        help="Capture a UI bridge window screenshot.",
        parents=[command_options],
    )
    snapshot_parser.add_argument("window_id")
    snapshot_parser.add_argument("--capture-scope", default="window")
    snapshot_parser.add_argument("--timeout-ms", type=int, default=750)

    viewer_parser = subparsers.add_parser(
        "viewer-payloads",
        help="Read viewer-agnostic layer, axis, image, and shape payload records.",
        parents=[command_options],
    )
    viewer_parser.add_argument("port", type=int)
    viewer_parser.add_argument("--include-array-values", action="store_true")
    viewer_parser.add_argument("--include-shape-payloads", action="store_true")
    viewer_parser.add_argument("--max-array-elements", type=int, default=256)
    viewer_parser.add_argument("--max-shape-payloads", type=int, default=32)
    viewer_parser.add_argument("--control-timeout-ms", type=int, default=750)

    subparsers.add_parser(
        "ui-smoke",
        help="Call health plus UI bridge status, bridge list, and window list.",
        parents=[command_options],
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
    if args.command == "health":
        return _health_calls()
    if args.command == "ui-smoke":
        return _ui_smoke_calls()
    if args.command == "call":
        return (
            McpDevToolCall(
                args.tool_name,
                parse_json_object(args.arguments),
            ),
        )
    if args.command == "selected-workflow":
        return (
            McpDevToolCall(
                "openhcs_ui_selected_plate_workflow",
                {"workflow": args.workflow},
            ),
        )
    if args.command == "widget-tree":
        return (
            McpDevToolCall(
                "openhcs_ui_get_widget_tree",
                {
                    "window_id": args.window_id,
                    "maximum_text_length": args.maximum_text_length,
                    "timeout_ms": args.timeout_ms,
                },
            ),
        )
    if args.command == "window-snapshot":
        return (
            McpDevToolCall(
                "openhcs_ui_snapshot_window",
                {
                    "window_id": args.window_id,
                    "capture_scope": args.capture_scope,
                    "timeout_ms": args.timeout_ms,
                },
            ),
        )
    if args.command == "viewer-payloads":
        return (
            McpDevToolCall(
                "openhcs_get_viewer_window_payloads",
                _viewer_payload_arguments(args),
            ),
        )
    raise ValueError(f"Unsupported MCP dev command: {args.command}")


async def _run_async(args: argparse.Namespace) -> JsonObject:
    server_spec = McpDevServerSpec(args.python)
    if args.command == "tools":
        return await list_fresh_mcp_tools(server_spec, args.timeout_seconds)
    return await call_fresh_mcp_server(
        server_spec,
        _calls_from_args(args),
        args.timeout_seconds,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    payload = asyncio.run(_run_async(args))
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.allow_error_payloads:
        return 0
    return 1 if _command_failed(payload) else 0


if __name__ == "__main__":
    raise SystemExit(main())
