"""Fail-soft bootstrap for the OpenHCS MCP stdio process."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
import logging
import os
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from openhcs.agent.capabilities import LocalCapabilitySurfaceProfile
    from mcp.server.fastmcp import FastMCP


MCP_BOOTSTRAP_SCHEMA_VERSION = "openhcs.mcp.bootstrap.v1"
MCP_BOOTSTRAP_FAILURE_HINT = (
    "The OpenHCS MCP process started, but the full agent server could not be "
    "constructed or run. Fix the startup exception and restart the MCP client."
)
MCP_BOOTSTRAP_SERVER_INSTRUCTIONS = (
    "OpenHCS could not construct its full MCP server. Call openhcs_health_check "
    "or openhcs_bootstrap_failure for the structured startup error, fix the local "
    "installation, and restart this stdio process."
)
MCP_LOCAL_SURFACE_ENVIRONMENT_VARIABLE = "OPENHCS_MCP_LOCAL_SURFACE"
MCP_VERBOSE_ENVIRONMENT_VARIABLE = "OPENHCS_MCP_VERBOSE"
MCP_STABLE_LAUNCH_COMMAND_ENVIRONMENT_VARIABLE = (
    "OPENHCS_MCP_STABLE_LAUNCH_COMMAND_JSON"
)
MCP_INSTALLATION_POINTER_ENVIRONMENT_VARIABLE = "OPENHCS_MCP_INSTALLATION_POINTER"


class McpBootstrapFailurePhase(str, Enum):
    BUILD_SERVER = "build_server"
    RUN_SERVER = "run_server"


@dataclass(frozen=True, slots=True)
class McpBootstrapFailurePayload:
    """Structured startup failure payload exposed over MCP."""

    schema_version: str
    ok: bool
    status: str
    service: str
    phase: McpBootstrapFailurePhase
    exception_type: str
    message: str
    hint: str


class McpBootstrapFailurePayloadWire(TypedDict):
    """FastMCP-compatible wire schema for bootstrap failure payloads."""

    schema_version: str
    ok: bool
    status: str
    service: str
    phase: str
    exception_type: str
    message: str
    hint: str


class McpBootstrapFailurePayloadAuthority:
    """Authoritative projection for MCP bootstrap failure payloads."""

    @staticmethod
    def from_failure(failure: "McpBootstrapFailure") -> McpBootstrapFailurePayload:
        return McpBootstrapFailurePayload(
            schema_version=MCP_BOOTSTRAP_SCHEMA_VERSION,
            ok=False,
            status="unavailable",
            service="openhcs.mcp",
            phase=failure.phase,
            exception_type=failure.exception_type,
            message=failure.message,
            hint=MCP_BOOTSTRAP_FAILURE_HINT,
        )

    @staticmethod
    def as_wire(payload: McpBootstrapFailurePayload) -> McpBootstrapFailurePayloadWire:
        return McpBootstrapFailurePayloadWire(
            schema_version=payload.schema_version,
            ok=payload.ok,
            status=payload.status,
            service=payload.service,
            phase=payload.phase.value,
            exception_type=payload.exception_type,
            message=payload.message,
            hint=payload.hint,
        )


@dataclass(frozen=True, slots=True)
class McpBootstrapFailure:
    """Structured startup failure exposed over MCP instead of closing stdio."""

    phase: McpBootstrapFailurePhase
    exception_type: str
    message: str

    @classmethod
    def from_exception(
        cls,
        exception: BaseException,
        phase: McpBootstrapFailurePhase,
    ) -> "McpBootstrapFailure":
        return cls(
            phase=phase,
            exception_type=type(exception).__name__,
            message=str(exception),
        )

    def payload(self) -> McpBootstrapFailurePayloadWire:
        payload = McpBootstrapFailurePayloadAuthority.from_failure(self)
        return McpBootstrapFailurePayloadAuthority.as_wire(payload)


def build_bootstrap_failure_server(
    exception: BaseException,
    phase: McpBootstrapFailurePhase = McpBootstrapFailurePhase.BUILD_SERVER,
) -> "FastMCP":
    """Build a minimal MCP server that reports startup failure through health."""
    from mcp.server.fastmcp import FastMCP

    failure = McpBootstrapFailure.from_exception(exception, phase)
    server = FastMCP("OpenHCS", instructions=MCP_BOOTSTRAP_SERVER_INSTRUCTIONS)

    @server.tool()
    def openhcs_health_check() -> McpBootstrapFailurePayloadWire:
        """Report why the full OpenHCS MCP server could not start."""
        return failure.payload()

    @server.tool()
    def openhcs_bootstrap_failure() -> McpBootstrapFailurePayloadWire:
        """Return the OpenHCS MCP startup failure payload."""
        return failure.payload()

    return server


def build_bootstrapped_server(
    capability_surface_profile: "LocalCapabilitySurfaceProfile | None" = None,
) -> "FastMCP":
    """Build the full OpenHCS MCP server, or a fail-soft bootstrap server."""
    try:
        from openhcs.agent.capabilities import DesktopLocalCapabilitySurfaceProfile
        from openhcs.mcp.server import build_server

        return build_server(
            capability_surface_profile=(
                DesktopLocalCapabilitySurfaceProfile()
                if capability_surface_profile is None
                else capability_surface_profile
            )
        )
    except Exception as exc:
        return build_bootstrap_failure_server(
            exc, McpBootstrapFailurePhase.BUILD_SERVER
        )


def run_bootstrapped_server(
    capability_surface_profile: "LocalCapabilitySurfaceProfile | None" = None,
) -> None:
    """Run the OpenHCS MCP server with transport-owned protocol stdout."""
    from openhcs.mcp.stdio import McpStdioTransport

    with McpStdioTransport.reserve_process_stdout() as stdio_transport:
        try:
            server = (
                build_bootstrapped_server()
                if capability_surface_profile is None
                else build_bootstrapped_server(capability_surface_profile)
            )
            stdio_transport.run(server)
        except Exception as exc:
            stdio_transport.run(
                build_bootstrap_failure_server(
                    exc,
                    McpBootstrapFailurePhase.RUN_SERVER,
                )
            )


def _build_parser() -> argparse.ArgumentParser:
    from openhcs.agent.capabilities import (
        DesktopLocalCapabilitySurfaceProfile,
        LocalCapabilitySurfaceProfile,
    )

    parser = argparse.ArgumentParser(description="Run the OpenHCS MCP stdio server.")
    parser.add_argument(
        "--surface",
        choices=LocalCapabilitySurfaceProfile.names(),
        default=os.environ.get(
            MCP_LOCAL_SURFACE_ENVIRONMENT_VARIABLE,
            DesktopLocalCapabilitySurfaceProfile.name,
        ),
        help=(
            "Capability surface exposed to the MCP client. The default desktop "
            "surface keeps normal UI and viewer workflows while hiding headless, "
            "runtime-server, fallback, and expert-only tools."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    from openhcs.agent.capabilities import LocalCapabilitySurfaceProfile

    disabled_level = logging.root.manager.disable
    try:
        if os.getenv(MCP_VERBOSE_ENVIRONMENT_VARIABLE) is None:
            logging.disable(logging.INFO)
        args = _build_parser().parse_args(argv)
        run_bootstrapped_server(LocalCapabilitySurfaceProfile.for_name(args.surface))
    finally:
        logging.disable(disabled_level)
