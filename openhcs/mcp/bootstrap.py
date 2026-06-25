"""Fail-soft bootstrap for the OpenHCS MCP stdio process."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


MCP_BOOTSTRAP_SCHEMA_VERSION = "openhcs.mcp.bootstrap.v1"
MCP_BOOTSTRAP_FAILURE_HINT = (
    "The OpenHCS MCP process started, but the full agent server could not be "
    "constructed or run. Fix the startup exception and restart the MCP client."
)


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
    server = FastMCP("OpenHCS")

    @server.tool()
    def openhcs_health_check() -> McpBootstrapFailurePayloadWire:
        """Report why the full OpenHCS MCP server could not start."""
        return failure.payload()

    @server.tool()
    def openhcs_bootstrap_failure() -> McpBootstrapFailurePayloadWire:
        """Return the OpenHCS MCP startup failure payload."""
        return failure.payload()

    return server


def build_bootstrapped_server() -> "FastMCP":
    """Build the full OpenHCS MCP server, or a fail-soft bootstrap server."""
    try:
        from openhcs.mcp.server import build_server

        return build_server()
    except Exception as exc:
        return build_bootstrap_failure_server(exc, McpBootstrapFailurePhase.BUILD_SERVER)


def run_bootstrapped_server() -> None:
    """Run the OpenHCS MCP server, preserving stdio for early run failures."""
    try:
        build_bootstrapped_server().run()
    except Exception as exc:
        build_bootstrap_failure_server(
            exc,
            McpBootstrapFailurePhase.RUN_SERVER,
        ).run()


def main() -> None:
    run_bootstrapped_server()
