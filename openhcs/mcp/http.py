"""Authenticated Streamable HTTP boundary for hosted OpenHCS MCP."""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

from openhcs.agent.capabilities import AgentCapabilitySpec, CapabilityTransport
from openhcs.mcp.context import OpenHCSAgentContext
from openhcs.mcp.http_auth import (
    IntrospectionTokenVerifier,
    McpHttpResourceServerSettings,
)
from openhcs.mcp.server import McpInvocationOutcome, build_server


_AUDIT_LOGGER = logging.getLogger("openhcs.mcp.audit")


def create_hosted_invocation_observer(
    settings: McpHttpResourceServerSettings,
) -> Callable[[AgentCapabilitySpec, McpInvocationOutcome], None]:
    """Create a token-free structured audit observer for one tenant instance."""

    def observe(
        capability: AgentCapabilitySpec,
        outcome: McpInvocationOutcome,
    ) -> None:
        _AUDIT_LOGGER.info(
            json.dumps(
                {
                    "schema_version": "openhcs.mcp.audit.v1",
                    "event": "capability_invocation",
                    "timestamp_unix": time.time(),
                    "tenant_subject": settings.tenant_subject,
                    "capability": capability.name,
                    "transport": CapabilityTransport.HOSTED_STREAMABLE_HTTP.value,
                    "outcome": outcome.value,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        )

    return observe


def create_http_fastmcp_factory(
    settings: McpHttpResourceServerSettings,
    *,
    fastmcp_type: Callable[..., Any] | None = None,
    token_verifier: Any | None = None,
) -> Callable[..., Any]:
    """Project hosted settings into the MCP SDK's authenticated server factory."""
    if fastmcp_type is None:
        from mcp.server.fastmcp import FastMCP

        fastmcp_type = FastMCP

    from mcp.server.auth.settings import AuthSettings
    from mcp.server.transport_security import TransportSecuritySettings

    verifier = token_verifier or IntrospectionTokenVerifier(settings)
    return partial(
        fastmcp_type,
        host=settings.bind_host,
        port=settings.bind_port,
        streamable_http_path=settings.streamable_http_path,
        stateless_http=True,
        json_response=True,
        token_verifier=verifier,
        auth=AuthSettings(
            issuer_url=settings.issuer_url,
            resource_server_url=settings.public_url,
            required_scopes=list(settings.required_scopes),
        ),
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=list(settings.allowed_hosts),
            allowed_origins=list(settings.allowed_origins),
        ),
    )


def build_http_server(
    settings: McpHttpResourceServerSettings | None = None,
    *,
    context: OpenHCSAgentContext | None = None,
    fastmcp_type: Callable[..., Any] | None = None,
    token_verifier: Any | None = None,
):
    """Build the fail-closed, hosted-only OpenHCS MCP surface."""
    resolved_settings = settings or McpHttpResourceServerSettings.from_environment()
    return build_server(
        context,
        fastmcp_factory=create_http_fastmcp_factory(
            resolved_settings,
            fastmcp_type=fastmcp_type,
            token_verifier=token_verifier,
        ),
        capability_transport=CapabilityTransport.HOSTED_STREAMABLE_HTTP,
        invocation_observer=create_hosted_invocation_observer(resolved_settings),
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the hosted MCP CLI parser without resolving runtime settings."""
    return argparse.ArgumentParser(
        description="Run the authenticated, stateless OpenHCS MCP HTTP server.",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run authenticated, stateless Streamable HTTP MCP."""
    _build_parser().parse_args(argv)
    build_http_server().run(transport="streamable-http")
    return 0


if __name__ == "__main__":
    main()
