"""Streamable HTTP boundary for public or OAuth-protected OpenHCS MCP."""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

from openhcs.agent.capabilities import (
    AgentCapabilityRegistry,
    AgentCapabilitySpec,
    CapabilityTransport,
    get_capability_registry,
)
from openhcs.mcp.context import OpenHCSAgentContext, create_hosted_agent_context
from openhcs.mcp.http_auth import (
    IntrospectionTokenVerifier,
    McpHttpConfigurationError,
    McpHttpResourceServerSettings,
)
from openhcs.mcp.server import McpInvocationOutcome, build_server

_AUDIT_LOGGER = logging.getLogger("openhcs.mcp.audit")


def create_hosted_invocation_observer(
    settings: McpHttpResourceServerSettings,
) -> Callable[[AgentCapabilitySpec, McpInvocationOutcome], None]:
    """Create a token-free structured audit observer for one hosted instance."""

    def observe(
        capability: AgentCapabilitySpec,
        outcome: McpInvocationOutcome,
    ) -> None:
        event = {
            "schema_version": "openhcs.mcp.audit.v2",
            "event": "capability_invocation",
            "timestamp_unix": time.time(),
            "authentication_mode": settings.authentication_mode.value,
            "capability": capability.name,
            "transport": CapabilityTransport.HOSTED_STREAMABLE_HTTP.value,
            "outcome": outcome.value,
        }
        if settings.tenant_subject is not None:
            event["tenant_subject"] = settings.tenant_subject
        _AUDIT_LOGGER.info(
            json.dumps(
                event,
                separators=(",", ":"),
                sort_keys=True,
            )
        )

    return observe


def hosted_tool_security_schemes(
    settings: McpHttpResourceServerSettings,
) -> tuple[dict[str, object], ...]:
    """Project the server access policy into every advertised hosted tool."""
    return settings.authentication_mode.tool_security_schemes(
        settings.required_scopes,
    )


def create_hosted_fastmcp_type(
    security_schemes: tuple[dict[str, object], ...],
):
    """Create the SDK adapter that emits current and legacy auth metadata."""
    from mcp.server.fastmcp import FastMCP
    from mcp.types import Tool as McpTool

    class OpenHCSHostedFastMCP(FastMCP):
        async def list_tools(self) -> list[McpTool]:
            tools = await super().list_tools()
            projected_tools = []
            for tool in tools:
                payload = tool.model_dump(by_alias=True, exclude_none=True)
                schemes = [dict(scheme) for scheme in security_schemes]
                payload["securitySchemes"] = schemes
                meta = dict(payload.get("_meta", {}))
                meta["securitySchemes"] = schemes
                payload["_meta"] = meta
                projected_tools.append(McpTool(**payload))
            return projected_tools

    return OpenHCSHostedFastMCP


def create_http_fastmcp_factory(
    settings: McpHttpResourceServerSettings,
    *,
    fastmcp_type: Callable[..., Any] | None = None,
    token_verifier: Any | None = None,
) -> Callable[..., Any]:
    """Project hosted settings into the MCP SDK server factory."""
    security_schemes = hosted_tool_security_schemes(settings)
    if fastmcp_type is None:
        fastmcp_type = create_hosted_fastmcp_type(security_schemes)

    from mcp.server.transport_security import TransportSecuritySettings

    common_kwargs = {
        "host": settings.bind_host,
        "port": settings.bind_port,
        "streamable_http_path": settings.streamable_http_path,
        "stateless_http": True,
        "json_response": True,
        "transport_security": TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=list(settings.allowed_hosts),
            allowed_origins=list(settings.allowed_origins),
        ),
    }
    if settings.oauth is not None:
        from mcp.server.auth.settings import AuthSettings

        common_kwargs.update(
            token_verifier=token_verifier or IntrospectionTokenVerifier(settings),
            auth=AuthSettings(
                issuer_url=settings.oauth.issuer_url,
                resource_server_url=settings.public_url,
                service_documentation_url=settings.resource_documentation_url,
                required_scopes=list(settings.oauth.required_scopes),
            ),
        )

    def factory(*args, **kwargs):
        return fastmcp_type(*args, **common_kwargs, **kwargs)

    return factory


def hosted_capability_registry() -> AgentCapabilityRegistry:
    """Return the canonical hosted registry after applying HTTP access policy."""
    registry = get_capability_registry(CapabilityTransport.HOSTED_STREAMABLE_HTTP)
    if registry.non_read_only_tools:
        raise McpHttpConfigurationError(
            "Hosted MCP cannot expose mutating capabilities: "
            + ", ".join(capability.name for capability in registry.non_read_only_tools)
        )
    return registry


def register_public_http_routes(
    server,
    settings: McpHttpResourceServerSettings,
) -> None:
    """Register unauthenticated operational and OpenAI verification routes."""
    from starlette.responses import JSONResponse, PlainTextResponse

    @server.custom_route(
        "/healthz",
        methods=["GET"],
        name="openhcs_hosted_health",
        include_in_schema=False,
    )
    async def healthz(_request):
        return JSONResponse(
            {
                "status": "ok",
                "transport": CapabilityTransport.HOSTED_STREAMABLE_HTTP.value,
                "authentication_mode": settings.authentication_mode.value,
            }
        )

    if settings.openai_domain_challenge_token is None:
        return

    @server.custom_route(
        "/.well-known/openai-apps-challenge",
        methods=["GET"],
        name="openhcs_openai_domain_challenge",
        include_in_schema=False,
    )
    async def openai_domain_challenge(_request):
        return PlainTextResponse(settings.openai_domain_challenge_token)


def build_http_server(
    settings: McpHttpResourceServerSettings | None = None,
    *,
    context: OpenHCSAgentContext | None = None,
    fastmcp_type: Callable[..., Any] | None = None,
    token_verifier: Any | None = None,
):
    """Build the fail-closed, hosted-only OpenHCS MCP surface."""
    resolved_settings = settings or McpHttpResourceServerSettings.from_environment()
    hosted_capability_registry()
    server = build_server(
        context if context is not None else create_hosted_agent_context(),
        fastmcp_factory=create_http_fastmcp_factory(
            resolved_settings,
            fastmcp_type=fastmcp_type,
            token_verifier=token_verifier,
        ),
        capability_transport=CapabilityTransport.HOSTED_STREAMABLE_HTTP,
        invocation_observer=create_hosted_invocation_observer(resolved_settings),
    )
    register_public_http_routes(server, resolved_settings)
    return server


def _build_parser() -> argparse.ArgumentParser:
    """Build the hosted MCP CLI parser without resolving runtime settings."""
    return argparse.ArgumentParser(
        description="Run the stateless OpenHCS MCP HTTP server.",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run stateless Streamable HTTP MCP."""
    _build_parser().parse_args(argv)
    build_http_server().run(transport="streamable-http")
    return 0


if __name__ == "__main__":
    main()
