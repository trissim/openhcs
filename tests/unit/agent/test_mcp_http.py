"""Tests for the authenticated hosted MCP server construction boundary."""

import json
import logging
from types import SimpleNamespace

import httpx
import pytest

from openhcs.agent.capabilities import CapabilityTransport, agent_capabilities
from openhcs.mcp import http as mcp_http
from openhcs.mcp.http_auth import (
    McpHttpConfigurationError,
    McpHttpOAuthSettings,
    McpHttpResourceServerSettings,
)
from openhcs.mcp.server import McpInvocationOutcome


def _settings(
    *,
    authenticated=True,
    challenge_token=None,
    allowed_hosts=("mcp.openhcs.example",),
):
    return McpHttpResourceServerSettings(
        public_url="https://mcp.openhcs.example/mcp",
        allowed_hosts=allowed_hosts,
        oauth=(
            McpHttpOAuthSettings(
                issuer_url="https://auth.openhcs.example",
                introspection_url="https://auth.openhcs.example/introspect",
                introspection_client_id="openhcs-resource-server",
                introspection_client_secret="secret",
                tenant_subject="tenant-user-1",
                required_scopes=("openhcs:use",),
            )
            if authenticated
            else None
        ),
        allowed_origins=("https://chatgpt.com",),
        resource_documentation_url="https://openhcs.readthedocs.io/",
        openai_domain_challenge_token=challenge_token,
        bind_host="127.0.0.1",
        bind_port=8123,
    )


def test_http_factory_projects_fail_closed_transport_settings():
    calls = []

    def recording_fastmcp(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    token_verifier = object()
    factory = mcp_http.create_http_fastmcp_factory(
        _settings(),
        fastmcp_type=recording_fastmcp,
        token_verifier=token_verifier,
    )

    result = factory("OpenHCS", instructions="hosted")

    assert result is not None
    args, kwargs = calls[0]
    assert args == ("OpenHCS",)
    assert kwargs["instructions"] == "hosted"
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 8123
    assert kwargs["streamable_http_path"] == "/mcp"
    assert kwargs["stateless_http"] is True
    assert kwargs["json_response"] is True
    assert kwargs["token_verifier"] is token_verifier
    assert kwargs["auth"].required_scopes == ["openhcs:use"]
    assert str(kwargs["auth"].service_documentation_url) == (
        "https://openhcs.readthedocs.io/"
    )
    assert kwargs["transport_security"].enable_dns_rebinding_protection is True
    assert kwargs["transport_security"].allowed_hosts == ["mcp.openhcs.example"]


def test_public_http_factory_omits_authentication_middleware():
    calls = []

    def recording_fastmcp(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    factory = mcp_http.create_http_fastmcp_factory(
        _settings(authenticated=False),
        fastmcp_type=recording_fastmcp,
    )

    factory("OpenHCS", instructions="hosted")

    _, kwargs = calls[0]
    assert "token_verifier" not in kwargs
    assert "auth" not in kwargs
    assert mcp_http.hosted_tool_security_schemes(_settings(authenticated=False)) == (
        {"type": "noauth"},
    )


@pytest.mark.asyncio
async def test_hosted_fastmcp_projects_security_schemes_top_level_and_legacy_meta():
    server_type = mcp_http.create_hosted_fastmcp_type(
        ({"type": "oauth2", "scopes": ["openhcs:use"]},)
    )
    server = server_type("OpenHCS")

    @server.tool()
    def inspect_registry(query: str) -> dict:
        return {"query": query}

    (tool,) = await server.list_tools()
    payload = tool.model_dump(by_alias=True)
    expected = [{"type": "oauth2", "scopes": ["openhcs:use"]}]
    assert payload["securitySchemes"] == expected
    assert payload["_meta"]["securitySchemes"] == expected


def test_build_http_server_requests_only_hosted_nominal_surface(monkeypatch):
    captured = {}

    class Server:
        def __init__(self):
            self.routes = {}

        def custom_route(self, path, **_kwargs):
            def register(handler):
                self.routes[path] = handler
                return handler

            return register

    expected_server = Server()

    def recording_build_server(context, **kwargs):
        captured["context"] = context
        captured.update(kwargs)
        return expected_server

    monkeypatch.setattr(mcp_http, "build_server", recording_build_server)

    result = mcp_http.build_http_server(
        _settings(),
        context=None,
        fastmcp_type=lambda *args, **kwargs: object(),
        token_verifier=object(),
    )

    assert result is expected_server
    assert captured["capability_transport"] is (
        CapabilityTransport.HOSTED_STREAMABLE_HTTP
    )
    assert callable(captured["invocation_observer"])
    assert "/healthz" in expected_server.routes


def test_public_http_rejects_future_mutating_hosted_capability(monkeypatch):
    unsafe = SimpleNamespace(
        name="unsafe",
        mutating=True,
        side_effects=("writes",),
    )
    monkeypatch.setattr(
        mcp_http,
        "get_capability_registry",
        lambda *_args, **_kwargs: SimpleNamespace(non_read_only_tools=(unsafe,)),
    )

    with pytest.raises(McpHttpConfigurationError, match="unsafe"):
        mcp_http.hosted_capability_registry()


def test_hosted_invocation_audit_is_structured_and_token_free(caplog):
    observer = mcp_http.create_hosted_invocation_observer(_settings())

    with caplog.at_level(logging.INFO, logger="openhcs.mcp.audit"):
        observer(
            agent_capabilities.list_capabilities,
            McpInvocationOutcome.SUCCEEDED,
        )

    payload = json.loads(caplog.records[-1].message)
    assert payload["schema_version"] == "openhcs.mcp.audit.v2"
    assert payload["tenant_subject"] == "tenant-user-1"
    assert payload["capability"] == "openhcs_list_capabilities"
    assert payload["transport"] == "hosted_streamable_http"
    assert payload["outcome"] == "succeeded"
    assert payload["authentication_mode"] == "oauth_introspection"
    assert payload["tenant_subject"] == "tenant-user-1"
    assert "token" not in caplog.records[-1].message


def test_public_hosted_invocation_audit_has_no_synthetic_tenant(caplog):
    observer = mcp_http.create_hosted_invocation_observer(
        _settings(authenticated=False)
    )

    with caplog.at_level(logging.INFO, logger="openhcs.mcp.audit"):
        observer(
            agent_capabilities.list_capabilities,
            McpInvocationOutcome.SUCCEEDED,
        )

    payload = json.loads(caplog.records[-1].message)
    assert payload["schema_version"] == "openhcs.mcp.audit.v2"
    assert payload["authentication_mode"] == "public_read_only"
    assert "tenant_subject" not in payload


@pytest.mark.asyncio
async def test_openai_domain_challenge_returns_exact_token():
    class Server:
        def __init__(self):
            self.routes = {}

        def custom_route(self, path, **_kwargs):
            def register(handler):
                self.routes[path] = handler
                return handler

            return register

    server = Server()
    mcp_http.register_public_http_routes(
        server,
        _settings(challenge_token="exact-openai-token"),
    )

    response = await server.routes["/.well-known/openai-apps-challenge"](None)

    assert response.body == b"exact-openai-token"


@pytest.mark.asyncio
async def test_public_streamable_http_is_anonymous_and_serves_operational_routes():
    server = mcp_http.build_http_server(
        _settings(
            authenticated=False,
            challenge_token="exact-openai-token",
            allowed_hosts=("mcp.openhcs.example", "testserver"),
        )
    )
    app = server.streamable_http_app()
    async with server.session_manager.run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            health = await client.get("/healthz")
            challenge = await client.get("/.well-known/openai-apps-challenge")
            initialized = await client.post(
                "/mcp",
                headers={
                    "Accept": "application/json, text/event-stream",
                    "Content-Type": "application/json",
                },
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "1"},
                    },
                },
            )
            listed = await client.post(
                "/mcp",
                headers={
                    "Accept": "application/json, text/event-stream",
                    "Content-Type": "application/json",
                },
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/list",
                    "params": {},
                },
            )

    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "transport": "hosted_streamable_http",
        "authentication_mode": "public_read_only",
    }
    assert challenge.status_code == 200
    assert challenge.text == "exact-openai-token"
    assert initialized.status_code == 200
    assert initialized.json()["result"]["serverInfo"]["name"] == "OpenHCS"
    assert listed.status_code == 200
    tools = listed.json()["result"]["tools"]
    assert tools
    assert all(tool["securitySchemes"] == [{"type": "noauth"}] for tool in tools)
    assert all(
        tool["_meta"]["securitySchemes"] == [{"type": "noauth"}] for tool in tools
    )
    assert all(tool["annotations"]["readOnlyHint"] is True for tool in tools)


@pytest.mark.asyncio
async def test_oauth_streamable_http_advertises_resource_and_challenges_anonymous_calls():
    server = mcp_http.build_http_server(
        _settings(allowed_hosts=("mcp.openhcs.example", "testserver"))
    )
    app = server.streamable_http_app()
    async with server.session_manager.run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            metadata = await client.get("/.well-known/oauth-protected-resource/mcp")
            initialized = await client.post(
                "/mcp",
                headers={
                    "Accept": "application/json, text/event-stream",
                    "Content-Type": "application/json",
                },
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "1"},
                    },
                },
            )

    assert metadata.status_code == 200
    assert metadata.json() == {
        "resource": "https://mcp.openhcs.example/mcp",
        "authorization_servers": ["https://auth.openhcs.example/"],
        "scopes_supported": ["openhcs:use"],
        "bearer_methods_supported": ["header"],
    }
    assert initialized.status_code == 401
    assert "resource_metadata=" in initialized.headers["WWW-Authenticate"]


def test_http_main_uses_streamable_http(monkeypatch):
    transports = []

    class Server:
        def run(self, *, transport):
            transports.append(transport)

    monkeypatch.setattr(mcp_http, "build_http_server", lambda: Server())

    mcp_http.main([])

    assert transports == ["streamable-http"]
