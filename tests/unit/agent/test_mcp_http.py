"""Tests for the authenticated hosted MCP server construction boundary."""

import json
import logging

from openhcs.agent.capabilities import CapabilityTransport, agent_capabilities
from openhcs.mcp import http as mcp_http
from openhcs.mcp.http_auth import McpHttpResourceServerSettings
from openhcs.mcp.server import McpInvocationOutcome


def _settings():
    return McpHttpResourceServerSettings(
        public_url="https://mcp.openhcs.example/mcp",
        issuer_url="https://auth.openhcs.example",
        introspection_url="https://auth.openhcs.example/introspect",
        introspection_client_id="openhcs-resource-server",
        introspection_client_secret="secret",
        tenant_subject="tenant-user-1",
        required_scopes=("openhcs:use",),
        allowed_hosts=("mcp.openhcs.example",),
        allowed_origins=("https://chatgpt.com",),
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
    assert kwargs["transport_security"].enable_dns_rebinding_protection is True
    assert kwargs["transport_security"].allowed_hosts == ["mcp.openhcs.example"]


def test_build_http_server_requests_only_hosted_nominal_surface(monkeypatch):
    captured = {}
    expected_server = object()

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


def test_hosted_invocation_audit_is_structured_and_token_free(caplog):
    observer = mcp_http.create_hosted_invocation_observer(_settings())

    with caplog.at_level(logging.INFO, logger="openhcs.mcp.audit"):
        observer(
            agent_capabilities.list_capabilities,
            McpInvocationOutcome.SUCCEEDED,
        )

    payload = json.loads(caplog.records[-1].message)
    assert payload["tenant_subject"] == "tenant-user-1"
    assert payload["capability"] == "openhcs_list_capabilities"
    assert payload["transport"] == "hosted_streamable_http"
    assert payload["outcome"] == "succeeded"
    assert "token" not in caplog.records[-1].message


def test_http_main_uses_streamable_http(monkeypatch):
    transports = []

    class Server:
        def run(self, *, transport):
            transports.append(transport)

    monkeypatch.setattr(mcp_http, "build_http_server", lambda: Server())

    mcp_http.main()

    assert transports == ["streamable-http"]
