"""Tests for the hosted MCP OAuth resource-server boundary."""

import time

import httpx
import pytest

from openhcs.mcp.http_auth import (
    IntrospectionTokenVerifier,
    McpHttpConfigurationError,
    McpHttpResourceServerSettings,
)


def _settings(**overrides):
    values = {
        "public_url": "https://mcp.openhcs.example/mcp",
        "issuer_url": "https://auth.openhcs.example",
        "introspection_url": "https://auth.openhcs.example/introspect",
        "introspection_client_id": "openhcs-resource-server",
        "introspection_client_secret": "secret",
        "tenant_subject": "tenant-user-1",
        "required_scopes": ("openhcs:use",),
        "allowed_hosts": ("mcp.openhcs.example",),
    }
    values.update(overrides)
    return McpHttpResourceServerSettings(**values)


def test_http_settings_reject_insecure_public_url():
    with pytest.raises(McpHttpConfigurationError, match="HTTPS"):
        _settings(public_url="http://mcp.openhcs.example/mcp")


def test_http_settings_allow_explicit_loopback_development():
    settings = _settings(
        public_url="http://127.0.0.1:8000/mcp",
        issuer_url="http://127.0.0.1:9000",
        introspection_url="http://127.0.0.1:9000/introspect",
        allowed_hosts=("127.0.0.1:8000",),
        allow_insecure_loopback=True,
    )

    assert settings.streamable_http_path == "/mcp"


@pytest.mark.asyncio
async def test_introspection_verifier_accepts_only_bound_token():
    settings = _settings()
    payload = {
        "active": True,
        "iss": settings.issuer_url,
        "sub": settings.tenant_subject,
        "aud": [settings.public_url],
        "exp": int(time.time()) + 60,
        "scope": "openhcs:use profile",
        "client_id": "codex-client",
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == settings.introspection_url
        assert b"token=opaque-token" in await request.aread()
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    def client_factory(**kwargs):
        kwargs["transport"] = transport
        return real_client(**kwargs)

    access = await IntrospectionTokenVerifier(
        settings,
        http_client_factory=client_factory,
    ).verify_token("opaque-token")

    assert access is not None
    assert access.subject == settings.tenant_subject
    assert access.resource == settings.public_url
    assert access.scopes == ["openhcs:use", "profile"]
    assert access.token == "opaque-token"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("active", False),
        ("iss", "https://wrong.example"),
        ("sub", "other-tenant"),
        ("aud", ["https://other.example/mcp"]),
        ("exp", 1),
        ("scope", "profile"),
        ("client_id", ""),
    ],
)
async def test_introspection_verifier_fails_closed(field, value):
    settings = _settings()
    payload = {
        "active": True,
        "iss": settings.issuer_url,
        "sub": settings.tenant_subject,
        "aud": [settings.public_url],
        "exp": int(time.time()) + 60,
        "scope": "openhcs:use",
        "client_id": "claude-client",
    }
    payload[field] = value

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    def client_factory(**kwargs):
        kwargs["transport"] = transport
        return real_client(**kwargs)

    assert (
        await IntrospectionTokenVerifier(
            settings,
            http_client_factory=client_factory,
        ).verify_token("token")
        is None
    )
