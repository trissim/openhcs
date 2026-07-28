"""Tests for the hosted MCP OAuth resource-server boundary."""

import time

import httpx
import pytest

from openhcs.mcp.http_auth import (
    IntrospectionTokenVerifier,
    McpHttpAuthenticationMode,
    McpHttpConfigurationError,
    McpHttpOAuthSettings,
    McpHttpResourceServerSettings,
)


def _oauth(**overrides):
    values = {
        "issuer_url": "https://auth.openhcs.example",
        "introspection_url": "https://auth.openhcs.example/introspect",
        "introspection_client_id": "openhcs-resource-server",
        "introspection_client_secret": "secret",
        "tenant_subject": "tenant-user-1",
        "required_scopes": ("openhcs:use",),
    }
    values.update(overrides)
    return McpHttpOAuthSettings(**values)


def _settings(**overrides):
    values = {
        "public_url": "https://mcp.openhcs.example/mcp",
        "allowed_hosts": ("mcp.openhcs.example",),
        "oauth": _oauth(),
    }
    values.update(overrides)
    return McpHttpResourceServerSettings(**values)


def test_http_settings_reject_insecure_public_url():
    with pytest.raises(McpHttpConfigurationError, match="HTTPS"):
        _settings(public_url="http://mcp.openhcs.example/mcp")


@pytest.mark.parametrize(
    "public_url",
    (
        "https://user:secret@mcp.openhcs.example/mcp",
        "https://mcp.openhcs.example/mcp?token=secret",
        "https://mcp.openhcs.example/mcp#fragment",
        "https://mcp.openhcs.example/mcp\nheader",
    ),
)
def test_http_settings_reject_ambiguous_or_credential_bearing_public_urls(
    public_url,
):
    with pytest.raises(McpHttpConfigurationError):
        _settings(public_url=public_url)


@pytest.mark.parametrize(
    "allowed_hosts",
    (("",), ("mcp.openhcs.example/path",), (" mcp.openhcs.example",)),
)
def test_http_settings_reject_invalid_allowed_hosts(allowed_hosts):
    with pytest.raises(McpHttpConfigurationError, match="Allowed Host"):
        _settings(allowed_hosts=allowed_hosts)


def test_http_settings_allow_explicit_loopback_development():
    settings = _settings(
        public_url="http://127.0.0.1:8000/mcp",
        oauth=_oauth(
            issuer_url="http://127.0.0.1:9000",
            introspection_url="http://127.0.0.1:9000/introspect",
        ),
        allowed_hosts=("127.0.0.1:8000",),
        allow_insecure_loopback=True,
    )

    assert settings.streamable_http_path == "/mcp"


def test_public_read_only_settings_need_no_oauth_credentials():
    settings = _settings(oauth=None)

    assert settings.authentication_mode is McpHttpAuthenticationMode.PUBLIC_READ_ONLY
    assert settings.authentication_mode.requires_oauth is False
    assert settings.authentication_mode.access_qualifier == "public"
    assert settings.required_scopes == ()
    assert settings.tenant_subject is None
    assert settings.authentication_mode.tool_security_schemes(
        settings.required_scopes
    ) == ({"type": "noauth"},)


def test_oauth_settings_project_auth_facets_from_the_mode_owner():
    settings = _settings()

    assert settings.authentication_mode is McpHttpAuthenticationMode.OAUTH_INTROSPECTION
    assert settings.authentication_mode.requires_oauth is True
    assert (
        settings.authentication_mode.access_qualifier
        == "authenticated, subject-isolated"
    )
    assert settings.required_scopes == ("openhcs:use",)
    assert settings.tenant_subject == "tenant-user-1"
    assert settings.authentication_mode.tool_security_schemes(
        settings.required_scopes
    ) == ({"type": "oauth2", "scopes": ["openhcs:use"]},)


def test_domain_challenge_rejects_whitespace():
    with pytest.raises(McpHttpConfigurationError, match="challenge token"):
        _settings(openai_domain_challenge_token="not valid")


def test_environment_requires_explicit_authentication_mode(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "OPENHCS_MCP_HTTP_PUBLIC_URL",
        "https://mcp.openhcs.example/mcp",
    )
    monkeypatch.setenv("OPENHCS_AGENT_READ_ROOTS", str(tmp_path))
    monkeypatch.setenv("OPENHCS_AGENT_WRITE_ROOTS", str(tmp_path))

    with pytest.raises(McpHttpConfigurationError, match="AUTH_MODE"):
        McpHttpResourceServerSettings.from_environment()


def test_environment_builds_public_read_only_settings(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "OPENHCS_MCP_HTTP_PUBLIC_URL",
        "https://mcp.openhcs.example/mcp",
    )
    monkeypatch.setenv("OPENHCS_MCP_HTTP_AUTH_MODE", "public_read_only")
    monkeypatch.setenv("OPENHCS_AGENT_READ_ROOTS", str(tmp_path))
    monkeypatch.setenv("OPENHCS_AGENT_WRITE_ROOTS", str(tmp_path))
    monkeypatch.setenv(
        "OPENHCS_MCP_HTTP_OPENAI_DOMAIN_CHALLENGE_TOKEN",
        "openai-domain-token",
    )

    settings = McpHttpResourceServerSettings.from_environment()

    assert settings.oauth is None
    assert settings.authentication_mode is McpHttpAuthenticationMode.PUBLIC_READ_ONLY
    assert settings.openai_domain_challenge_token == "openai-domain-token"


@pytest.mark.asyncio
async def test_introspection_verifier_accepts_only_bound_token():
    settings = _settings()
    assert settings.oauth is not None
    payload = {
        "active": True,
        "iss": settings.oauth.issuer_url,
        "sub": settings.oauth.tenant_subject,
        "aud": [settings.public_url],
        "exp": int(time.time()) + 60,
        "scope": "openhcs:use profile",
        "client_id": "codex-client",
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == settings.oauth.introspection_url
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
    assert access.subject == settings.oauth.tenant_subject
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
    assert settings.oauth is not None
    payload = {
        "active": True,
        "iss": settings.oauth.issuer_url,
        "sub": settings.oauth.tenant_subject,
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


def test_public_settings_cannot_construct_token_verifier():
    with pytest.raises(McpHttpConfigurationError, match="requires OAuth"):
        IntrospectionTokenVerifier(_settings(oauth=None))
