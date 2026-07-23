"""Fail-closed OAuth resource-server configuration for hosted OpenHCS MCP."""

from __future__ import annotations

import hmac
import os
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlparse


class McpHttpConfigurationError(ValueError):
    """Raised when hosted MCP security configuration is incomplete or unsafe."""


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise McpHttpConfigurationError(
            f"Required environment variable is missing: {name}"
        )
    return value


def _comma_separated_environment(name: str) -> tuple[str, ...]:
    return tuple(
        value.strip() for value in os.environ.get(name, "").split(",") if value.strip()
    )


def _require_secure_url(value: str, *, name: str, allow_loopback_http: bool) -> str:
    parsed = urlparse(value)
    loopback = parsed.hostname in {"127.0.0.1", "::1", "localhost"}
    if parsed.scheme == "https" and parsed.netloc:
        return value
    if allow_loopback_http and parsed.scheme == "http" and loopback and parsed.netloc:
        return value
    raise McpHttpConfigurationError(
        f"{name} must use HTTPS"
        + (" or explicit loopback HTTP" if allow_loopback_http else "")
        + f": {value}"
    )


@dataclass(frozen=True, slots=True)
class McpHttpResourceServerSettings:
    """Configuration for one subject-isolated hosted MCP resource server."""

    public_url: str
    issuer_url: str
    introspection_url: str
    introspection_client_id: str
    introspection_client_secret: str
    tenant_subject: str
    required_scopes: tuple[str, ...]
    allowed_hosts: tuple[str, ...]
    allowed_origins: tuple[str, ...] = ()
    bind_host: str = "127.0.0.1"
    bind_port: int = 8000
    allow_insecure_loopback: bool = False

    def __post_init__(self) -> None:
        _require_secure_url(
            self.public_url,
            name="public_url",
            allow_loopback_http=self.allow_insecure_loopback,
        )
        _require_secure_url(
            self.issuer_url,
            name="issuer_url",
            allow_loopback_http=self.allow_insecure_loopback,
        )
        _require_secure_url(
            self.introspection_url,
            name="introspection_url",
            allow_loopback_http=self.allow_insecure_loopback,
        )
        if not self.introspection_client_id or not self.introspection_client_secret:
            raise McpHttpConfigurationError(
                "OAuth introspection client credentials must be non-empty."
            )
        if not self.tenant_subject:
            raise McpHttpConfigurationError("A tenant subject is required.")
        if not self.required_scopes:
            raise McpHttpConfigurationError("At least one OAuth scope is required.")
        if not self.allowed_hosts:
            raise McpHttpConfigurationError(
                "At least one allowed Host value is required."
            )
        if not 1 <= self.bind_port <= 65535:
            raise McpHttpConfigurationError("bind_port must be between 1 and 65535.")

    @property
    def streamable_http_path(self) -> str:
        path = urlparse(self.public_url).path
        return path or "/mcp"

    @classmethod
    def from_environment(cls) -> "McpHttpResourceServerSettings":
        allow_insecure = (
            os.environ.get("OPENHCS_MCP_HTTP_ALLOW_INSECURE_LOOPBACK") == "1"
        )
        public_url = _required_environment("OPENHCS_MCP_HTTP_PUBLIC_URL")
        public_host = urlparse(public_url).netloc
        configured_hosts = _comma_separated_environment(
            "OPENHCS_MCP_HTTP_ALLOWED_HOSTS"
        )
        allowed_hosts = configured_hosts or ((public_host,) if public_host else ())
        read_roots = _required_environment("OPENHCS_AGENT_READ_ROOTS")
        write_roots = _required_environment("OPENHCS_AGENT_WRITE_ROOTS")
        if (
            not read_roots or not write_roots
        ):  # pragma: no cover - required helper guards
            raise McpHttpConfigurationError("Hosted path roots must be explicit.")
        return cls(
            public_url=public_url,
            issuer_url=_required_environment("OPENHCS_MCP_HTTP_ISSUER_URL"),
            introspection_url=_required_environment(
                "OPENHCS_MCP_HTTP_INTROSPECTION_URL"
            ),
            introspection_client_id=_required_environment(
                "OPENHCS_MCP_HTTP_INTROSPECTION_CLIENT_ID"
            ),
            introspection_client_secret=_required_environment(
                "OPENHCS_MCP_HTTP_INTROSPECTION_CLIENT_SECRET"
            ),
            tenant_subject=_required_environment("OPENHCS_MCP_HTTP_TENANT_SUBJECT"),
            required_scopes=tuple(
                _required_environment("OPENHCS_MCP_HTTP_REQUIRED_SCOPES").split()
            ),
            allowed_hosts=allowed_hosts,
            allowed_origins=_comma_separated_environment(
                "OPENHCS_MCP_HTTP_ALLOWED_ORIGINS"
            ),
            bind_host=os.environ.get("OPENHCS_MCP_HTTP_BIND_HOST", "127.0.0.1"),
            bind_port=int(os.environ.get("OPENHCS_MCP_HTTP_BIND_PORT", "8000")),
            allow_insecure_loopback=allow_insecure,
        )


class IntrospectionTokenVerifier:
    """Validate opaque bearer tokens with RFC 7662 and strict tenant binding."""

    def __init__(
        self,
        settings: McpHttpResourceServerSettings,
        *,
        http_client_factory: Callable[..., Any] | None = None,
    ):
        self.settings = settings
        self._http_client_factory = http_client_factory

    @staticmethod
    def _audiences(payload: dict[str, Any]) -> tuple[str, ...]:
        audience = payload.get("aud")
        if isinstance(audience, str):
            return (audience,)
        if isinstance(audience, list) and all(
            isinstance(value, str) for value in audience
        ):
            return tuple(audience)
        return ()

    async def verify_token(self, token: str):
        """Return MCP access metadata only for a live, scoped, tenant-bound token."""
        import httpx
        from mcp.server.auth.provider import AccessToken

        http_client_factory = self._http_client_factory or httpx.AsyncClient
        try:
            async with http_client_factory(
                auth=httpx.BasicAuth(
                    self.settings.introspection_client_id,
                    self.settings.introspection_client_secret,
                ),
                timeout=httpx.Timeout(10.0, connect=5.0),
                verify=True,
            ) as client:
                response = await client.post(
                    self.settings.introspection_url,
                    data={"token": token, "token_type_hint": "access_token"},
                    headers={"Accept": "application/json"},
                )
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError):
            return None
        if not isinstance(payload, dict) or payload.get("active") is not True:
            return None
        if not hmac.compare_digest(
            str(payload.get("iss", "")), self.settings.issuer_url
        ):
            return None
        if not hmac.compare_digest(
            str(payload.get("sub", "")),
            self.settings.tenant_subject,
        ):
            return None
        audiences = self._audiences(payload)
        if not any(
            hmac.compare_digest(audience, self.settings.public_url)
            for audience in audiences
        ):
            return None
        expires_at = payload.get("exp")
        if not isinstance(expires_at, int) or expires_at <= int(time.time()):
            return None
        scopes = tuple(str(payload.get("scope", "")).split())
        if not set(self.settings.required_scopes).issubset(scopes):
            return None
        client_id = payload.get("client_id")
        if not isinstance(client_id, str) or not client_id:
            return None
        return AccessToken(
            token=token,
            client_id=client_id,
            scopes=list(scopes),
            expires_at=expires_at,
            resource=self.settings.public_url,
            subject=self.settings.tenant_subject,
            claims={
                "iss": self.settings.issuer_url,
                "aud": list(audiences),
                "tenant_subject": self.settings.tenant_subject,
            },
        )
