"""Shared fail-fast timeout policy for MCP control endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


DEFAULT_MCP_CONTROL_TIMEOUT_MS = 750
MAX_MCP_CONTROL_TIMEOUT_MS = 2_000
MIN_MCP_CONTROL_TIMEOUT_MS = 1


@dataclass(frozen=True, slots=True)
class BoundedMcpTimeoutPolicy:
    label: str
    default_ms: int
    min_ms: int
    max_ms: int

    def resolve(self, requested_timeout_ms: int | None) -> int:
        if requested_timeout_ms is None:
            return self.default_ms
        if requested_timeout_ms < self.min_ms:
            raise ValueError(
                f"{self.label} MCP timeout must be at least {self.min_ms}ms."
            )
        if requested_timeout_ms > self.max_ms:
            raise ValueError(
                f"{self.label} MCP timeout must not exceed {self.max_ms}ms."
            )
        return requested_timeout_ms


class McpControlTimeoutPolicy:
    """Shared fail-fast timeout contract for Codex-facing MCP control tools."""

    label: ClassVar[str]
    default_ms: ClassVar[int] = DEFAULT_MCP_CONTROL_TIMEOUT_MS
    min_ms: ClassVar[int] = MIN_MCP_CONTROL_TIMEOUT_MS
    max_ms: ClassVar[int] = MAX_MCP_CONTROL_TIMEOUT_MS

    @classmethod
    def resolve(cls, requested_timeout_ms: int | None) -> int:
        return BoundedMcpTimeoutPolicy(
            label=cls.label,
            default_ms=cls.default_ms,
            min_ms=cls.min_ms,
            max_ms=cls.max_ms,
        ).resolve(requested_timeout_ms)


class McpUiBridgeTimeoutPolicy(McpControlTimeoutPolicy):
    """Fail-fast timeout contract for Codex-facing UI bridge tools."""

    label = "UI bridge"


class McpUiBridgeCommandTimeoutPolicy(McpControlTimeoutPolicy):
    """Bounded timeout contract for UI bridge commands."""

    label = "UI bridge command"

    @classmethod
    def resolve(cls, requested_timeout_ms: int | None) -> int:
        if requested_timeout_ms is None:
            requested_timeout_ms = cls.max_ms
        return super().resolve(requested_timeout_ms)


class McpViewerTimeoutPolicy(McpControlTimeoutPolicy):
    """Fail-fast timeout contract for viewer state and snapshot tools."""

    label = "Viewer"


class McpViewerCommandTimeoutPolicy(McpViewerTimeoutPolicy):
    """Bounded timeout contract for viewer state-mutating commands."""

    @classmethod
    def resolve(cls, requested_timeout_ms: int | None) -> int:
        if requested_timeout_ms is None:
            requested_timeout_ms = cls.max_ms
        return super().resolve(requested_timeout_ms)
