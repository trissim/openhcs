"""Shared fail-fast timeout policy for MCP control endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from openhcs.agent.dto.viewer import VIEWER_WINDOW_CONTROL_TIMEOUT_MS_DEFAULT
from openhcs.agent.services.ui_bridge_service import DEFAULT_UI_BRIDGE_TIMEOUT_MS

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
    """Shared bounds mechanism for declaration-owned MCP control timeouts."""

    label: ClassVar[str]
    default_ms: ClassVar[int]
    min_ms: ClassVar[int] = MIN_MCP_CONTROL_TIMEOUT_MS
    max_ms: ClassVar[int]

    @classmethod
    def resolve(cls, requested_timeout_ms: int | None) -> int:
        return BoundedMcpTimeoutPolicy(
            label=cls.label,
            default_ms=cls.default_ms,
            min_ms=cls.min_ms,
            max_ms=cls.max_ms,
        ).resolve(requested_timeout_ms)


class McpUiBridgeTimeoutPolicy(McpControlTimeoutPolicy):
    """MCP projection of the UI bridge connection timeout contract."""

    label = "UI bridge"
    default_ms = DEFAULT_UI_BRIDGE_TIMEOUT_MS
    max_ms = DEFAULT_UI_BRIDGE_TIMEOUT_MS


class McpUiBridgeCommandTimeoutPolicy(McpUiBridgeTimeoutPolicy):
    """UI bridge command profile with the bridge-owned timeout bounds."""

    label = "UI bridge command"


class McpViewerTimeoutPolicy(McpControlTimeoutPolicy):
    """MCP projection of the viewer control timeout contract."""

    label = "Viewer"
    default_ms = VIEWER_WINDOW_CONTROL_TIMEOUT_MS_DEFAULT
    max_ms = VIEWER_WINDOW_CONTROL_TIMEOUT_MS_DEFAULT


class McpViewerCommandTimeoutPolicy(McpViewerTimeoutPolicy):
    """Viewer command profile with the viewer-owned timeout bounds."""
