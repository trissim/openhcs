"""MCP boundary DTOs for the headless OpenHCS agent API."""

from __future__ import annotations

from dataclasses import dataclass

from openhcs.agent.dto.common import AgentTimedStatusEnvelope


@dataclass(frozen=True, slots=True)
class McpServerHealthResult(AgentTimedStatusEnvelope):
    """Health payload returned by the OpenHCS MCP transport boundary."""

    service: str
    server_process_id: int
    server_source_path: str
    server_import_mtime_ns: int
    server_current_mtime_ns: int
    server_source_changed_since_import: bool
    stale_source_paths: tuple[str, ...] = ()
