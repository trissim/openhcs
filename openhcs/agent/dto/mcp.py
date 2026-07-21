"""MCP boundary DTOs for the headless OpenHCS agent API."""

from __future__ import annotations

from dataclasses import dataclass

from openhcs.agent.dto.common import AgentTimedStatusEnvelope


@dataclass(frozen=True, slots=True)
class McpServerHealthResult(AgentTimedStatusEnvelope):
    """Health payload returned by the OpenHCS MCP transport boundary."""

    service: str
    openhcs_version: str
    packaged_resources_ready: bool
    packaged_resource_count: int
    missing_packaged_resource_paths: tuple[str, ...]
    server_process_id: int
    server_source_path: str
    server_import_mtime_ns: int
    server_current_mtime_ns: int
    server_source_changed_since_import: bool
    stale_source_paths: tuple[str, ...] = ()
    restart_required: bool = False
    restart_command: tuple[str, ...] = ()
    restart_hint: str | None = None
