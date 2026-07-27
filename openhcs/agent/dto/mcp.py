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
    server_current_mtime_ns: int | None
    server_source_changed_since_import: bool
    stale_source_paths: tuple[str, ...]
    recovery_reason: str
    installation_pointer_path: str | None
    installation_pointer_changed_since_import: bool
    installation_pointer_available: bool | None
    restart_required: bool
    restart_command: tuple[str, ...]
    restart_command_is_stable: bool
    reconnect_required: bool
    reconnect_owner: str | None
    retry_after_reconnect: bool
    automatic_recovery_on_reconnect: bool
    restart_hint: str | None
