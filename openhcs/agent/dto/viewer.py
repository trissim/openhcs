"""DTOs for agent-facing viewer window interactions."""

from __future__ import annotations

from dataclasses import dataclass, field

from openhcs.agent.dto.common import (
    AgentError,
    AgentResourceRef,
    AgentResultEnvelope,
    JsonObject,
    SCHEMA_VERSION,
)
from openhcs.agent.dto.execution import ExecutionConnectionProjection, ExecutionConnectionSpec
from openhcs.runtime.window_snapshot import WindowSnapshotCaptureSpec


@dataclass(frozen=True, slots=True)
class ViewerWindowDescriptor:
    """Agent-facing identity for the viewer window that produced a resource."""

    viewer_type: str
    title: str


@dataclass(frozen=True, slots=True)
class ViewerWindowSnapshotRequest(ExecutionConnectionProjection):
    snapshot: WindowSnapshotCaptureSpec
    timeout_ms: int = 5000


@dataclass(frozen=True, slots=True)
class ViewerWindowSnapshotResult(AgentResultEnvelope, ExecutionConnectionProjection):
    captured: bool
    resource: AgentResourceRef | None = None
    viewer: ViewerWindowDescriptor | None = None
    width: int | None = None
    height: int | None = None
    snapshot: WindowSnapshotCaptureSpec | None = None
    response: JsonObject = field(default_factory=dict)


def viewer_window_snapshot_error(
    *,
    connection: ExecutionConnectionSpec,
    error: AgentError,
) -> ViewerWindowSnapshotResult:
    return ViewerWindowSnapshotResult(
        schema_version=SCHEMA_VERSION,
        connection=connection,
        captured=False,
        errors=(error,),
    )
