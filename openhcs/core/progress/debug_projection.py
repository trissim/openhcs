"""Debugger projection built from the normal OpenHCS progress stream."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from zmqruntime.messages import QueuedExecutionInfo, RunningExecutionInfo

from openhcs.core.debug import (
    DebugBoundaryEventDeclarationBase,
    DebugBoundaryOutcome,
    DebugBoundaryState,
    DebugCursor,
    DebugEventType,
    DebugProgressContext,
    DebugSession,
    DebugSnapshot,
    DebugTerminalSummary,
    DebugTimelineNodeState,
)
from openhcs.core.progress.types import ProgressEvent, ProgressIdentity

from .projection import (
    ExecutionRuntimeProjection,
    build_execution_runtime_projection,
)


@dataclass(frozen=True, slots=True)
class DebugProgressRecord:
    """One debug-aware progress event with parsed typed context."""

    event: ProgressEvent
    context: DebugProgressContext
    snapshot: DebugSnapshot | None = None

    @classmethod
    def from_progress_event(
        cls,
        event: ProgressEvent,
    ) -> "DebugProgressRecord | None":
        context = event.context
        if context is None:
            return None
        if not DebugProgressContext.is_progress_context(context):
            return None
        return cls(
            event=event,
            context=DebugProgressContext.from_progress_context(context),
        )

    def with_snapshot(self, snapshot: DebugSnapshot | None) -> "DebugProgressRecord":
        return DebugProgressRecord(
            event=self.event,
            context=self.context,
            snapshot=snapshot,
        )

    @property
    def session_id(self) -> str:
        return self.context.debug_session_id

    @property
    def cursor(self) -> DebugCursor:
        return self.context.cursor

    @property
    def progress_identity(self) -> ProgressIdentity:
        return self.event.identity

    @property
    def boundary(self) -> DebugBoundaryState | None:
        return self.snapshot

    @property
    def snapshot_id(self) -> str | None:
        return self.context.snapshot_id

    @property
    def step_name(self) -> str:
        boundary = self.boundary
        if boundary is not None:
            return boundary.step_name
        return self.event.step_name

    @property
    def callable_name(self) -> str | None:
        boundary = self.boundary
        if boundary is None:
            return None
        return boundary.callable_name

    @property
    def event_type(self) -> DebugEventType:
        return self.context.event_type

    @property
    def boundary_outcome(self) -> DebugBoundaryOutcome:
        return DebugBoundaryEventDeclarationBase.for_event_type(
            self.event_type
        ).boundary_outcome

    @property
    def timeline_node_state(self) -> DebugTimelineNodeState:
        return DebugBoundaryEventDeclarationBase.for_event_type(
            self.event_type
        ).timeline_node_state


@dataclass(frozen=True, slots=True)
class DebugRuntimeFrame:
    """Current or historical frame derived from one debug progress record."""

    record: DebugProgressRecord

    @property
    def cursor(self) -> DebugCursor:
        return self.record.cursor

    @property
    def event_type(self) -> DebugEventType:
        return self.record.event_type

    @property
    def snapshot_id(self) -> str | None:
        return self.record.snapshot_id

    @property
    def progress_identity(self) -> ProgressIdentity:
        return self.record.progress_identity

    @property
    def boundary(self) -> DebugBoundaryState | None:
        return self.record.boundary

    @property
    def step_name(self) -> str:
        return self.record.step_name

    @property
    def callable_name(self) -> str | None:
        return self.record.callable_name


@dataclass(frozen=True, slots=True)
class DebugTimelineNode:
    """Minimal typed timeline entry over a debug progress frame."""

    frame: DebugRuntimeFrame

    @property
    def cursor(self) -> DebugCursor:
        return self.frame.cursor

    @property
    def state(self) -> DebugTimelineNodeState:
        return self.frame.record.timeline_node_state


@dataclass(frozen=True, slots=True)
class DebugRuntimeProjection:
    """Generic debugger state projected from progress and debug contexts."""

    runtime_projection: ExecutionRuntimeProjection
    session: DebugSession | None = None
    terminal_summary: DebugTerminalSummary | None = None
    current_frame: DebugRuntimeFrame | None = None
    last_frame: DebugRuntimeFrame | None = None
    timeline: tuple[DebugTimelineNode, ...] = ()
    records: tuple[DebugProgressRecord, ...] = ()

    @classmethod
    def empty(
        cls,
        runtime_projection: ExecutionRuntimeProjection | None = None,
    ) -> "DebugRuntimeProjection":
        return cls(runtime_projection=runtime_projection or ExecutionRuntimeProjection())

    @property
    def has_active_frame(self) -> bool:
        return self.current_frame is not None

    @property
    def debug_session_id(self) -> str | None:
        if self.session is not None:
            return self.session.debug_session_id
        if self.terminal_summary is not None:
            return self.terminal_summary.debug_session_id
        if self.current_frame is not None:
            return self.current_frame.record.session_id
        if self.last_frame is not None:
            return self.last_frame.record.session_id
        return None

    @property
    def current_progress_identity(self) -> ProgressIdentity | None:
        if self.current_frame is None:
            return None
        return self.current_frame.progress_identity

    def node_state_for_cursor(
        self,
        *,
        cursor: DebugCursor,
    ) -> DebugTimelineNodeState:
        for record in reversed(self.records):
            if record.cursor == cursor:
                return record.timeline_node_state
        return DebugTimelineNodeState.PENDING


@dataclass(frozen=True, slots=True)
class DebugProjectionSource:
    """Core inputs required to build a debug runtime projection."""

    events_by_execution: Mapping[str, Sequence[ProgressEvent]]
    runtime_projection: ExecutionRuntimeProjection
    session: DebugSession | None = None
    terminal_summary: DebugTerminalSummary | None = None
    snapshots: Sequence[DebugSnapshot] = ()

    def snapshot_for_id(self, snapshot_id: str | None) -> DebugSnapshot | None:
        if snapshot_id is None:
            return None
        for snapshot in self.snapshots:
            if snapshot.snapshot_id == snapshot_id:
                return snapshot
        return None


class DebugRuntimeProjectionBuilder:
    """Build debugger state from progress events and debug progress context."""

    def build(
        self,
        source: DebugProjectionSource,
    ) -> DebugRuntimeProjection:
        records = self._records_from_events(source)
        last_frame = self._last_frame(records)
        current_frame = self._current_frame(source=source, records=records)
        return DebugRuntimeProjection(
            runtime_projection=source.runtime_projection,
            session=source.session,
            terminal_summary=source.terminal_summary,
            current_frame=current_frame,
            last_frame=last_frame,
            timeline=tuple(DebugTimelineNode(DebugRuntimeFrame(record)) for record in records),
            records=records,
        )

    def _records_from_events(
        self,
        source: DebugProjectionSource,
    ) -> tuple[DebugProgressRecord, ...]:
        session_id = self._session_id_for_source(source)
        records: list[DebugProgressRecord] = []
        for events in source.events_by_execution.values():
            for event in events:
                record = DebugProgressRecord.from_progress_event(event)
                if record is None:
                    continue
                if session_id is not None and record.session_id != session_id:
                    continue
                records.append(record.with_snapshot(source.snapshot_for_id(record.snapshot_id)))
        return tuple(sorted(records, key=lambda record: record.event.timestamp))

    def _last_frame(
        self,
        records: Sequence[DebugProgressRecord],
    ) -> DebugRuntimeFrame | None:
        if not records:
            return None
        return DebugRuntimeFrame(records[-1])

    def _current_frame(
        self,
        *,
        source: DebugProjectionSource,
        records: Sequence[DebugProgressRecord],
    ) -> DebugRuntimeFrame | None:
        if source.terminal_summary is not None:
            return None
        if not records:
            return None
        if source.session is None or source.session.cursor is None:
            return DebugRuntimeFrame(records[-1])
        for record in reversed(records):
            if record.cursor == source.session.cursor:
                return DebugRuntimeFrame(record)
        return DebugRuntimeFrame(records[-1])

    def _session_id_for_source(
        self,
        source: DebugProjectionSource,
    ) -> str | None:
        if source.session is not None:
            return source.session.debug_session_id
        if source.terminal_summary is not None:
            return source.terminal_summary.debug_session_id
        return None


@dataclass(frozen=True, slots=True)
class RuntimeProjectionSource:
    """Shared input for all core runtime projections over one event snapshot."""

    events_by_execution: Mapping[str, Sequence[ProgressEvent]]
    running_executions: Sequence[RunningExecutionInfo] = ()
    queued_executions: Sequence[QueuedExecutionInfo] = ()
    session: DebugSession | None = None
    terminal_summary: DebugTerminalSummary | None = None
    snapshots: Sequence[DebugSnapshot] = ()


@dataclass(frozen=True, slots=True)
class RuntimeProjectionBundle:
    """Core runtime projections built from the same progress event snapshot."""

    execution: ExecutionRuntimeProjection
    debug: DebugRuntimeProjection

    @classmethod
    def empty(cls) -> "RuntimeProjectionBundle":
        execution = ExecutionRuntimeProjection()
        return cls(
            execution=execution,
            debug=DebugRuntimeProjection.empty(execution),
        )


class RuntimeProjectionBuilder:
    """Build all core runtime projections from one progress event snapshot."""

    def __init__(
        self,
        *,
        debug_builder: DebugRuntimeProjectionBuilder | None = None,
    ) -> None:
        self._debug_builder = debug_builder or DebugRuntimeProjectionBuilder()

    def build(
        self,
        source: RuntimeProjectionSource,
    ) -> RuntimeProjectionBundle:
        events_by_execution = {
            execution_id: list(events)
            for execution_id, events in source.events_by_execution.items()
        }
        execution_projection = build_execution_runtime_projection(
            events_by_execution,
            running_executions=source.running_executions,
            queued_executions=source.queued_executions,
        )
        debug_projection = self._debug_builder.build(
            DebugProjectionSource(
                events_by_execution=events_by_execution,
                runtime_projection=execution_projection,
                session=source.session,
                terminal_summary=source.terminal_summary,
                snapshots=source.snapshots,
            )
        )
        return RuntimeProjectionBundle(
            execution=execution_projection,
            debug=debug_projection,
        )


def is_debug_projection_export(name: str, value: object) -> bool:
    return (
        isinstance(value, type)
        and value.__module__ == __name__
        and not name.startswith("_")
    )


__all__ = tuple(
    name
    for name, value in globals().items()
    if is_debug_projection_export(name, value)
)
