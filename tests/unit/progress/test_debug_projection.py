import pytest

from openhcs.core.debug import (
    DebugCursor,
    DebugEventType,
    DebugProgressContext,
    DebugSession,
    DebugSnapshot,
    DebugTerminalSummary,
    DebugTimelineNodeState,
)
from openhcs.core.progress import (
    ProgressEvent,
    ProgressIdentity,
    ProgressPhase,
    ProgressStatus,
)
from openhcs.core.progress.debug_projection import (
    DebugProgressRecord,
    RuntimeProjectionBuilder,
    RuntimeProjectionSource,
)


def _event(
    *,
    context=None,
    timestamp: float = 1.0,
) -> ProgressEvent:
    return ProgressEvent(
        identity=ProgressIdentity(
            execution_id="exec-1",
            plate_id="/tmp/plate",
            axis_id="A01",
            step_name="segment",
        ),
        phase=ProgressPhase.PATTERN_GROUP,
        status=ProgressStatus.STARTED,
        percent=50.0,
        completed=1,
        total=2,
        timestamp=timestamp,
        pid=1234,
        context=context,
    )


def _cursor() -> DebugCursor:
    return DebugCursor(
        step_index=1,
        step_scope_id="step-1",
        group_key="default",
        invocation_key="default:0:segment",
    )


def _debug_context(
    *,
    event_type: DebugEventType = DebugEventType.BEFORE_INVOCATION,
) -> dict:
    return DebugProgressContext(
        debug_session_id="debug-1",
        snapshot_id="snapshot-1",
        cursor=_cursor(),
        event_type=event_type,
        snapshot_store_ref="/tmp/debug",
    ).to_progress_context()


def test_debug_progress_record_ignores_normal_progress_context() -> None:
    assert DebugProgressRecord.from_progress_event(_event(context=None)) is None
    assert (
        DebugProgressRecord.from_progress_event(
            _event(context={"live_measurements": {}})
        )
        is None
    )


def test_debug_progress_record_fails_loudly_for_malformed_debug_context() -> None:
    with pytest.raises(KeyError):
        DebugProgressRecord.from_progress_event(
            _event(
                context={
                    DebugProgressContext.progress_context_discriminator: "debug-1"
                }
            )
        )


def test_runtime_projection_builder_projects_debug_records_from_same_snapshot() -> None:
    event = _event(
        context=_debug_context(event_type=DebugEventType.AFTER_INVOCATION),
        timestamp=2.0,
    )
    session = DebugSession(
        debug_session_id="debug-1",
        execution_id="exec-1",
        plate_id="/tmp/plate",
    ).with_cursor(_cursor())

    bundle = RuntimeProjectionBuilder().build(
        RuntimeProjectionSource(
            events_by_execution={"exec-1": [event]},
            session=session,
            snapshots=(
                DebugSnapshot(
                    snapshot_id="snapshot-1",
                    cursor=_cursor(),
                    step_name="snapshot segment",
                    callable_name="segment_callable",
                ),
            ),
        )
    )

    assert bundle.execution.get_plate("/tmp/plate", "exec-1") is not None
    assert len(bundle.debug.records) == 1
    assert bundle.debug.debug_session_id == session.debug_session_id
    assert bundle.debug.current_frame is not None
    assert bundle.debug.current_frame.cursor == _cursor()
    assert bundle.debug.current_frame.snapshot_id == "snapshot-1"
    assert bundle.debug.current_frame.step_name == "snapshot segment"
    assert bundle.debug.current_frame.callable_name == "segment_callable"
    assert (
        bundle.debug.node_state_for_cursor(cursor=_cursor())
        is DebugTimelineNodeState.COMPLETED
    )


def test_terminal_debug_projection_keeps_last_frame_without_active_frame() -> None:
    event = _event(context=_debug_context(), timestamp=2.0)
    terminal_summary = DebugTerminalSummary(
        debug_session_id="debug-1",
        plate_id="/tmp/plate",
        terminal_status="complete",
        cursor=_cursor(),
    )

    bundle = RuntimeProjectionBuilder().build(
        RuntimeProjectionSource(
            events_by_execution={"exec-1": [event]},
            terminal_summary=terminal_summary,
        )
    )

    assert bundle.debug.current_frame is None
    assert bundle.debug.last_frame is not None
    assert bundle.debug.debug_session_id == "debug-1"
