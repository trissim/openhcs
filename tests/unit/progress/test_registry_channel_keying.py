from openhcs.core.progress import (
    ProgressChannel,
    ProgressChannelDeclarationBase,
    ProgressEvent,
    ProgressIdentity,
    ProgressPhaseDeclarationBase,
    ProgressPhase,
    ProgressStatusDeclarationBase,
    ProgressStatus,
    registry,
)


def _event(
    *,
    phase: ProgressPhase,
    status: ProgressStatus,
    percent: float,
    axis_id: str = "A01",
    step_name: str = "pipeline",
    completed: int = 0,
    total: int = 1,
) -> ProgressEvent:
    return ProgressEvent(
        identity=ProgressIdentity(
            execution_id="exec-1",
            plate_id="/tmp/plate",
            axis_id=axis_id,
            step_name=step_name,
        ),
        phase=phase,
        status=status,
        percent=percent,
        completed=completed,
        total=total,
        timestamp=1.0,
        pid=1111,
    )


def test_registry_keeps_pipeline_and_step_channels_separate():
    tracker = registry()
    tracker.clear_all()

    pipeline_event = _event(
        phase=ProgressPhase.STEP_COMPLETED,
        status=ProgressStatus.SUCCESS,
        percent=25.0,
        completed=1,
        total=4,
    )
    step_event = _event(
        phase=ProgressPhase.PATTERN_GROUP,
        status=ProgressStatus.RUNNING,
        percent=50.0,
        completed=1,
        total=2,
        step_name="max_project",
    )

    tracker.register_event("exec-1", pipeline_event)
    tracker.register_event("exec-1", step_event)

    events = tracker.get_events("exec-1")
    assert len(events) == 2
    phase_set = {event.phase for event in events}
    assert phase_set == {ProgressPhase.STEP_COMPLETED, ProgressPhase.PATTERN_GROUP}

    tracker.clear_all()


def test_progress_semantic_declarations_cover_wire_tokens():
    assert set(ProgressChannelDeclarationBase.__registry__) == set(ProgressChannel)
    assert set(ProgressPhaseDeclarationBase.__registry__) == set(ProgressPhase)
    assert set(ProgressStatusDeclarationBase.__registry__) == set(ProgressStatus)
