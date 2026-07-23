import pytest

from openhcs.core.progress import (
    ProgressEvent,
    ProgressIdentity,
    ProgressPhase,
    ProgressStatus,
)
from openhcs.pyqt_gui.widgets.shared.server_browser import ProgressTopologyState


def _event(
    *,
    execution_id: str = "exec-1",
    plate_id: str = "/tmp/plate",
    axis_id: str = "A01",
    phase: ProgressPhase = ProgressPhase.STEP_STARTED,
    worker_slot: str | None = "worker_0",
    owned_wells: list[str] | None = None,
    worker_assignments: dict[str, list[str]] | None = None,
    total_wells: list[str] | None = None,
) -> ProgressEvent:
    return ProgressEvent(
        identity=ProgressIdentity(
            execution_id=execution_id,
            plate_id=plate_id,
            axis_id=axis_id,
            step_name="s",
        ),
        phase=phase,
        status=ProgressStatus.RUNNING,
        percent=0.0,
        completed=0,
        total=1,
        timestamp=1.0,
        pid=1,
        worker_slot=worker_slot,
        owned_wells=owned_wells,
        worker_assignments=worker_assignments,
        total_wells=total_wells,
    )


def test_progress_topology_state_tracks_assignments_and_wells():
    state = ProgressTopologyState()
    event = _event(
        worker_assignments={"worker_0": ["A01"]},
        owned_wells=["A01"],
        total_wells=["A01", "B01"],
    )

    state.register_event(event)

    key = ("exec-1", "/tmp/plate")
    assert state.worker_assignments[key] == {"worker_0": ["A01"]}
    assert state.known_wells[key] == ["A01", "B01"]
    assert "exec-1" in state.seen_execution_ids


def test_progress_topology_state_rejects_claim_mismatch():
    state = ProgressTopologyState()
    seed_event = _event(
        worker_assignments={"worker_0": ["A01"]},
        owned_wells=["A01"],
    )
    state.register_event(seed_event)

    bad_event = _event(
        worker_assignments=None,
        owned_wells=["B01"],
    )
    with pytest.raises(ValueError, match="Worker claim mismatch"):
        state.register_event(bad_event)
