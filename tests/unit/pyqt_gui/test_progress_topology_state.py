import pytest

from openhcs.core.progress import (
    ProgressEvent,
    ProgressIdentity,
    ProgressPhase,
    ProgressStatus,
)
from openhcs.core.progress.runtime_tree import RuntimeExecutionTopology


def _event(
    *,
    execution_id: str = "exec-1",
    plate_id: str = "/tmp/plate",
    axis_id: str = "A01",
    phase: ProgressPhase = ProgressPhase.STEP_STARTED,
    timestamp: float = 1.0,
    worker_slot: str | None = "worker_0",
    owned_wells: list[str] | None = None,
    worker_assignments: dict[str, list[str]] | None = None,
    total_wells: list[str] | None = None,
    step_names: list[str] | None = None,
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
        timestamp=timestamp,
        pid=1,
        worker_slot=worker_slot,
        owned_wells=owned_wells,
        worker_assignments=worker_assignments,
        total_wells=total_wells,
        step_names=step_names,
    )


def test_runtime_topology_is_derived_from_retained_event_snapshot():
    init_event = _event(
        axis_id="",
        phase=ProgressPhase.INIT,
        worker_slot=None,
        worker_assignments={"worker_0": ["A01"]},
        total_wells=["A01", "B01"],
        step_names=["normalize", "segment"],
    )
    worker_event = _event(
        timestamp=2.0,
        owned_wells=["A01"],
    )

    topology = RuntimeExecutionTopology.from_events(
        {"exec-1": [worker_event, init_event]}
    )

    key = ("exec-1", "/tmp/plate")
    assert topology.worker_assignments[key] == {"worker_0": ("A01",)}
    assert topology.known_wells[key] == ("A01", "B01")
    assert topology.step_names[("exec-1", "/tmp/plate", "A01")] == {
        0: "normalize",
        1: "segment",
    }


def test_runtime_topology_rejects_worker_claim_mismatch():
    init_event = _event(
        axis_id="",
        phase=ProgressPhase.INIT,
        worker_slot=None,
        worker_assignments={"worker_0": ["A01"]},
    )
    bad_event = _event(timestamp=2.0, owned_wells=["B01"])

    with pytest.raises(ValueError, match="Worker claim mismatch"):
        RuntimeExecutionTopology.from_events({"exec-1": [init_event, bad_event]})


def test_runtime_topology_allows_late_subscription_without_init_event():
    topology = RuntimeExecutionTopology.from_events(
        {"exec-1": [_event(worker_slot=None)]}
    )

    assert topology.worker_assignments == {}
    assert topology.known_wells == {}
    assert topology.step_names == {}
