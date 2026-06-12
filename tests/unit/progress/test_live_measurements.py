import pytest

from openhcs.core.artifacts import ArtifactKind, ArtifactOutputPlan
from openhcs.core.progress import ProgressEvent, ProgressPhase, ProgressStatus
from openhcs.core.progress.live_measurements import (
    LiveMeasurementPayloadError,
    LiveMeasurementProgressPayload,
)
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.runtime_values import normalize_artifact_value


def _measurement_record(
    *,
    name: str = "MeasureObjectIntensity",
    rows=None,
    path: str = "/memory/measurements.pkl",
):
    store = RuntimeValueStore()
    value = normalize_artifact_value(
        ArtifactOutputPlan(
            name=name,
            path=path,
            kind=ArtifactKind.MEASUREMENTS,
            group_keys=("DAPI",),
        ),
        rows
        if rows is not None
        else (
            {"object_id": 1, "mean": 10.0, "nullable": None},
            {"object_id": 2, "mean": 20.0, "nullable": "x"},
        ),
        axis_id="A01",
    )
    return store.record(value, path=path, backend="memory")


def _event_with_context(context):
    return ProgressEvent(
        execution_id="exec-1",
        plate_id="/tmp/plate",
        axis_id="A01",
        step_name="Measure",
        phase=ProgressPhase.STEP_COMPLETED,
        status=ProgressStatus.SUCCESS,
        percent=100.0,
        completed=1,
        total=1,
        timestamp=1.0,
        pid=1234,
        context=context,
    )


def test_live_measurement_payload_round_trips_through_progress_context():
    payload = LiveMeasurementProgressPayload.from_records((_measurement_record(),))
    assert payload is not None

    event = _event_with_context(payload.to_context())
    decoded_event = ProgressEvent.from_dict(event.to_dict())
    decoded_payload = LiveMeasurementProgressPayload.from_context(decoded_event.context)

    assert decoded_payload is not None
    assert decoded_payload.preview_count == 1
    preview = decoded_payload.previews[0]
    assert preview.address.name == "MeasureObjectIntensity"
    assert preview.address.axis_id == "A01"
    assert preview.address.group_key == "DAPI"
    assert preview.columns == ("object_id", "mean", "nullable")
    assert preview.rows[0]["mean"] == 10.0
    assert preview.rows[0]["nullable"] is None


def test_live_measurement_payload_truncates_rows_columns_and_previews():
    records = tuple(
        _measurement_record(
            name=f"measurements_{index}",
            path=f"/memory/measurements_{index}.pkl",
            rows=(
                {"a": 1, "b": 2, "c": 3},
                {"a": 4, "b": 5, "c": 6},
            ),
        )
        for index in range(3)
    )

    payload = LiveMeasurementProgressPayload.from_records(
        records,
        row_limit=1,
        column_limit=2,
        preview_limit=2,
    )

    assert payload is not None
    assert payload.preview_count == 3
    assert payload.truncated_previews is True
    assert len(payload.previews) == 2
    assert payload.previews[0].columns == ("a", "b")
    assert payload.previews[0].rows == ({"a": 1, "b": 2},)
    assert payload.previews[0].truncated_rows is True
    assert payload.previews[0].truncated_columns is True


def test_live_measurement_payload_absent_context_is_noop():
    assert LiveMeasurementProgressPayload.from_context(None) is None
    assert LiveMeasurementProgressPayload.from_context({}) is None


def test_live_measurement_payload_malformed_context_fails_loudly():
    with pytest.raises(LiveMeasurementPayloadError, match="must be a mapping"):
        LiveMeasurementProgressPayload.from_context({"live_measurements": []})
