from dataclasses import dataclass

import pytest

from openhcs.constants.constants import AllComponents
from openhcs.core.artifacts import ArtifactOutputPlan, MeasurementsArtifactType
from openhcs.core.progress import (
    ProgressEvent,
    ProgressIdentity,
    ProgressPhase,
    ProgressStatus,
)
from openhcs.core.progress.live_measurements import (
    LiveMeasurementPayloadError,
    LiveMeasurementProgressPayload,
)
from openhcs.core.measurement_row_materialization import (
    DataclassMeasurementColumnarRows,
)
from openhcs.core.runtime_artifact_values import RuntimeValue
from openhcs.core.runtime_measurements import (
    MeasurementScope,
    MeasurementSubject,
    MeasurementTable,
)
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.runtime_tabular_values import (
    ColumnarRows,
    FieldSpec,
)


@dataclass(frozen=True, slots=True)
class _MeasurementPreviewRow:
    object_id: int
    mean: float
    nullable: str | None


@dataclass(frozen=True, slots=True)
class _TruncatedPreviewRow:
    a: int
    b: int
    c: int


@dataclass(frozen=True, slots=True)
class _LargePreviewRow:
    object_id: int
    mean: float


class _AccessTrackingColumnarRows(ColumnarRows):
    def __init__(self, *, allowed_columns: tuple[str, ...]):
        self._column_names = (
            "runtime_z",
            "runtime_a",
            "discarded_runtime_column",
        )
        self._columns = {
            "runtime_z": (1, 2),
            "runtime_a": (3, 4),
            "discarded_runtime_column": (5, 6),
        }
        self._allowed_columns = frozenset(allowed_columns)
        self.requested_columns: list[str] = []

    @property
    def fields(self):
        return tuple(FieldSpec(name) for name in self._column_names)

    @property
    def columns(self):
        return self._column_names

    def column_values(self, column: str):
        self.requested_columns.append(column)
        if column not in self._allowed_columns:
            raise AssertionError(f"discarded column {column!r} was materialized")
        return self._columns[column]

    def row_count(self):
        return 2


def _measurement_record(
    *,
    name: str = "MeasureObjectIntensity",
    rows: ColumnarRows | None = None,
    path: str = "/memory/measurements.pkl",
):
    store = RuntimeValueStore()
    table_rows = (
        rows
        if rows is not None
        else DataclassMeasurementColumnarRows(
            (
                _MeasurementPreviewRow(1, 10.0, None),
                _MeasurementPreviewRow(2, 20.0, "x"),
            )
        )
    )
    value = RuntimeValue.normalize(
        ArtifactOutputPlan(
            name=name,
            path=path,
            artifact_type=MeasurementsArtifactType,
            group_keys=("DAPI",),
            group_component=AllComponents.CHANNEL,
        ),
        MeasurementTable(
            name=name,
            rows=table_rows,
            subject=MeasurementSubject(MeasurementScope.OBJECT, "Objects"),
        ),
        axis_id="A01",
    )
    return store.record(value, path=path, backend="memory")


def _event_with_context(context):
    return ProgressEvent(
        identity=ProgressIdentity(
            execution_id="exec-1",
            plate_id="/tmp/plate",
            axis_id="A01",
            step_name="Measure",
        ),
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
    assert preview.address.key.name == "MeasureObjectIntensity"
    assert preview.address.key.scope.axis_id == "A01"
    assert preview.address.key.scope.value_text == "DAPI"
    assert preview.columns == ("object_id", "mean", "nullable")
    assert preview.rows[0]["mean"] == 10.0
    assert preview.rows[0]["nullable"] is None


def test_live_measurement_payload_truncates_rows_columns_and_previews():
    records = tuple(
        _measurement_record(
            name=f"measurements_{index}",
            path=f"/memory/measurements_{index}.pkl",
            rows=DataclassMeasurementColumnarRows(
                (
                    _TruncatedPreviewRow(1, 2, 3),
                    _TruncatedPreviewRow(4, 5, 6),
                )
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


def test_live_measurement_payload_previews_columnar_rows_without_materializing_all_rows():
    class ExplodingColumnarRows(ColumnarRows):
        @property
        def fields(self):
            return FieldSpec.from_dataclass_type(_LargePreviewRow)

        @property
        def columns(self):
            return dict(
                zip(
                    (field.name for field in self.fields),
                    (
                        tuple(range(100)),
                        tuple(float(index) for index in range(100)),
                    ),
                    strict=True,
                )
            )

        def row_mappings(self):
            raise AssertionError("live preview should not materialize all rows")

    payload = LiveMeasurementProgressPayload.from_records(
        (
            _measurement_record(
                rows=ExplodingColumnarRows(),
            ),
        ),
        row_limit=2,
    )

    assert payload is not None
    preview = payload.previews[0]
    assert preview.row_count == 100
    assert preview.truncated_rows is True
    assert preview.columns == ("object_id", "mean")
    assert preview.rows == (
        {"object_id": 0, "mean": 0.0},
        {"object_id": 1, "mean": 1.0},
    )


@pytest.mark.parametrize(
    ("column_limit", "expected_columns", "expected_row"),
    (
        (2, ("runtime_z", "runtime_a"), {"runtime_z": 1, "runtime_a": 3}),
        (0, (), {}),
        (-1, ("runtime_z", "runtime_a"), {"runtime_z": 1, "runtime_a": 3}),
    ),
)
def test_live_measurement_payload_pushes_column_limit_before_materialization(
    column_limit,
    expected_columns,
    expected_row,
):
    rows = _AccessTrackingColumnarRows(allowed_columns=expected_columns)

    payload = LiveMeasurementProgressPayload.from_records(
        (_measurement_record(rows=rows),),
        row_limit=1,
        column_limit=column_limit,
    )

    assert payload is not None
    preview = payload.previews[0]
    assert preview.columns == expected_columns
    assert preview.rows == (expected_row,)
    assert preview.row_count == 2
    assert preview.truncated_rows is True
    assert preview.truncated_columns is True
    assert tuple(rows.requested_columns) == expected_columns


def test_live_measurement_payload_absent_context_is_noop():
    assert LiveMeasurementProgressPayload.from_context(None) is None
    assert LiveMeasurementProgressPayload.from_context({}) is None


def test_live_measurement_payload_malformed_context_fails_loudly():
    with pytest.raises(LiveMeasurementPayloadError, match="must be a mapping"):
        LiveMeasurementProgressPayload.from_context({"live_measurements": []})
