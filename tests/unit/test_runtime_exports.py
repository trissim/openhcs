from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openhcs.core.artifacts import (
    ArtifactSpec,
    MeasurementsArtifactType,
)
from openhcs.core.component_group_scope import RuntimeExecutionAxisScope
from openhcs.core.measurement_row_materialization import (
    DataclassMeasurementColumnarRows,
    MeasurementProjectedColumnarRows,
)
from openhcs.core.runtime_artifact_values import (
    ArtifactKey,
    RuntimeValue,
)
from openhcs.core.runtime_exports import (
    RuntimeExportExpectation,
    RuntimeExportObservation,
    runtime_export_failures,
)
from openhcs.core.runtime_measurements import MeasurementTable
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
)
from openhcs.core.runtime_measurements import (
    MeasurementScope,
    MeasurementSubject,
)
from openhcs.core.runtime_stores import RuntimeArtifactLocation, StoredRuntimeValue
from openhcs.core.runtime_tabular_values import ColumnarRows
from openhcs.processing.materialization import CsvOptions, MaterializationSpec


def test_runtime_export_validation_accepts_header_only_empty_table(
    tmp_path: Path,
) -> None:
    table_path = tmp_path / "A01_Measurements_step1.csv"
    table_path.write_text("slice_index\n", encoding="utf-8")
    record = _stored_measurements_record()

    failures = runtime_export_failures(
        _table_export_expectation(),
        RuntimeExportObservation.from_output_root(tmp_path),
        {"A01": (record,)},
    )

    assert failures == ()


def test_runtime_export_validation_accepts_header_only_empty_column_table(
    tmp_path: Path,
) -> None:
    table_path = tmp_path / "A01_Measurements_step1.csv"
    table_path.write_text("slice_index\n", encoding="utf-8")
    record = _stored_measurements_record(
        rows=MeasurementProjectedColumnarRows(
            columns={"slice_index": ()},
            fields=(FieldSpec("slice_index", int),),
        )
    )

    failures = runtime_export_failures(
        _table_export_expectation(),
        RuntimeExportObservation.from_output_root(tmp_path),
        {"A01": (record,)},
    )

    assert failures == ()


def test_runtime_export_validation_rejects_header_only_nonempty_table(
    tmp_path: Path,
) -> None:
    table_path = tmp_path / "A01_Measurements_step1.csv"
    table_path.write_text("slice_index\n", encoding="utf-8")
    record = _stored_measurements_record(rows=_measurement_rows(0))

    failures = runtime_export_failures(
        _table_export_expectation(),
        RuntimeExportObservation.from_output_root(tmp_path),
        {"A01": (record,)},
    )

    assert failures == (f"table output {table_path} has no data rows",)


def test_runtime_export_validation_checks_table_schema_fields(
    tmp_path: Path,
) -> None:
    table_path = tmp_path / "A01_Measurements_step1.csv"
    table_path.write_text("wrong_field\n0\n", encoding="utf-8")
    record = _stored_measurements_record(rows=_measurement_rows(0))

    failures = runtime_export_failures(
        _table_export_expectation(),
        RuntimeExportObservation.from_output_root(tmp_path),
        {"A01": (record,)},
    )

    assert failures == (
        f"table output {table_path} for artifact 'Measurements' is "
        "missing schema fields ('slice_index',)",
    )


def test_runtime_export_validation_scopes_table_outputs_by_axis(
    tmp_path: Path,
) -> None:
    a01_table_path = tmp_path / "A01_Measurements_step1.csv"
    a02_table_path = tmp_path / "A02_Measurements_step1.csv"
    a01_table_path.write_text("slice_index\n", encoding="utf-8")
    a02_table_path.write_text("slice_index\n0\n", encoding="utf-8")
    a01_record = _stored_measurements_record(axis_id="A01")
    a02_record = _stored_measurements_record(
        axis_id="A02",
        rows=_measurement_rows(0),
    )

    failures = runtime_export_failures(
        _table_export_expectation(),
        RuntimeExportObservation.from_output_root(tmp_path),
        {"A01": (a01_record,), "A02": (a02_record,)},
    )

    assert failures == ()


def test_runtime_export_validation_accepts_format_compatible_schema_fields(
    tmp_path: Path,
) -> None:
    table_path = tmp_path / "A01_Measurements_step1.csv"
    table_path.write_text("SliceIndex\n0\n", encoding="utf-8")
    record = _stored_measurements_record(rows=_measurement_rows(0))

    failures = runtime_export_failures(
        _table_export_expectation(),
        RuntimeExportObservation.from_output_root(tmp_path),
        {"A01": (record,)},
    )

    assert failures == ()


@dataclass(frozen=True, slots=True)
class _RuntimeExportMeasurementRow:
    slice_index: int


def _measurement_rows(*slice_indices: int) -> DataclassMeasurementColumnarRows:
    return DataclassMeasurementColumnarRows(
        tuple(_RuntimeExportMeasurementRow(value) for value in slice_indices),
        row_type=_RuntimeExportMeasurementRow,
    )


def _stored_measurements_record(
    *,
    axis_id: str = "A01",
    rows: ColumnarRows | None = None,
) -> StoredRuntimeValue:
    if rows is None:
        rows = _measurement_rows()
    value = RuntimeValue(
        key=ArtifactKey(
            name="Measurements",
            artifact_type=MeasurementsArtifactType,
            scope=RuntimeExecutionAxisScope(axis_id=axis_id),
        ),
        data=MeasurementTable(
            name="Measurements",
            rows=rows,
            subject=MeasurementSubject(MeasurementScope.ARTIFACT, "Measurements"),
        ),
    )
    return StoredRuntimeValue(
        value,
        RuntimeArtifactLocation(
            path=f"results/{axis_id}_Measurements_step1.csv",
            backend="disk",
        ),
    )


def _table_export_expectation() -> RuntimeExportExpectation:
    return RuntimeExportExpectation.from_output_specs(
        (
            ArtifactSpec.output(
                "Measurements",
                MeasurementsArtifactType,
                materialization=MaterializationSpec(CsvOptions(filename_suffix=".csv")),
            ),
        )
    )
