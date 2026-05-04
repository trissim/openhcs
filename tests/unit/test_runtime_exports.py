from __future__ import annotations

from pathlib import Path

from openhcs.core.artifacts import (
    ArtifactKey,
    ArtifactKind,
    ArtifactScope,
)
from openhcs.core.runtime_exports import (
    RuntimeExportExpectation,
    RuntimeExportObservation,
    runtime_export_failures,
)
from openhcs.core.runtime_semantics import FieldSpec
from openhcs.core.runtime_stores import RuntimeArtifactLocation, StoredRuntimeValue
from openhcs.core.runtime_values import RuntimeValue, RuntimeValueSchema


def test_runtime_export_validation_accepts_header_only_empty_table(
    tmp_path: Path,
) -> None:
    table_path = tmp_path / "A01_Measurements_step1.csv"
    table_path.write_text("slice_index\n", encoding="utf-8")
    record = _stored_measurements_record()

    failures = runtime_export_failures(
        RuntimeExportExpectation.from_flags(
            table_exports=True,
            image_exports=False,
            table_artifact_kinds=frozenset((ArtifactKind.MEASUREMENTS,)),
        ),
        RuntimeExportObservation.from_output_root(tmp_path),
        {"A01": (record,)},
    )

    assert failures == ()


def test_runtime_export_validation_accepts_header_only_empty_column_table(
    tmp_path: Path,
) -> None:
    table_path = tmp_path / "A01_Measurements_step1.csv"
    table_path.write_text("slice_index\n", encoding="utf-8")
    record = _stored_measurements_record(rows={"slice_index": []})

    failures = runtime_export_failures(
        RuntimeExportExpectation.from_flags(
            table_exports=True,
            image_exports=False,
            table_artifact_kinds=frozenset((ArtifactKind.MEASUREMENTS,)),
        ),
        RuntimeExportObservation.from_output_root(tmp_path),
        {"A01": (record,)},
    )

    assert failures == ()


def test_runtime_export_validation_rejects_header_only_nonempty_table(
    tmp_path: Path,
) -> None:
    table_path = tmp_path / "A01_Measurements_step1.csv"
    table_path.write_text("slice_index\n", encoding="utf-8")
    record = _stored_measurements_record(rows=({"slice_index": 0},))

    failures = runtime_export_failures(
        RuntimeExportExpectation.from_flags(
            table_exports=True,
            image_exports=False,
            table_artifact_kinds=frozenset((ArtifactKind.MEASUREMENTS,)),
        ),
        RuntimeExportObservation.from_output_root(tmp_path),
        {"A01": (record,)},
    )

    assert failures == (f"table output {table_path} has no data rows",)


def test_runtime_export_validation_checks_table_schema_fields(
    tmp_path: Path,
) -> None:
    table_path = tmp_path / "A01_Measurements_step1.csv"
    table_path.write_text("wrong_field\n0\n", encoding="utf-8")
    record = _stored_measurements_record(rows=({"slice_index": 0},))

    failures = runtime_export_failures(
        RuntimeExportExpectation.from_flags(
            table_exports=True,
            image_exports=False,
            table_artifact_kinds=frozenset((ArtifactKind.MEASUREMENTS,)),
        ),
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
        rows=({"slice_index": 0},),
    )

    failures = runtime_export_failures(
        RuntimeExportExpectation.from_flags(
            table_exports=True,
            image_exports=False,
            table_artifact_kinds=frozenset((ArtifactKind.MEASUREMENTS,)),
        ),
        RuntimeExportObservation.from_output_root(tmp_path),
        {"A01": (a01_record,), "A02": (a02_record,)},
    )

    assert failures == ()


def test_runtime_export_validation_accepts_format_compatible_schema_fields(
    tmp_path: Path,
) -> None:
    table_path = tmp_path / "A01_Measurements_step1.csv"
    table_path.write_text("SliceIndex\n0\n", encoding="utf-8")
    record = _stored_measurements_record(rows=({"slice_index": 0},))

    failures = runtime_export_failures(
        RuntimeExportExpectation.from_flags(
            table_exports=True,
            image_exports=False,
            table_artifact_kinds=frozenset((ArtifactKind.MEASUREMENTS,)),
        ),
        RuntimeExportObservation.from_output_root(tmp_path),
        {"A01": (record,)},
    )

    assert failures == ()


def _stored_measurements_record(
    *,
    axis_id: str = "A01",
    rows: object = (),
) -> StoredRuntimeValue:
    value = RuntimeValue(
        key=ArtifactKey(
            name="Measurements",
            kind=ArtifactKind.MEASUREMENTS,
            scope=ArtifactScope(axis_id=axis_id),
        ),
        data=rows,
        schema=RuntimeValueSchema(
            kind=ArtifactKind.MEASUREMENTS,
            fields=(FieldSpec("slice_index"),),
        ),
    )
    return StoredRuntimeValue(
        value,
        RuntimeArtifactLocation(
            path=f"results/{axis_id}_Measurements_step1.csv",
            backend="disk",
        ),
    )
