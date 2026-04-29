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


def test_runtime_export_validation_rejects_header_only_table(
    tmp_path: Path,
) -> None:
    table_path = tmp_path / "axis_Measurements_step1.csv"
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

    assert failures == (f"table output {table_path} has no data rows",)


def test_runtime_export_validation_checks_table_schema_fields(
    tmp_path: Path,
) -> None:
    table_path = tmp_path / "axis_Measurements_step1.csv"
    table_path.write_text("wrong_field\n0\n", encoding="utf-8")
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

    assert failures == (
        f"table output {table_path} for artifact 'Measurements' is "
        "missing schema fields ('slice_index',)",
    )


def _stored_measurements_record() -> StoredRuntimeValue:
    value = RuntimeValue(
        key=ArtifactKey(
            name="Measurements",
            kind=ArtifactKind.MEASUREMENTS,
            scope=ArtifactScope(axis_id="A01"),
        ),
        data=(),
        schema=RuntimeValueSchema(
            kind=ArtifactKind.MEASUREMENTS,
            fields=(FieldSpec("slice_index"),),
        ),
    )
    return StoredRuntimeValue(
        value,
        RuntimeArtifactLocation(
            path="results/axis_Measurements_step1.csv",
            backend="disk",
        ),
    )
