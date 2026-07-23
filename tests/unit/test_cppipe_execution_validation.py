from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openhcs.core.artifacts import (
    ArtifactSpec,
    ImageArtifactType,
    MeasurementsArtifactType,
    SpecialArtifactType,
)
from openhcs.core.component_group_scope import RuntimeExecutionAxisScope
from openhcs.core.runtime_artifact_values import (
    ArtifactKey,
    RuntimeValue,
)
from openhcs.core.runtime_execution_validation import (
    RuntimeArtifactExecutionExpectation,
)
from openhcs.core.runtime_exports import (
    RuntimeExportExpectation,
    RuntimeExportObservation,
)
from openhcs.core.runtime_image_values import ImagePayloadMetadata
from openhcs.core.measurement_row_materialization import (
    MeasurementSparseColumnarRows,
)
from openhcs.core.runtime_measurements import MeasurementTable
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
)
from openhcs.core.runtime_measurements import (
    MeasurementScope,
    MeasurementSubject,
)
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.processing.materialization import (
    CsvOptions,
    FileBundleOptions,
    ImageFileOptions,
    MaterializationSpec,
)
from openhcs.runtime.zmq_execution_observation import (
    ZMQ_RUNTIME_OBSERVATION_EXPORT_SCHEMA_VERSION,
    ZMQRuntimeExecutionObservationExport,
)


def test_cppipe_execution_validation_rejects_header_only_csv(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "A01_Measurements_step1.csv"
    csv_path.write_text("slice_index\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="has no data rows"):
        _validate(
            _measurement_output_specs(),
            _successful_execution_with_measurement_record(
                rows=({"slice_index": 0},),
            ),
            tmp_path,
        )


def test_cppipe_execution_validation_accepts_header_only_empty_csv(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "A01_Measurements_step1.csv"
    csv_path.write_text("slice_index\n", encoding="utf-8")

    observation = _validate(
        _measurement_output_specs(),
        _successful_execution_with_measurement_record(),
        tmp_path,
    )

    assert observation.exports.table_row_counts_by_path[csv_path] == 0


def test_cppipe_execution_validation_accepts_csv_with_data_rows(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "A01_Measurements_step1.csv"
    csv_path.write_text("slice_index\n0\n", encoding="utf-8")

    observation = _validate(
        _measurement_output_specs(),
        _successful_execution_with_measurement_record(rows=({"slice_index": 0},)),
        tmp_path,
    )

    assert observation.exports.table_row_counts_by_path[csv_path] == 1


def test_cppipe_execution_validation_derives_image_export_from_materialization(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "source_plate_image.tif").write_bytes(b"placeholder")

    observation = _validate(
        _image_output_specs(),
        _successful_execution_with_image_record(),
        tmp_path,
    )

    assert observation.exports.image_outputs == (image_dir / "source_plate_image.tif",)


def test_cppipe_execution_validation_rejects_missing_materialized_image_record(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "source_plate_image.tif").write_bytes(b"placeholder")

    with pytest.raises(
        RuntimeError,
        match="produced no runtime record for materialized artifact 'SavedImage'",
    ):
        _validate(
            _image_output_specs(),
            _successful_execution_with_measurement_record(),
            tmp_path,
        )


def test_cppipe_execution_validation_rejects_missing_materialized_image_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        RuntimeError,
        match="image exports were expected but no image outputs exist",
    ):
        _validate(
            _image_output_specs(),
            _successful_execution_with_image_record(),
            tmp_path,
        )


def test_cppipe_execution_validation_uses_file_bundle_record_paths(
    tmp_path: Path,
) -> None:
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    (export_dir / "Measurements.csv").write_text(
        "slice_index\n0\n",
        encoding="utf-8",
    )
    (tmp_path / "analysis.sqlite").write_bytes(b"sqlite")

    observation = _validate(
        _file_bundle_output_specs(),
        _successful_execution_with_file_bundle_record(),
        tmp_path,
    )

    assert export_dir / "Measurements.csv" in observation.exports.output_files
    assert tmp_path / "analysis.sqlite" in observation.exports.output_files


def test_cppipe_execution_validation_rejects_missing_file_bundle_path(
    tmp_path: Path,
) -> None:
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    (export_dir / "Measurements.csv").write_text(
        "slice_index\n0\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="declared missing output path 'analysis.sqlite'",
    ):
        _validate(
            _file_bundle_output_specs(),
            _successful_execution_with_file_bundle_record(),
            tmp_path,
        )


def _measurement_output_specs() -> tuple[ArtifactSpec, ...]:
    return (
        ArtifactSpec.output(
            name="Measurements",
            artifact_type=MeasurementsArtifactType,
            materialization=MaterializationSpec(CsvOptions(filename_suffix=".csv")),
        ),
    )


def _image_output_specs() -> tuple[ArtifactSpec, ...]:
    return (
        ArtifactSpec.output(
            name="SavedImage",
            artifact_type=ImageArtifactType,
            materialization=MaterializationSpec(
                ImageFileOptions(filename_suffix=".tif")
            ),
        ),
    )


def _file_bundle_output_specs() -> tuple[ArtifactSpec, ...]:
    return (
        ArtifactSpec.output(
            name="ExportFiles",
            artifact_type=SpecialArtifactType,
            materialization=MaterializationSpec(FileBundleOptions()),
        ),
    )


def _successful_execution_with_measurement_record(
    *,
    rows: tuple[dict[str, int], ...] = (),
) -> RuntimeValueStore:
    store = RuntimeValueStore()
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="Measurements",
                artifact_type=MeasurementsArtifactType,
                scope=RuntimeExecutionAxisScope(axis_id="A01"),
            ),
            data=MeasurementTable(
                name="Measurements",
                rows=MeasurementSparseColumnarRows.from_rows(
                    rows,
                    fields=(FieldSpec("slice_index"),),
                ),
                subject=MeasurementSubject(MeasurementScope.ARTIFACT, "Measurements"),
            ),
        ),
        path="results/axis_Measurements_step1.csv",
        backend="disk",
    )
    return store


def _successful_execution_with_image_record() -> RuntimeValueStore:
    store = RuntimeValueStore()
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="SavedImage",
                artifact_type=ImageArtifactType,
                scope=RuntimeExecutionAxisScope(axis_id="A01"),
            ),
            data=ImagePayloadMetadata().payload_with(np.zeros((1, 1), dtype=np.uint8)),
        ),
        path="results/axis_SavedImage_step1.pkl",
        backend="memory",
    )
    return store


def _successful_execution_with_file_bundle_record() -> RuntimeValueStore:
    store = RuntimeValueStore()
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="ExportFiles",
                artifact_type=SpecialArtifactType,
                scope=RuntimeExecutionAxisScope(axis_id="A01"),
            ),
            data={
                "exports/Measurements.csv": b"slice_index\n0\n",
                "analysis.sqlite": b"sqlite",
            },
        ),
        path="results/axis_ExportFiles_step1.pkl",
        backend="memory",
    )
    return store


def _validate(
    output_specs: tuple[ArtifactSpec, ...],
    store: RuntimeValueStore,
    output_root: Path,
):
    expectation = RuntimeArtifactExecutionExpectation.from_output_specs(
        output_specs,
        exports=RuntimeExportExpectation.from_output_specs(output_specs),
    )
    return ZMQRuntimeExecutionObservationExport(
        schema_version=ZMQ_RUNTIME_OBSERVATION_EXPORT_SCHEMA_VERSION,
        expectation=expectation,
        records_by_axis={"A01": tuple(store.observed_values)},
        exports=RuntimeExportObservation.from_output_roots((output_root,)),
        output_roots=(output_root,),
        execution_success_by_axis={"A01": True},
    ).require_valid_observation()
