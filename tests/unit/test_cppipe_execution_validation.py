from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from openhcs.interop.cellprofiler.execution_validation import (
    CPPipeExecutionValidationError,
    validate_cppipe_execution,
)
from openhcs.interop.cellprofiler.parser import ModuleBlock
from openhcs.interop.cellprofiler.runtime_pipeline import DirectPipelineExecution
from openhcs.core.artifacts import (
    ArtifactKey,
    ArtifactScope,
    ArtifactSpec,
    ImageArtifactType,
    MeasurementsArtifactType,
)
from openhcs.core.runtime_exports import RuntimeImageExportBitDepth
from openhcs.core.runtime_semantics import FieldSpec
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.runtime_values import RuntimeValue, RuntimeValueSchema


def test_cppipe_execution_validation_rejects_header_only_csv(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "A01_Measurements_step1.csv"
    csv_path.write_text("slice_index\n", encoding="utf-8")

    with pytest.raises(CPPipeExecutionValidationError, match="has no data rows"):
        validate_cppipe_execution(
            _prepared_exporting_measurements(),
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

    validation = validate_cppipe_execution(
        _prepared_exporting_measurements(),
        _successful_execution_with_measurement_record(),
        tmp_path,
    )

    assert validation.observation.exports.table_row_counts_by_path[csv_path] == 0


def test_cppipe_execution_validation_accepts_csv_with_data_rows(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "A01_Measurements_step1.csv"
    csv_path.write_text("slice_index\n0\n", encoding="utf-8")

    validation = validate_cppipe_execution(
        _prepared_exporting_measurements(),
        _successful_execution_with_measurement_record(rows=({"slice_index": 0},)),
        tmp_path,
    )

    assert validation.observation.exports.table_row_counts_by_path[csv_path] == 1


def test_cppipe_execution_validation_tracks_saveimages_artifact_exports(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "source_plate_image.tif").write_bytes(b"placeholder")

    validation = validate_cppipe_execution(
        _prepared_exporting_images(),
        _successful_execution_with_image_record(),
        tmp_path,
    )

    assert validation.expectation.exports.image_artifact_names == frozenset(
        ("RGBImage",)
    )
    assert tuple(
        spec.bit_depth for spec in validation.expectation.exports.image_export_specs
    ) == (RuntimeImageExportBitDepth.UINT8,)


def test_cppipe_execution_validation_rejects_missing_saveimages_artifact(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "source_plate_image.tif").write_bytes(b"placeholder")

    with pytest.raises(
        CPPipeExecutionValidationError,
        match="produced no runtime image artifact 'RGBImage'",
    ):
        validate_cppipe_execution(
            _prepared_exporting_images(),
            _successful_execution_with_measurement_record(),
            tmp_path,
        )


def _prepared_exporting_measurements() -> SimpleNamespace:
    return SimpleNamespace(
        infrastructure_modules=(
            ModuleBlock(name="ExportToSpreadsheet", module_num=1),
        ),
        generated_pipeline=SimpleNamespace(
            artifact_contracts=(
                SimpleNamespace(
                    outputs=(
                        ArtifactSpec.output(
                            name="Measurements",
                            artifact_type=MeasurementsArtifactType,
                        ),
                    )
                ),
            )
        ),
    )


def _prepared_exporting_images() -> SimpleNamespace:
    return SimpleNamespace(
        infrastructure_modules=(
            ModuleBlock(
                name="SaveImages",
                module_num=1,
                settings={
                    "Select the image to save": "RGBImage",
                    "Image bit depth": "8-bit integer",
                    "Saved file format": "tiff",
                },
            ),
        ),
        generated_pipeline=SimpleNamespace(
            artifact_contracts=(
                SimpleNamespace(
                    outputs=(
                        ArtifactSpec.output(
                            name="RGBImage",
                            artifact_type=ImageArtifactType,
                        ),
                    )
                ),
            )
        ),
    )


def _successful_execution_with_measurement_record(
    *,
    rows: tuple[dict[str, int], ...] = (),
) -> DirectPipelineExecution:
    store = RuntimeValueStore()
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="Measurements",
                artifact_type=MeasurementsArtifactType,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=rows,
            schema=RuntimeValueSchema(
                artifact_type=MeasurementsArtifactType,
                fields=(FieldSpec("slice_index"),),
            ),
        ),
        path="results/axis_Measurements_step1.csv",
        backend="disk",
    )
    return DirectPipelineExecution(
        compiled_contexts={
            "A01": SimpleNamespace(runtime_value_store=store, step_plans={}),
        },
        execution_results={
            "A01": SimpleNamespace(is_success=lambda: True),
        },
    )


def _successful_execution_with_image_record() -> DirectPipelineExecution:
    store = RuntimeValueStore()
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="RGBImage",
                artifact_type=ImageArtifactType,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=object(),
            schema=RuntimeValueSchema(artifact_type=ImageArtifactType),
        ),
        path="results/axis_RGBImage_step1.pkl",
        backend="memory",
    )
    return DirectPipelineExecution(
        compiled_contexts={
            "A01": SimpleNamespace(runtime_value_store=store, step_plans={}),
        },
        execution_results={
            "A01": SimpleNamespace(is_success=lambda: True),
        },
    )
