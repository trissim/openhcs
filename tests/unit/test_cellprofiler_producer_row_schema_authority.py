"""Focused coverage for producer-owned CellProfiler result row schemas."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from openhcs.core.artifacts import (
    ArtifactSpec,
    ImageArtifactType,
    MeasurementsArtifactType,
)
from openhcs.core.callable_contract import CallableContract, CallableMetadata
from openhcs.core.measurement_row_materialization import (
    DataclassMeasurementColumnarRows,
)
from openhcs.core.runtime_tabular_values import FieldSpec
from openhcs.interop.cellprofiler.runtime import measurement_rows
from openhcs.processing.backends.cellprofiler.alignment import (
    AlignModule,
    AlignShiftMeasurement,
)
from openhcs.processing.backends.cellprofiler.classification import (
    ClassificationResult,
    ClassifyObjectsSingleMeasurementModule,
)


PROJECT_ROOT = Path(__file__).parents[2]


def test_align_projection_uses_producer_annotations_and_field_dtypes() -> None:
    source_rows = DataclassMeasurementColumnarRows(
        (AlignShiftMeasurement(4, 1, -3, 2),),
        row_type=AlignShiftMeasurement,
    )

    projected = AlignModule.MeasurementRows(
        source_rows,
        module_type=AlignModule,
        image_output_names=("DNA", "RNA"),
    ).rows()

    assert projected.row_mappings() == (
        {
            "slice_index": 4,
            "source_image_name": "RNA",
            "Align_Xshift_RNA": -3,
            "Align_Yshift_RNA": 2,
        },
    )
    assert projected.fields == (
        FieldSpec("slice_index", int),
        FieldSpec("source_image_name", str),
        FieldSpec("Align_Xshift_DNA", int, required=False),
        FieldSpec("Align_Yshift_DNA", int, required=False),
        FieldSpec("Align_Xshift_RNA", int, required=False),
        FieldSpec("Align_Yshift_RNA", int, required=False),
    )


def test_align_projection_reads_ordered_image_outputs_from_callable_contract() -> None:
    source_rows = DataclassMeasurementColumnarRows(
        (
            AlignShiftMeasurement(0, 0, 1, 2),
            AlignShiftMeasurement(0, 1, 3, 4),
        ),
        row_type=AlignShiftMeasurement,
    )
    callable_contract = CallableContract(
        func=lambda: None,
        function_name="align",
        module_name="Align",
        metadata=CallableMetadata(
            artifact_outputs=(
                ArtifactSpec.output("DNA", ImageArtifactType),
                ArtifactSpec.output("AlignMeasurements", MeasurementsArtifactType),
                ArtifactSpec.output("RNA", ImageArtifactType),
            )
        ),
    )

    projected = AlignModule.MeasurementRows.for_request(
        AlignModule,
        SimpleNamespace(
            callable_contract=callable_contract,
            output_value=source_rows,
        ),
    )

    assert projected.image_output_names == ("DNA", "RNA")
    assert tuple(
        row["source_image_name"] for row in projected.rows().row_mappings()
    ) == ("DNA", "RNA")


def test_classification_projection_uses_producer_feature_templates() -> None:
    source_rows = ClassificationResult.columnar(
        ClassificationResult(
            total_objects=3,
            bin_counts='{"Small": 1, "Large": 1}',
            bin_percentages='{"Small": 33.0, "Large": 33.0}',
            object_classes='{"1": "Small", "3": "Large"}',
            slice_index=2,
        )
    )

    projected = ClassifyObjectsSingleMeasurementModule.MeasurementRows(
        source_rows,
        module_type=ClassifyObjectsSingleMeasurementModule,
        object_name="Cells",
    ).rows()
    fields = {field.name: field for field in projected.fields}

    assert fields["Classify_Small_NumObjectsPerBin"].dtype is int
    assert fields["Classify_Small_PctObjectsPerBin"].dtype is float
    assert fields["Classify_Small"].dtype is int
    assert projected.row_mappings()[-2:] == (
        {
            "slice_index": 2,
            "object_name": "Cells",
            "object_label": 2,
            "Classify_Small": 0,
            "Classify_Large": 0,
        },
        {
            "slice_index": 2,
            "object_name": "Cells",
            "object_label": 3,
            "Classify_Small": 0,
            "Classify_Large": 1,
        },
    )


def test_result_stat_schema_mirrors_are_deleted() -> None:
    assert not hasattr(measurement_rows, "CellProfilerMeasurementStatField")
    assert not hasattr(
        measurement_rows.ModuleOwnedResultMeasurementRows,
        "stat_field_type",
    )
    assert not hasattr(AlignModule, "MeasurementStatField")
    assert not hasattr(
        ClassifyObjectsSingleMeasurementModule,
        "MeasurementStatField",
    )

    owned_sources = (
        PROJECT_ROOT / "openhcs/interop/cellprofiler/runtime/measurement_rows.py",
        PROJECT_ROOT / "openhcs/processing/backends/cellprofiler/alignment.py",
        PROJECT_ROOT / "openhcs/processing/backends/cellprofiler/classification.py",
    )
    source = "\n".join(path.read_text(encoding="utf-8") for path in owned_sources)
    assert "CellProfilerMeasurementStatField" not in source
    assert "MeasurementStatField" not in source
    assert "stat_field_type" not in source
