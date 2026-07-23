from dataclasses import fields
from types import SimpleNamespace

import numpy as np

from openhcs.core.measurement_row_materialization import (
    ConcatenatedColumnarRows,
    DataclassMeasurementColumnarRows,
    MeasurementSparseColumnarRows,
)
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
)
from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
    MeasurementScope,
)
from openhcs.core.runtime_tabular_values import ColumnarRows
from openhcs.interop.cellprofiler.module_artifact_declarations import (
    ObjectArtifactOutputModule,
)
from openhcs.interop.cellprofiler.runtime.invocation import CellProfilerMeasurementImage
from openhcs.interop.cellprofiler.runtime.measurement_rows import (
    CellProfilerObjectCoreMeasurementFeature,
)
from openhcs.processing.backends.cellprofiler.colocalization import (
    ColocalizationMeasurements,
    MeasureColocalizationModule,
    ObjectColocalizationMeasurements,
    ObjectColocalizationMetricArrays,
)
from openhcs.processing.backends.cellprofiler.secondary import (
    IdentifySecondaryObjectsModule,
    IdentifyTertiaryObjectsModule,
    SecondaryMethod,
)
from openhcs.processing.backends.cellprofiler.thresholding import (
    CellProfilerThresholdResult,
)


def test_colocalization_slope_is_image_only() -> None:
    assert "measurement_value_fields" not in vars(MeasureColocalizationModule)
    assert "materialized_measurement_fields" not in vars(MeasureColocalizationModule)

    slope_feature = MeasureColocalizationModule.MeasurementFeature.REGRESSION_SLOPE
    assert slope_feature.emitted_in_scope(MeasurementScope.IMAGE)
    assert not slope_feature.emitted_in_scope(MeasurementScope.OBJECT)

    assert "slope" in {field.name for field in fields(ColocalizationMeasurements)}
    assert "slope" not in {
        field.name for field in fields(ObjectColocalizationMeasurements)
    }
    object_rows = ObjectColocalizationMetricArrays.empty(1).rows_for(
        np.asarray((1,), dtype=np.int32)
    )
    assert "slope" not in object_rows.columns

    measurement_image = CellProfilerMeasurementImage(
        source_image_name="DNA__ER",
        source_aliases=("DNA", "ER"),
        payload=np.zeros((2, 2, 2), dtype=np.float32),
    )
    source_pair = measurement_image.source_image_pairs()[0]
    object_rows.metrics.correlation[0] = 0.5
    projected_object_rows = (
        MeasureColocalizationModule.project_source_pair_columnar_rows(
            object_rows,
            source_pair,
        )
    )
    image_values = {field.name: 0.0 for field in fields(ColocalizationMeasurements)}
    image_values.update(slice_index=0, correlation=0.5, slope=0.25)
    image_rows = DataclassMeasurementColumnarRows(
        (ColocalizationMeasurements(**image_values),),
        row_type=ColocalizationMeasurements,
    )
    projected_image_rows = (
        MeasureColocalizationModule.project_source_pair_columnar_rows(
            image_rows,
            source_pair,
        )
    )
    projected_combined_rows = (
        MeasureColocalizationModule.project_source_pair_columnar_rows(
            ConcatenatedColumnarRows((image_rows, object_rows)),
            source_pair,
        )
    )

    assert "Correlation_Slope_DNA_ER" not in projected_object_rows.columns
    assert projected_image_rows.columns["Correlation_Slope_DNA_ER"][0] == 0.25
    assert isinstance(projected_combined_rows, ConcatenatedColumnarRows)
    combined_image_rows, combined_object_rows = projected_combined_rows.row_batches
    assert tuple(field.name for field in combined_image_rows.fields) == tuple(
        field.name for field in projected_image_rows.fields
    )
    assert tuple(field.name for field in combined_object_rows.fields) == tuple(
        field.name for field in projected_object_rows.fields
    )


def test_identify_tertiary_objects_emits_xy_locations_only(monkeypatch) -> None:
    feature_field = MeasurementRowAxisField.FEATURE_NAME.value
    sparse_field = "sparse_value"

    source_rows = MeasurementSparseColumnarRows.from_rows(
        (
            {
                feature_field: "Count_Cytoplasm",
                "result_value": 1,
                sparse_field: 3.0,
            },
            *(
                {feature_field: feature.value, "result_value": 2.0}
                for feature in CellProfilerObjectCoreMeasurementFeature
            ),
        ),
        fields=(
            FieldSpec(feature_field, str),
            FieldSpec("result_value", float),
            FieldSpec(sparse_field, float, required=False),
        ),
    )

    def inherited_rows(cls, request):
        del cls, request
        return source_rows

    def reject_row_reconstruction(self):
        del self
        raise AssertionError("tertiary projection must consume declared columns")

    monkeypatch.setattr(
        ObjectArtifactOutputModule,
        "measurement_record_rows",
        classmethod(inherited_rows),
    )
    monkeypatch.setattr(
        MeasurementSparseColumnarRows,
        "iter_row_mappings",
        reject_row_reconstruction,
    )

    rows = IdentifyTertiaryObjectsModule.measurement_record_rows(object())
    row_mappings = rows.row_mappings()

    assert [row[feature_field] for row in row_mappings] == [
        "Count_Cytoplasm",
        CellProfilerObjectCoreMeasurementFeature.OBJECT_NUMBER.value,
        CellProfilerObjectCoreMeasurementFeature.CENTER_X.value,
        CellProfilerObjectCoreMeasurementFeature.CENTER_Y.value,
    ]
    assert row_mappings[0][sparse_field] == 3.0
    assert all(sparse_field not in row for row in row_mappings[1:])


def test_identify_secondary_distance_n_omits_threshold_features() -> None:
    def rows_for(method: SecondaryMethod) -> ColumnarRows:
        threshold = CellProfilerThresholdResult(
            final_threshold=0.5,
            original_threshold=0.5,
            mask=np.ones((2, 2), dtype=bool),
        )
        request = SimpleNamespace(
            output_value=(
                threshold.measurement_rows()
                if method.requires_threshold
                else MeasurementSparseColumnarRows.from_rows((), fields=())
            ),
            single_output_object_name=lambda: "Cells",
        )
        projection = IdentifySecondaryObjectsModule.MeasurementRows.for_request(
            IdentifySecondaryObjectsModule,
            request,
        )
        return projection.rows()

    assert IdentifySecondaryObjectsModule.measurement_row_projection_types() == (
        IdentifySecondaryObjectsModule.MeasurementRows,
    )
    assert rows_for(SecondaryMethod.DISTANCE_N).fields == ()
    assert {
        field.name
        for field in rows_for(SecondaryMethod.PROPAGATION).fields
        if field.name != MeasurementRowAxisField.SLICE_INDEX.value
    } == {
        "Threshold_FinalThreshold_Cells",
        "Threshold_OrigThreshold_Cells",
        "Threshold_WeightedVariance_Cells",
        "Threshold_SumOfEntropies_Cells",
    }
