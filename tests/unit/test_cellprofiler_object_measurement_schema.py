import numpy as np

from openhcs.core.measurement_row_materialization import (
    MeasurementProjectedColumnarRows,
)
from openhcs.core.runtime_measurements import MeasurementRowAxisField
from openhcs.core.runtime_tabular_values import FieldSpec
from openhcs.processing.backends.cellprofiler.colocalization import (
    ObjectColocalizationMetricArrays,
)
from openhcs.processing.backends.cellprofiler.intensity import (
    MeasureObjectIntensityModule,
    ObjectIntensityMeasurementRows,
)
from openhcs.processing.backends.cellprofiler.intensity_object_quantiles_numba import (
    ObjectIntensityArrays,
    ObjectIntensityFeatureValues,
)
from openhcs.processing.backends.cellprofiler.shape import (
    MeasureObjectSizeShapeModule,
    ShapeObjectMeasurementRows,
)


def test_measure_object_size_shape_projects_declared_area_shape_category() -> None:
    projected = MeasureObjectSizeShapeModule.project_measurement_record_rows(
        ShapeObjectMeasurementRows.from_rows(
            [
                {
                    "object_label": 1,
                    "Area": 12.0,
                    "Zernike_0_0": 0.25,
                },
            ],
            declared_field_names=("object_label", "Area", "Zernike_0_0"),
        ),
        source_image_name=None,
    )

    assert tuple(projected.columns) == (
        "object_label",
        "AreaShape_Area",
        "AreaShape_Zernike_0_0",
    )
    assert tuple(field.dtype for field in projected.fields) == (int, float, float)


def test_measure_object_size_shape_center_z_is_declared_only_for_3d() -> None:
    center_z = MeasureObjectSizeShapeModule.MeasurementFeature.CENTER_Z.value

    assert center_z not in MeasureObjectSizeShapeModule.measurement_field_names(
        dimensions=2
    )
    assert center_z in MeasureObjectSizeShapeModule.measurement_field_names(
        dimensions=3
    )


def test_shape_colocalization_and_intensity_carriers_preserve_zero_row_schema() -> None:
    shape_rows = ShapeObjectMeasurementRows.from_rows(
        [],
        declared_field_names=("slice_index", "object_label", "Area"),
    )
    colocalization_rows = ObjectColocalizationMetricArrays.empty(0).rows_for(
        np.empty(0, dtype=np.int32),
        slice_index=0,
    )
    intensity_rows = ObjectIntensityMeasurementRows.from_arrays(
        ObjectIntensityArrays.empty(np.empty(0, dtype=np.int32)),
        slice_index=0,
    )

    for rows in (shape_rows, colocalization_rows, intensity_rows):
        assert tuple(field.name for field in rows.fields) == tuple(rows.columns)
        assert rows.row_count() == 0
    assert tuple(field.dtype for field in shape_rows.fields) == (int, int, float)
    assert all(field.dtype is not None for field in colocalization_rows.fields)
    assert all(field.dtype is not None for field in intensity_rows.fields)


def test_object_intensity_dataclass_is_the_only_feature_schema_owner() -> None:
    feature_fields = FieldSpec.from_dataclass_type(ObjectIntensityFeatureValues)
    feature_names = tuple(field.name for field in feature_fields)
    feature_name_set = frozenset(feature_names)
    row_fields = ObjectIntensityMeasurementRows.fields
    row_fields_by_name = {field.name: field for field in row_fields}

    assert "feature_names" not in vars(ObjectIntensityFeatureValues)
    assert "feature_items" not in vars(ObjectIntensityFeatureValues)
    assert tuple(
        field.name for field in row_fields if field.name in feature_name_set
    ) == feature_names
    assert frozenset(row_fields_by_name) == feature_name_set | {
        MeasurementRowAxisField.SLICE_INDEX.value,
        MeasurementRowAxisField.OBJECT_LABEL.value,
    }
    assert all(
        row_fields_by_name[feature_name].dtype is float
        for feature_name in feature_names
    )


def test_measure_object_intensity_projects_location_features_by_nominal_marker() -> (
    None
):
    projected = MeasureObjectIntensityModule.project_measurement_record_rows(
        MeasurementProjectedColumnarRows(
            {
                "slice_index": (0,),
                "object_label": (1,),
                "mean_intensity": (0.5,),
                "center_mass_intensity_x": (3.0,),
                "center_mass_intensity_y": (4.0,),
                "center_mass_intensity_z": (5.0,),
                "max_intensity_x": (4.0,),
                "max_intensity_y": (5.0,),
                "max_intensity_z": (6.0,),
            },
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("object_label", int),
                FieldSpec("mean_intensity", float),
                FieldSpec("center_mass_intensity_x", float),
                FieldSpec("center_mass_intensity_y", float),
                FieldSpec("center_mass_intensity_z", float),
                FieldSpec("max_intensity_x", float),
                FieldSpec("max_intensity_y", float),
                FieldSpec("max_intensity_z", float),
            ),
        ),
        source_image_name="DNA",
    )

    assert tuple(projected.columns) == (
        "slice_index",
        "object_label",
        "Intensity_MeanIntensity_DNA",
        "Location_CenterMassIntensity_X_DNA",
        "Location_CenterMassIntensity_Y_DNA",
        "Location_CenterMassIntensity_Z_DNA",
        "Location_MaxIntensity_X_DNA",
        "Location_MaxIntensity_Y_DNA",
        "Location_MaxIntensity_Z_DNA",
    )
    assert projected.column_values("Location_CenterMassIntensity_X_DNA") == (3.0,)
    assert projected.column_values("Location_CenterMassIntensity_Y_DNA") == (4.0,)
    assert projected.column_values("Location_CenterMassIntensity_Z_DNA") == (5.0,)
    assert projected.column_values("Location_MaxIntensity_X_DNA") == (4.0,)
    assert projected.column_values("Location_MaxIntensity_Y_DNA") == (5.0,)
    assert projected.column_values("Location_MaxIntensity_Z_DNA") == (6.0,)


def test_measure_object_intensity_max_location_relations_are_nominally_scoped() -> None:
    declarations = (
        MeasureObjectIntensityModule.derived_measurement_feature_relation_declarations()
    )

    assert tuple(declaration.source_feature.name for declaration in declarations) == (
        "MAX_INTENSITY_X",
        "MAX_INTENSITY_Y",
        "MAX_INTENSITY_Z",
    )
    assert tuple(
        declaration.relation.target_feature.name for declaration in declarations
    ) == ("MAX_INTENSITY", "MAX_INTENSITY", "MAX_INTENSITY")
