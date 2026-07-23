import ast
import inspect
import textwrap

import numpy as np
import pytest

from openhcs.core.measurement_row_materialization import (
    ConcatenatedColumnarRows,
    DataclassMeasurementColumnarRows,
    MeasurementSparseColumnarRows,
    WideMeasurementRowAccumulator,
    coalesced_sparse_measurement_row_mappings,
)
from openhcs.core.runtime_image_values import ImagePayloadMetadata
from openhcs.core.runtime_measurements import MeasurementTable
from openhcs.core.runtime_measurements import (
    DEFAULT_RUNTIME_MEASUREMENT_ROW_IDENTITY_CONTRACT,
    MeasurementRowAxisField,
    MeasurementScope,
    MeasurementSubject,
)
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
    MeasurementObjectRowIdentity,
)
from openhcs.core.runtime_tabular_values import (
    ColumnarRows,
)
from openhcs.core.runtime_object_label_domains import ObjectLabelDomain
from openhcs.core.runtime_object_labels import (
    ObjectLabelPayload,
    ObjectLabelVariantData,
)
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxis,
    RuntimePlaneAxisValueProjection,
)
from openhcs.core.source_image_provenance import SourceImageProvenancePlanes


from openhcs.interop.cellprofiler.runtime.object_measurement_row_completion import (
    MissingObjectMeasurementValuePolicy,
    ObjectMeasurementRowCompletionSchema,
    ObjectMeasurementProjectedRowKeys,
    ObjectMeasurementRowIdentityProjectionResult,
    ObjectMeasurementRowOrdinalProjectionState,
    RowSequenceMeasurementObjectRowIdentityProjectionStrategy,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_row_policies import (
    CellProfilerObjectMeasurementRowPolicy,
)
from openhcs.interop.cellprofiler.runtime.measurement_rows import (
    measurement_source_image_name_for_slice,
)
from openhcs.processing.backends.cellprofiler.intensity import (
    ImageIntensityMeasurement,
    ImageIntensityPercentileSpec,
    MeasureImageIntensityModule,
)
from openhcs.processing.backends.cellprofiler.crop import CropModule


class _ColumnOnlyMeasurementRows(ColumnarRows):
    def __init__(
        self,
        columns: dict[str, tuple[object, ...]],
        fields: tuple[FieldSpec, ...],
        *,
        declared_object_measurement_domain_covered: bool = False,
        object_row_identity: MeasurementObjectRowIdentity | None = None,
    ) -> None:
        self._columns = columns
        self._fields = fields
        self._declared_object_measurement_domain_covered = (
            declared_object_measurement_domain_covered
        )
        self.object_row_identity = object_row_identity
        self.validate_fields()

    @property
    def columns(self) -> dict[str, tuple[object, ...]]:
        return self._columns

    @property
    def fields(self) -> tuple[FieldSpec, ...]:
        return self._fields

    @property
    def covers_declared_object_measurement_domain(self) -> bool:
        return self._declared_object_measurement_domain_covered

    def iter_row_mappings(self):
        raise AssertionError("columnar pivot must not materialize input row mappings")


class _CountingColumnOnlyMeasurementRows(_ColumnOnlyMeasurementRows):
    def __init__(
        self,
        columns: dict[str, tuple[object, ...]],
        fields: tuple[FieldSpec, ...],
        *,
        object_row_identity: MeasurementObjectRowIdentity | None = None,
    ) -> None:
        super().__init__(
            columns,
            fields,
            object_row_identity=object_row_identity,
        )
        self.column_value_calls: dict[str, int] = {}

    def column_values(self, column: str):
        self.column_value_calls[column] = self.column_value_calls.get(column, 0) + 1
        return super().column_values(column)


class _ZeroWithinPositiveExtentRowPolicy(CellProfilerObjectMeasurementRowPolicy):
    missing_value_policy = (
        MissingObjectMeasurementValuePolicy.ZERO_WITHIN_POSITIVE_EXTENT
    )


def test_measurement_source_name_projects_exact_runtime_metadata_plane() -> None:
    source_names = ("OrigBlue", "OrigGreen", "OrigRed")
    metadata = ImagePayloadMetadata(
        source_image_names=source_names,
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=tuple(f"/plate/A01_{name}.tif" for name in source_names),
        ),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
    )
    projection = RuntimePlaneAxisValueProjection.preserve(
        axis=RuntimePlaneAxis.SOURCE_BINDING,
        axis_size=len(source_names),
        source_aliases=source_names,
    )

    assert (
        tuple(
            measurement_source_image_name_for_slice(metadata, projection, slice_index)
            for slice_index in range(len(source_names))
        )
        == source_names
    )


def test_measurement_source_name_does_not_fall_back_to_projection_alias_order() -> None:
    source_aliases = ("OrigBlue", "OrigGreen", "OrigRed")
    metadata = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=tuple(f"/plate/A01_{name}.tif" for name in source_aliases),
        ),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
    )
    projection = RuntimePlaneAxisValueProjection.preserve(
        axis=RuntimePlaneAxis.SOURCE_BINDING,
        axis_size=len(source_aliases),
        source_aliases=source_aliases,
    )

    with pytest.raises(
        ValueError,
        match="requires exactly one source image name",
    ):
        measurement_source_image_name_for_slice(metadata, projection, 1)


def test_image_intensity_rows_own_exact_source_qualified_features() -> None:
    measurement = ImageIntensityMeasurement.from_pixels(
        np.asarray((1.0, 2.0, 3.0)),
        percentile_spec=ImageIntensityPercentileSpec(
            enabled=True,
            raw_percentiles="10,90",
        ),
    )

    rows = MeasureImageIntensityModule.MeasurementRows(
        DataclassMeasurementColumnarRows((measurement,)),
        module_type=MeasureImageIntensityModule,
        measurement_name="CropBlue",
    ).rows()
    rows_by_feature = {
        str(row[MeasurementRowAxisField.FEATURE_NAME.value]): row for row in rows
    }

    assert "Intensity_TotalIntensity_CropBlue" in rows_by_feature
    assert "Intensity_LowerQuartileIntensity_CropBlue" in rows_by_feature
    assert rows_by_feature["Intensity_Percentile_10_CropBlue"]["result_value"] == 1.2
    assert rows_by_feature["Intensity_Percentile_90_CropBlue"]["result_value"] == 2.8
    assert all("PercentileValues" not in feature for feature in rows_by_feature)


def test_crop_rows_own_exact_source_qualified_features() -> None:
    projected_rows = CropModule.prepare_measurement_record_rows(
        MeasurementSparseColumnarRows.from_rows(
            (
                {
                    "slice_index": 0,
                    "area_retained": 82,
                    "original_area": 100,
                    "fraction_retained": 0.82,
                },
            ),
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("area_retained", int),
                FieldSpec("original_area", int),
                FieldSpec("fraction_retained", float),
            ),
        ),
        source_image_name="CropBlue",
    )

    assert dict(projected_rows.columns) == {
        "slice_index": (0,),
        "Crop_AreaRetainedAfterCropping_CropBlue": (82,),
        "Crop_OriginalImageArea_CropBlue": (100,),
    }


def test_measurement_table_declares_non_object_rows_as_image_subject() -> None:
    rows = MeasurementSparseColumnarRows.from_rows(
        ({"slice_index": 0, "Count_Cells": 2},),
        fields=(FieldSpec("slice_index", int), FieldSpec("Count_Cells", int)),
    )
    generic_image_table = MeasurementTable(
        name="count_measurements",
        rows=rows,
        subject=MeasurementSubject(
            MeasurementScope.IMAGE,
            MeasurementScope.IMAGE.value,
        ),
    )
    source_image_table = CropModule.build_measurement_table(
        name="crop_measurements",
        rows=rows,
        object_name=None,
        source_image_name="DNA",
        source_metadata=ImagePayloadMetadata(
            source_path="/images/dna.tif",
            source_image_names=("DNA",),
        ),
    )

    assert generic_image_table.subject == MeasurementSubject(
        MeasurementScope.IMAGE,
        MeasurementScope.IMAGE.value,
    )
    assert source_image_table.subject == MeasurementSubject(
        MeasurementScope.IMAGE,
        "DNA",
    )
    assert source_image_table.source_image_name == "DNA"
    assert source_image_table.source_path == "/images/dna.tif"
    assert source_image_table.source_image_names == ("DNA",)


def test_row_ordinal_state_requires_declared_axis_and_label_mapping() -> None:
    state = ObjectMeasurementRowOrdinalProjectionState(
        ordinal_by_axis_label={(): {10: 1, 20: 2}}
    )

    assert (
        state.ordinal_for_declared_object(
            {"object_label": 20},
            axis_key=(),
            object_id_field=MeasurementRowAxisField.OBJECT_LABEL.value,
        )
        == 2
    )
    with pytest.raises(ValueError, match="absent from the declared row-ordinal domain"):
        state.ordinal_for_declared_object(
            {"object_label": 30},
            axis_key=(),
            object_id_field=MeasurementRowAxisField.OBJECT_LABEL.value,
        )


def test_object_measurement_policy_splits_exact_sparse_columnar_rows() -> None:
    fields = (
        FieldSpec("slice_index", int),
        FieldSpec("object_label", int),
        FieldSpec("result_value", float),
    )
    rows = MeasurementSparseColumnarRows.from_rows(
        (
            {"slice_index": 0, "result_value": 0.25},
            {"slice_index": 0, "object_label": 7, "result_value": 0.75},
        ),
        fields=fields,
        object_row_identity=MeasurementObjectRowIdentity.LABEL_ID,
    )

    object_rows, image_rows = (
        CellProfilerObjectMeasurementRowPolicy().split_scoped_rows(rows)
    )

    assert object_rows.fields == fields
    assert image_rows.fields == fields
    assert object_rows.object_row_identity is MeasurementObjectRowIdentity.LABEL_ID
    assert image_rows.object_row_identity is None
    assert object_rows.row_mappings() == (
        {"slice_index": 0, "object_label": 7, "result_value": 0.75},
    )
    assert image_rows.row_mappings() == ({"slice_index": 0, "result_value": 0.25},)


def test_object_measurement_policy_skips_split_for_complete_identity_carrier() -> None:
    fields = (FieldSpec("object_label", int), FieldSpec("result_value", float))
    rows = _ColumnOnlyMeasurementRows(
        {"object_label": (1, 3), "result_value": (0.25, 0.75)},
        fields,
        declared_object_measurement_domain_covered=True,
        object_row_identity=MeasurementObjectRowIdentity.LABEL_ID,
    )

    object_rows, image_rows = (
        CellProfilerObjectMeasurementRowPolicy().split_scoped_rows(rows)
    )

    assert object_rows is rows
    assert image_rows.fields == fields
    assert image_rows.row_count() == 0
    assert image_rows.object_row_identity is None


def test_completion_projection_selects_and_orders_sparse_columns_by_row_keys() -> None:
    fields = (
        FieldSpec("object_label", int),
        FieldSpec("direction", int),
        FieldSpec("value", float),
    )
    rows = MeasurementSparseColumnarRows.from_rows(
        (
            {"object_label": 3, "direction": 1, "value": 31.0},
            {"object_label": 1, "direction": 0, "value": 10.0},
            {"object_label": 5, "direction": 0, "value": 50.0},
            {"object_label": 3, "direction": 0},
            {"object_label": 1, "direction": 1, "value": 11.0},
        ),
        fields=fields,
        object_row_identity=MeasurementObjectRowIdentity.LABEL_ID,
    )
    row_keys = ObjectMeasurementProjectedRowKeys(
        ((3, (1,)), (1, (0,)), (5, (0,)), (3, (0,)), (1, (1,)))
    )
    projection = ObjectMeasurementRowIdentityProjectionResult(
        rows=rows,
        row_keys=row_keys,
        measured_row_keys=row_keys,
        axis_keys=((1,), (0,)),
    )

    bounded = projection.within_axis_domain(
        axis_keys=((0,), (1,)),
        object_ids_by_axis={(0,): (1, 3), (1,): (1, 3)},
    )
    ordered = bounded.ordered_rows(
        object_ids=(1, 3),
        axis_keys=((0,), (1,)),
    )

    assert bounded.row_keys.entries == (
        (3, (1,)),
        (1, (0,)),
        (3, (0,)),
        (1, (1,)),
    )
    assert ordered.row_mappings() == (
        {"object_label": 1, "direction": 0, "value": 10.0},
        {"object_label": 3, "direction": 0},
        {"object_label": 1, "direction": 1, "value": 11.0},
        {"object_label": 3, "direction": 1, "value": 31.0},
    )


def test_row_sequence_projection_overlays_identity_without_materializing_rows() -> None:
    fields = (
        FieldSpec("object_label", int),
        FieldSpec("direction", int),
        FieldSpec("value", float),
    )
    rows = _ColumnOnlyMeasurementRows(
        {
            "object_label": (10, 30),
            "direction": (0, 0),
            "value": (1.0, 3.0),
        },
        fields,
        object_row_identity=MeasurementObjectRowIdentity.ROW_SEQUENCE,
    )
    schema = ObjectMeasurementRowCompletionSchema.from_fields(fields)

    projected = RowSequenceMeasurementObjectRowIdentityProjectionStrategy().project_rows(
        rows,
        schema,
        CellProfilerObjectMeasurementRowPolicy(),
    )

    assert projected.rows.fields == fields
    assert projected.rows.object_row_identity is MeasurementObjectRowIdentity.ROW_SEQUENCE
    assert tuple(projected.rows.column_values("object_label")) == (1, 2)
    assert tuple(projected.rows.column_values("value")) == (1.0, 3.0)
    assert projected.row_keys.entries == ((1, (0,)), (2, (0,)))
    assert projected.measured_row_keys.entries == projected.row_keys.entries


def test_row_sequence_projection_does_not_lookup_result_columns_per_row() -> None:
    fields = (
        FieldSpec("object_label", int),
        FieldSpec("direction", int),
        FieldSpec("first_value"),
        FieldSpec("second_value"),
    )
    rows = _CountingColumnOnlyMeasurementRows(
        {
            "object_label": (10, 20, 30),
            "direction": (0, 0, 0),
            "first_value": (np.nan, 0.0, None),
            "second_value": (None, np.nan, "observed"),
        },
        fields,
        object_row_identity=MeasurementObjectRowIdentity.ROW_SEQUENCE,
    )
    schema = ObjectMeasurementRowCompletionSchema.from_fields(fields)

    projected = (
        RowSequenceMeasurementObjectRowIdentityProjectionStrategy().project_rows(
            rows,
            schema,
            CellProfilerObjectMeasurementRowPolicy(),
        )
    )

    assert projected.row_keys.entries == ((1, (0,)), (2, (0,)), (3, (0,)))
    assert projected.measured_row_keys.entries == ((2, (0,)), (3, (0,)))
    # One lookup builds the reducer view; one belongs to the projected carrier.
    assert rows.column_value_calls["first_value"] == 2
    assert rows.column_value_calls["second_value"] == 2


def test_missing_value_batch_preserves_zero_extent_scalar_semantics() -> None:
    label_payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.asarray([[1, 0, 7]], dtype=np.int32)
        ),
        domain=ObjectLabelDomain(declared_object_ids=(1, 3, 7)),
    )
    policy = _ZeroWithinPositiveExtentRowPolicy()

    values = policy.missing_measurement_values(
        object_ids=(2, 8),
        label_payload=label_payload,
        field_name="value",
        positive_label_extents=(3, None),
    )

    assert values[0] == 0.0
    assert np.isnan(values[1])


def test_object_completion_preserves_noncontiguous_domain_and_missing_rows() -> None:
    rows = MeasurementSparseColumnarRows.from_rows(
        (
            {"object_label": 7, "value": 70.0},
            {"object_label": 1, "value": 10.0},
        ),
        fields=(FieldSpec("object_label", int), FieldSpec("value", float)),
        object_row_identity=MeasurementObjectRowIdentity.LABEL_ID,
    )
    label_payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.asarray([[1, 0, 7]], dtype=np.int32)
        ),
        domain=ObjectLabelDomain(declared_object_ids=(1, 3, 7)),
    )

    completed = CellProfilerObjectMeasurementRowPolicy().complete_rows(
        rows,
        label_payload=label_payload,
    )
    completed_rows = completed.row_mappings()

    assert completed.covers_declared_object_measurement_domain
    assert completed.object_row_identity is MeasurementObjectRowIdentity.LABEL_ID
    assert tuple(row["object_label"] for row in completed_rows) == (1, 3, 7)
    assert completed_rows[0]["value"] == 10.0
    assert np.isnan(completed_rows[1]["value"])
    assert completed_rows[2]["value"] == 70.0


@pytest.mark.parametrize(
    "method",
    (
        ObjectMeasurementRowIdentityProjectionResult.within_axis_domain,
        ObjectMeasurementRowIdentityProjectionResult.ordered_rows,
        CellProfilerObjectMeasurementRowPolicy.complete_object_domain_rows,
    ),
)
def test_completion_hotpaths_do_not_reconstruct_row_mappings(method) -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert called_attributes.isdisjoint({"iter_row_mappings", "row_mappings"})
    assert "MeasurementSparseColumnarRows.from_rows" not in inspect.getsource(method)


def test_second_completion_hotpaths_remain_structural() -> None:
    for method in (
        RowSequenceMeasurementObjectRowIdentityProjectionStrategy.project_rows,
        ObjectMeasurementRowCompletionSchema.missing_columnar_rows,
    ):
        source = inspect.getsource(method)
        tree = ast.parse(textwrap.dedent(source))
        called_attributes = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert called_attributes.isdisjoint({"iter_row_mappings", "row_mappings"})
        assert "MeasurementSparseColumnarRows.from_rows" not in source


def test_sparse_measurement_coalescing_accepts_equivalent_nan_cells() -> None:
    axis_field = MeasurementRowAxisField.SLICE_INDEX.value
    rows = coalesced_sparse_measurement_row_mappings(
        (
            {
                axis_field: 2,
                "metric": float("nan"),
            },
            {
                axis_field: np.int64(2),
                "metric": np.float64(np.nan),
                "other_metric": 4.0,
            },
        )
    )

    assert len(rows) == 1
    assert np.isnan(rows[0]["metric"])
    assert rows[0]["other_metric"] == 4.0


def test_sparse_measurement_coalescing_rejects_distinct_finite_cells() -> None:
    axis_field = MeasurementRowAxisField.SLICE_INDEX.value
    with pytest.raises(ValueError, match="Conflicting sparse measurement values"):
        coalesced_sparse_measurement_row_mappings(
            (
                {axis_field: 2, "metric": 3.0},
                {axis_field: 2, "metric": 4.0},
            )
        )


def test_wide_measurement_projection_reads_columns_without_materializing_rows() -> None:
    rows = _ColumnOnlyMeasurementRows(
        {
            "slice_index": (0, 0),
            "cell_id": (7, 7),
            "feature_name": ("contrast", "contrast"),
            "result_value": (0.25, 0.75),
            "source_image_name": ("DNA", "DNA"),
            "direction": (0, 1),
        },
        fields=(
            FieldSpec("slice_index", int),
            FieldSpec("cell_id", int),
            FieldSpec("feature_name", str),
            FieldSpec("result_value", float),
            FieldSpec("source_image_name", str),
            FieldSpec("direction", int),
        ),
    )

    accumulator = WideMeasurementRowAccumulator(
        DEFAULT_RUNTIME_MEASUREMENT_ROW_IDENTITY_CONTRACT
    )
    accumulator.add(
        rows,
        lambda feature, qualifiers: "_".join(
            (feature, *(str(value) for _field, value in qualifiers))
        ),
        default_subject="Cells",
        default_scope=MeasurementScope.OBJECT,
        object_id_field="cell_id",
        qualifier_field_names=("direction",),
    )
    projected = accumulator.row_mappings_by_subject()["Cells"]

    assert projected == (
        {
            "slice_index": 0,
            "object_label": 7,
            "contrast_0": 0.25,
            "contrast_1": 0.75,
        },
    )


def test_wide_measurement_projection_does_not_treat_descriptor_axes_as_identity() -> (
    None
):
    rows = _ColumnOnlyMeasurementRows(
        {
            "slice_index": (0, 0),
            "object_label": (7, 7),
            "feature_name": ("Zernike_0_0", "Zernike_1_1"),
            "result_value": (0.25, 0.75),
            "n": (0, 1),
            "m": (0, 1),
        },
        fields=(
            FieldSpec("slice_index", int),
            FieldSpec("object_label", int),
            FieldSpec("feature_name", str),
            FieldSpec("result_value", float),
            FieldSpec("n", int),
            FieldSpec("m", int),
        ),
    )

    accumulator = WideMeasurementRowAccumulator(
        DEFAULT_RUNTIME_MEASUREMENT_ROW_IDENTITY_CONTRACT
    )
    accumulator.add(
        rows,
        lambda feature, _qualifiers: feature,
        default_subject="Cells",
        default_scope=MeasurementScope.OBJECT,
    )

    assert accumulator.row_mappings_by_subject()["Cells"] == (
        {
            "slice_index": 0,
            "object_label": 7,
            "Zernike_0_0": 0.25,
            "Zernike_1_1": 0.75,
        },
    )


def test_concatenated_measurement_rows_preserve_structural_missing_cells() -> None:
    identity_fields = (
        FieldSpec("slice_index", int),
        FieldSpec("object_label", int),
    )
    rows = ConcatenatedColumnarRows(
        (
            MeasurementSparseColumnarRows.from_rows(
                ({"slice_index": 1, "object_label": 1, "area": 4.0},),
                fields=(*identity_fields, FieldSpec("area", float)),
            ),
            MeasurementSparseColumnarRows.from_rows(
                (
                    {
                        "slice_index": 1,
                        "object_label": 1,
                        "Parent_Tile_of_grid": 1,
                    },
                ),
                fields=(
                    *identity_fields,
                    FieldSpec("Parent_Tile_of_grid", int),
                ),
            ),
        )
    )
    accumulator = WideMeasurementRowAccumulator(
        DEFAULT_RUNTIME_MEASUREMENT_ROW_IDENTITY_CONTRACT
    )

    accumulator.add(
        rows,
        lambda feature, _qualifiers: feature,
        default_subject="Cells",
        default_scope=MeasurementScope.OBJECT,
    )

    assert accumulator.row_mappings_by_subject()["Cells"] == (
        {
            "slice_index": 1,
            "object_label": 1,
            "area": 4.0,
            "Parent_Tile_of_grid": 1,
        },
    )


def test_long_form_projection_omits_structurally_missing_qualifiers() -> None:
    base_fields = (
        FieldSpec("slice_index", int),
        FieldSpec("object_label", int),
        FieldSpec("feature_name", str),
        FieldSpec("result_value", float),
    )
    rows = ConcatenatedColumnarRows(
        (
            MeasurementSparseColumnarRows.from_rows(
                (
                    {
                        "slice_index": 1,
                        "object_label": 1,
                        "feature_name": "intensity",
                        "result_value": 4.0,
                    },
                ),
                fields=base_fields,
            ),
            MeasurementSparseColumnarRows.from_rows(
                (
                    {
                        "slice_index": 1,
                        "object_label": 1,
                        "feature_name": "texture",
                        "result_value": 2.0,
                        "direction": 1,
                    },
                ),
                fields=(*base_fields, FieldSpec("direction", int)),
            ),
        )
    )
    accumulator = WideMeasurementRowAccumulator(
        DEFAULT_RUNTIME_MEASUREMENT_ROW_IDENTITY_CONTRACT
    )

    accumulator.add(
        rows,
        lambda feature, qualifiers: "_".join(
            (feature, *(str(value) for _field, value in qualifiers))
        ),
        default_subject="Cells",
        default_scope=MeasurementScope.OBJECT,
        qualifier_field_names=("direction",),
    )

    assert accumulator.row_mappings_by_subject()["Cells"] == (
        {
            "slice_index": 1,
            "object_label": 1,
            "intensity": 4.0,
            "texture_1": 2.0,
        },
    )


def test_wide_measurement_projection_uses_row_owned_scope_for_artifact_table() -> None:
    accumulator = WideMeasurementRowAccumulator(
        DEFAULT_RUNTIME_MEASUREMENT_ROW_IDENTITY_CONTRACT
    )
    accumulator.add(
        MeasurementSparseColumnarRows.from_rows(
            (
                {
                    "slice_index": 0,
                    "object_label": 1,
                    "object_name": "Cells",
                    "source_image_name": "DNA",
                    "feature_name": "Intensity_IntegratedIntensity_DNA",
                    "result_value": 3.0,
                },
                {
                    "slice_index": 0,
                    "object_label": 1,
                    "object_name": "Cells",
                    "source_image_name": "PH3",
                    "feature_name": "Intensity_IntegratedIntensity_PH3",
                    "result_value": 9.0,
                },
            ),
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("object_label", int),
                FieldSpec("object_name", str),
                FieldSpec("source_image_name", str),
                FieldSpec("feature_name", str),
                FieldSpec("result_value", float),
            ),
        ),
        lambda feature, _qualifiers: feature,
        default_subject="Measurements",
        default_scope=MeasurementScope.ARTIFACT,
    )

    assert accumulator.row_mappings_by_subject()["Cells"] == (
        {
            "slice_index": 0,
            "object_label": 1,
            "Intensity_IntegratedIntensity_DNA": 3.0,
            "Intensity_IntegratedIntensity_PH3": 9.0,
        },
    )
