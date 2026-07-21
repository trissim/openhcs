from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pytest

from openhcs.constants.constants import AllComponents
from openhcs.core.artifacts import (
    ArtifactSpec,
    ObjectLabelsArtifactType,
    ArtifactOutputPlan,
    ArtifactType,
    MeasurementsArtifactType,
    RelationshipsArtifactType,
)
from openhcs.core.measurement_row_materialization import (
    ConcatenatedColumnarRows,
    MeasurementProjectedColumnarRows,
    MeasurementRowObjectName,
    MeasurementRowOwnership,
    MeasurementRowSourceImageName,
    MeasurementSparseColumnarRows,
    MEASUREMENT_SPARSE_CELL,
)
from openhcs.core.runtime_artifact_queries import (
    RuntimeArtifactQueryContext,
    MeasurementLabelSliceFeatureBatchQueryCache,
    MeasurementLabelSliceFeatureBatchQuery,
    MeasurementLabelSliceFeatureQuery,
    MeasurementTableAxisProjection,
    MeasurementTableUnion,
    measurement_table_axis_values,
    measurement_row_mapping,
    runtime_measurement_tables_for_object,
    runtime_relationship,
)
from openhcs.core.measurement_feature_queries import (
    MeasurementAxisValueProjection,
    MeasurementFeatureQuery,
    MeasurementFeatureValueIndex,
    MeasurementObjectFeatureAxisBatchQueryCache,
    MeasurementObjectFeatureVectorBatchQuery,
    MeasurementTableObjectFeatureSemantics,
    MeasurementTableObjectFeatureSemanticsCache,
    matching_measurement_field,
    measurement_feature_candidates,
    ordered_measurement_feature_candidates,
    measurement_values_for_feature,
)
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
    MeasurementObjectRowIdentity,
    MeasurementRowMappingCache,
)
from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
    MeasurementScope,
    MeasurementSubject,
)
from openhcs.core.runtime_object_label_domains import (
    ObjectLabelDomain,
    ObjectLabelDomainScope,
)
from openhcs.core.runtime_relationships import (
    DirectedObjectRelationshipPayload,
    ObjectRelationshipDeclaration,
)
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxis,
    RuntimePlaneProjection,
)
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.runtime_measurements import (
    MeasurementTable,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelVariantData,
    ObjectLabelSet,
)
from openhcs.core.runtime_relationships import (
    ObjectRelationship,
)
from openhcs.core.source_image_provenance import (
    RuntimeSourceImageProvenancePlane,
    SourceImageProvenancePlanes,
)
from openhcs.interop.cellprofiler.measurement_dialect import (
    CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
)
from openhcs.core.runtime_artifact_values import RuntimeValue

AXIS_ID = "A01"


@pytest.mark.parametrize(
    ("declaration_type", "field_name"),
    (
        (MeasurementRowObjectName, MeasurementRowAxisField.OBJECT_NAME.value),
        (
            MeasurementRowSourceImageName,
            MeasurementRowAxisField.SOURCE_IMAGE_NAME.value,
        ),
    ),
)
def test_declared_text_ownership_ignores_structural_missing_cells(
    declaration_type,
    field_name: str,
) -> None:
    assert (
        declaration_type.value_from_row({field_name: MEASUREMENT_SPARSE_CELL}) is None
    )


def measurement_values_for_label_slices(
    measurement_tables: tuple[MeasurementTable, ...],
    feature_name: str,
    labels: ObjectLabelSet,
    *,
    plane_projector: RuntimePlaneProjection,
    object_name: str | None = None,
    row_axis: MeasurementRowAxisField = MeasurementRowAxisField.SLICE_INDEX,
    dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
) -> tuple[object, ...]:
    return MeasurementLabelSliceFeatureQuery(
        measurement_tables=measurement_tables,
        feature_name=feature_name,
        object_name=object_name,
        dialect=dialect,
        row_axis=row_axis,
        plane_projector=plane_projector,
    ).values_for_labels(labels)


def object_labels(
    labels: object,
    *,
    declared_object_id_domains: tuple[tuple[int, ...], ...],
) -> ObjectLabelSet:
    return ObjectLabelSet(
        name="Labels",
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain.declared(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=declared_object_id_domains,
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )


@dataclass(frozen=True, slots=True)
class MeasurementRow:
    object_name: str
    object_label: int


def test_runtime_measurement_query_matches_schema_and_row_object_subjects() -> None:
    store = RuntimeValueStore()
    _record_native(
        store,
        MeasurementTable(
            name="NucleiMeasurements",
            rows=MeasurementSparseColumnarRows.from_rows(
                ({"object_label": 1, "area": 42.0},),
                fields=(
                    FieldSpec("object_label", int),
                    FieldSpec("area", float),
                ),
            ),
            subject=MeasurementSubject(MeasurementScope.OBJECT, "Nuclei"),
        ),
        MeasurementsArtifactType,
    )
    _record_native(
        store,
        MeasurementTable(
            name="MixedMeasurements",
            rows=MeasurementSparseColumnarRows.from_rows(
                (
                    {"object_name": "Nuclei", "object_label": 1, "mean": 3.0},
                    {"object_name": "Cells", "object_label": 1, "mean": 9.0},
                ),
                fields=(
                    FieldSpec("object_name", str),
                    FieldSpec("object_label", int),
                    FieldSpec("mean", float),
                ),
            ),
            subject=MeasurementSubject(MeasurementScope.ARTIFACT, "MixedMeasurements"),
        ),
        MeasurementsArtifactType,
    )
    _record_native(
        store,
        MeasurementTable(
            name="ImageMeasurements",
            rows=MeasurementSparseColumnarRows.from_rows(
                ({"area": 100.0},), fields=(FieldSpec("area", float),)
            ),
            subject=MeasurementSubject(MeasurementScope.ARTIFACT, "ImageMeasurements"),
        ),
        MeasurementsArtifactType,
    )

    tables = runtime_measurement_tables_for_object(
        RuntimeArtifactQueryContext(store, AXIS_ID),
        "Nuclei",
    )

    assert [table.name for table in tables] == [
        "NucleiMeasurements",
        "MixedMeasurements",
    ]


def test_measurement_row_mapping_accepts_slotted_dataclasses() -> None:
    MeasurementRowMappingCache.process_cache().entries.clear()
    row = MeasurementRow(object_name="Nuclei", object_label=1)
    mapping = measurement_row_mapping(row)

    assert dict(mapping) == {
        "object_name": "Nuclei",
        "object_label": 1,
    }
    assert mapping["object_name"] == "Nuclei"
    with pytest.raises(KeyError):
        mapping["not_a_field"]
    assert MeasurementRowOwnership(object_name="Cells").annotate_row(
        {"area": 42.0}
    ) == {
        "area": 42.0,
        "object_name": "Cells",
    }


def test_measurement_row_mapping_cache_reuses_dataclass_rows() -> None:
    cache = MeasurementRowMappingCache.process_cache()
    cache.entries.clear()
    row = MeasurementRow(object_name="Nuclei", object_label=1)

    first = measurement_row_mapping(row)
    second = measurement_row_mapping(row)

    assert first is second


def test_projected_columnar_rows_preserve_none_values() -> None:
    rows = MeasurementProjectedColumnarRows(
        {
            "object_label": (1, 2),
            "nullable_measurement": (None, 4.0),
        },
        fields=(
            FieldSpec("object_label", int),
            FieldSpec("nullable_measurement", float, required=False),
        ),
    )

    assert rows.row_mappings() == (
        {"object_label": 1, "nullable_measurement": None},
        {"object_label": 2, "nullable_measurement": 4.0},
    )


def test_sparse_columnar_rows_omit_structural_missing_cells_only() -> None:
    rows = MeasurementSparseColumnarRows(
        {
            "object_label": (1, 2),
            "optional_measurement": (MEASUREMENT_SPARSE_CELL, None),
        },
        fields=(
            FieldSpec("object_label", int),
            FieldSpec("optional_measurement", float, required=False),
        ),
    )

    assert rows.row_mappings() == (
        {"object_label": 1},
        {"object_label": 2, "optional_measurement": None},
    )


def test_axis_filtered_sparse_rows_preserve_image_and_object_ownership() -> None:
    table = MeasurementTable(
        name="ClassifyObjectsMeasurements",
        rows=MeasurementProjectedColumnarRows(
            {
                "slice_index": (0, 0, 1),
                "feature_name": (
                    "Classify_Positive_NumObjectsPerBin",
                    "Classify_Positive",
                    "Classify_Positive_NumObjectsPerBin",
                ),
                "result_value": (3, 1, 4),
                "object_name": (
                    MEASUREMENT_SPARSE_CELL,
                    "Nuclei",
                    MEASUREMENT_SPARSE_CELL,
                ),
                "object_label": (
                    MEASUREMENT_SPARSE_CELL,
                    1,
                    MEASUREMENT_SPARSE_CELL,
                ),
            },
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("feature_name", str),
                FieldSpec("result_value", int),
                FieldSpec("object_name", str, required=False),
                FieldSpec("object_label", int, required=False),
            ),
        ),
        subject=MeasurementSubject(
            MeasurementScope.ARTIFACT, "ClassifyObjectsMeasurements"
        ),
    )

    projected = MeasurementTableAxisProjection(
        axis=MeasurementRowAxisField.SLICE_INDEX,
        value=0,
        table=table,
    ).apply()

    assert projected.rows.row_mappings() == (
        {
            "slice_index": 0,
            "feature_name": "Classify_Positive_NumObjectsPerBin",
            "result_value": 3,
        },
        {
            "slice_index": 0,
            "feature_name": "Classify_Positive",
            "result_value": 1,
            "object_name": "Nuclei",
            "object_label": 1,
        },
    )


def test_sparse_columnar_rows_coalesce_duplicate_axis_identity_fragments() -> None:
    rows = MeasurementSparseColumnarRows.from_rows(
        (
            {
                "slice_index": 0,
                "object_name": "Cells",
                "object_label": 1,
                "correlation_a_b": 0.5,
            },
            {
                "slice_index": 0,
                "object_name": "Cells",
                "object_label": 1,
                "correlation_a_c": 0.75,
            },
            {
                "slice_index": 0,
                "object_name": "Cells",
                "object_label": 2,
                "correlation_a_b": 0.25,
            },
        ),
        fields=(
            FieldSpec("slice_index", int),
            FieldSpec("object_name", str),
            FieldSpec("object_label", int),
            FieldSpec("correlation_a_b", float, required=False),
            FieldSpec("correlation_a_c", float, required=False),
        ),
    )

    assert rows.row_mappings() == (
        {
            "slice_index": 0,
            "object_name": "Cells",
            "object_label": 1,
            "correlation_a_b": 0.5,
            "correlation_a_c": 0.75,
        },
        {
            "slice_index": 0,
            "object_name": "Cells",
            "object_label": 2,
            "correlation_a_b": 0.25,
        },
    )


def test_projected_columnar_rows_omit_structural_missing_cells() -> None:
    rows = MeasurementProjectedColumnarRows(
        {
            "slice_index": (0, 0),
            "feature_name": ("Classify_Small_NumObjectsPerBin", "Classify_Small"),
            "result_value": (2, 1),
            "object_name": (MEASUREMENT_SPARSE_CELL, "Nuclei"),
            "object_label": (MEASUREMENT_SPARSE_CELL, 1),
        },
        fields=(
            FieldSpec("slice_index", int),
            FieldSpec("feature_name", str),
            FieldSpec("result_value", int),
            FieldSpec("object_name", str, required=False),
            FieldSpec("object_label", int, required=False),
        ),
    )

    assert rows.row_mappings() == (
        {
            "slice_index": 0,
            "feature_name": "Classify_Small_NumObjectsPerBin",
            "result_value": 2,
        },
        {
            "slice_index": 0,
            "feature_name": "Classify_Small",
            "result_value": 1,
            "object_name": "Nuclei",
            "object_label": 1,
        },
    )


def test_source_qualified_columnar_query_uses_row_source_over_table_source() -> None:
    table = MeasurementTable(
        name="MeasureObjectIntensityMeasurements",
        rows=MeasurementProjectedColumnarRows(
            {
                "object_name": ("Nuclei", "Nuclei", "Nuclei", "Nuclei"),
                "object_label": (1, 2, 1, 2),
                "source_image_name": ("OrigGreen", "OrigGreen", "OrigBlue", "OrigBlue"),
                "max_intensity": (0.5, 0.9, 0.1, 0.2),
            },
            fields=(
                FieldSpec("object_name", str),
                FieldSpec("object_label", int),
                FieldSpec("source_image_name", str),
                FieldSpec("max_intensity", float),
            ),
        ),
        source_image_name="OrigBlue__OrigGreen",
        subject=MeasurementSubject(MeasurementScope.IMAGE, "OrigBlue__OrigGreen"),
    )

    values = measurement_values_for_feature(
        (table,),
        "Intensity_MaxIntensity_OrigGreen",
        object_count=2,
        object_name="Nuclei",
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    assert values.tolist() == [0.5, 0.9]


def test_source_qualified_sequence_query_uses_row_source_over_table_source() -> None:
    table = MeasurementTable(
        name="MeasureObjectIntensityMeasurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            (
                {
                    "object_name": "Nuclei",
                    "object_label": 1,
                    "source_image_name": "OrigGreen",
                    "max_intensity": 0.5,
                },
                {
                    "object_name": "Nuclei",
                    "object_label": 2,
                    "source_image_name": "OrigGreen",
                    "max_intensity": 0.9,
                },
                {
                    "object_name": "Nuclei",
                    "object_label": 1,
                    "source_image_name": "OrigBlue",
                    "max_intensity": 0.1,
                },
                {
                    "object_name": "Nuclei",
                    "object_label": 2,
                    "source_image_name": "OrigBlue",
                    "max_intensity": 0.2,
                },
            ),
            fields=(
                FieldSpec("object_name", str),
                FieldSpec("object_label", int),
                FieldSpec("source_image_name", str),
                FieldSpec("max_intensity", float),
            ),
        ),
        source_image_name="OrigBlue__OrigGreen",
        subject=MeasurementSubject(MeasurementScope.IMAGE, "OrigBlue__OrigGreen"),
    )
    query = MeasurementFeatureQuery(
        "Intensity_MaxIntensity_OrigGreen",
        object_name="Nuclei",
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    assert query.table_may_carry_feature(table)
    values = query.values_for_domain((table,), (1, 2))

    assert values.tolist() == [0.5, 0.9]


def test_measurement_table_axis_values_omit_sparse_columnar_axis_cells() -> None:
    table = MeasurementTable(
        name="SparseAxisMeasurements",
        rows=MeasurementProjectedColumnarRows(
            {
                "slice_index": (0, MEASUREMENT_SPARSE_CELL, 1, None),
                "feature_name": ("first", "missing", "second", "absent"),
                "result_value": (2.0, MEASUREMENT_SPARSE_CELL, 4.0, 6.0),
            },
            fields=(
                FieldSpec("slice_index", int, required=False),
                FieldSpec("feature_name", str),
                FieldSpec("result_value", float, required=False),
            ),
        ),
        subject=MeasurementSubject(MeasurementScope.ARTIFACT, "SparseAxisMeasurements"),
    )

    assert measurement_table_axis_values(
        table,
        MeasurementRowAxisField.SLICE_INDEX,
    ) == {0, 1}


def test_axis_projection_returns_original_columnar_table_when_filter_keeps_all_rows() -> (
    None
):
    table = MeasurementTable(
        name="Measurements",
        rows=MeasurementProjectedColumnarRows(
            {
                "slice_index": (0, 0),
                "object_name": ("Nuclei", "Cells"),
                "object_label": (1, 1),
                "Intensity_MeanIntensity_CorrProtein": (4.0, 5.0),
            },
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("object_name", str),
                FieldSpec("object_label", int),
                FieldSpec("Intensity_MeanIntensity_CorrProtein", float),
            ),
        ),
        subject=MeasurementSubject(MeasurementScope.ARTIFACT, "Measurements"),
    )

    projected = MeasurementTableAxisProjection(
        axis=MeasurementRowAxisField.SLICE_INDEX,
        value=0,
        table=table,
    ).apply()

    assert projected is table


def test_concatenated_sparse_columnar_rows_do_not_expose_missing_sentinel() -> None:
    rows = ConcatenatedColumnarRows(
        (
            MeasurementSparseColumnarRows(
                {
                    "object_label": (1,),
                    "area": (4.0,),
                    "mean": (MEASUREMENT_SPARSE_CELL,),
                },
                fields=(
                    FieldSpec("object_label", int),
                    FieldSpec("area", float, required=False),
                    FieldSpec("mean", float, required=False),
                ),
                object_row_identity=MeasurementObjectRowIdentity.LABEL_ID,
            ),
            MeasurementSparseColumnarRows(
                {
                    "object_label": (2,),
                    "area": (MEASUREMENT_SPARSE_CELL,),
                    "mean": (8.0,),
                },
                fields=(
                    FieldSpec("object_label", int),
                    FieldSpec("area", float, required=False),
                    FieldSpec("mean", float, required=False),
                ),
                object_row_identity=MeasurementObjectRowIdentity.LABEL_ID,
            ),
        )
    )

    assert rows.row_mappings() == (
        {"object_label": 1, "area": 4.0},
        {"object_label": 2, "mean": 8.0},
    )
    area_values = measurement_values_for_feature(
        (
            MeasurementTable(
                name="SparseMeasurements",
                rows=rows,
                subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
            ),
        ),
        "area",
        object_count=1,
        object_name="Cells",
    )
    assert area_values.tolist() == [4.0]


def test_measurement_feature_query_uses_table_object_id_field() -> None:
    table = MeasurementTable(
        name="ObjectMeasurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            (
                {"cell_id": 2, "area": 20.0},
                {"cell_id": 1, "area": 10.0},
            ),
            fields=(
                FieldSpec("cell_id", int),
                FieldSpec("area", float),
            ),
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells", "cell_id"),
    )

    values = measurement_values_for_feature(
        (table,),
        "area",
        object_count=2,
        object_name="Cells",
    )

    assert values.tolist() == [10.0, 20.0]


def test_measurement_table_semantics_cache_reuses_table_identity() -> None:
    cache = MeasurementTableObjectFeatureSemanticsCache.process_cache()
    cache.entries.clear()
    table = MeasurementTable(
        name="ObjectMeasurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            (
                {"object_name": "Cells", "object_label": 1, "area": 10.0},
                {"object_name": "Cells", "object_label": 2, "area": 20.0},
            ),
            fields=(
                FieldSpec("object_name", str),
                FieldSpec("object_label", int),
                FieldSpec("area", float),
            ),
        ),
        subject=MeasurementSubject(MeasurementScope.ARTIFACT, "ObjectMeasurements"),
    )

    first = MeasurementTableObjectFeatureSemantics.from_table(table)
    second = MeasurementTableObjectFeatureSemantics.from_table(table)

    assert first is second
    assert first.object_names == ("Cells",)
    assert "area" in first.feature_names


def test_measurement_feature_candidates_match_cellprofiler_compact_metric_names() -> (
    None
):
    candidates = measurement_feature_candidates("Intensity_MADIntensity_typeI")

    assert "madintensity" in candidates
    assert "mad_intensity".replace("_", "") in candidates


def test_matching_measurement_field_prefers_specific_feature_suffix() -> None:
    row = {
        "object_label": 1,
        "Area": 25.0,
        "FormFactor": 0.95,
    }

    field = matching_measurement_field(
        row,
        ordered_measurement_feature_candidates("AreaShape_FormFactor"),
    )

    assert field == "FormFactor"


def test_cellprofiler_shape_area_query_uses_volume_when_area_is_empty() -> None:
    table = MeasurementTable(
        name="ShapeMeasurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            (
                {"slice_index": 0, "object_label": 1, "area": "", "volume": 12.0},
                {"slice_index": 0, "object_label": 2, "area": None, "volume": 24.0},
            ),
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("object_label", int),
                FieldSpec("area", float, required=False),
                FieldSpec("volume", float),
            ),
            object_row_identity=MeasurementObjectRowIdentity.LABEL_ID,
        ),
        subject=MeasurementSubject(
            MeasurementScope.OBJECT,
            "Cells",
            MeasurementRowAxisField.OBJECT_LABEL.value,
        ),
    )

    values = measurement_values_for_feature(
        (table,),
        "AreaShape_Area",
        object_count=2,
        object_name="Cells",
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    assert values.tolist() == [12.0, 24.0]


def test_heterogeneous_shape_rows_prefer_area_over_later_volume_alias() -> None:
    table = MeasurementTable(
        name="MixedShapeMeasurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            (
                {
                    "slice_index": 0,
                    "object_name": "Cells",
                    "object_label": 1,
                    "volume": 7.0,
                },
                {
                    "slice_index": 0,
                    "object_name": "Nuclei",
                    "object_label": 1,
                    "area": 3.0,
                },
                {
                    "slice_index": 0,
                    "object_name": "Nuclei",
                    "object_label": 2,
                    "area": 5.0,
                },
            ),
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("object_name", str),
                FieldSpec("object_label", int),
                FieldSpec("volume", float, required=False),
                FieldSpec("area", float, required=False),
            ),
        ),
        subject=MeasurementSubject(MeasurementScope.ARTIFACT, "MixedShapeMeasurements"),
    )
    labels = ObjectLabelSet(
        name="Nuclei",
        variant_data=ObjectLabelVariantData(
            labels=pytest.importorskip("numpy").array([[1, 2]], dtype="int32")
        ),
        domain=ObjectLabelDomain.declared(
            scope=ObjectLabelDomainScope.PAYLOAD,
            declared_object_ids=(1, 2),
        ),
    )

    values = measurement_values_for_label_slices(
        (table,),
        "AreaShape_Area",
        labels,
        plane_projector=RuntimePlaneProjection.selected(0, 1),
        object_name="Nuclei",
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    assert tuple(value.tolist() for value in values) == ([3.0, 5.0],)


def test_label_slice_measurement_lookup_uses_runtime_slice_axis() -> None:
    labels = object_labels(
        pytest.importorskip("numpy").array(
            (
                ((1, 2),),
                ((1, 0),),
            ),
            dtype="int32",
        ),
        declared_object_id_domains=((1, 2), (1,)),
    )
    table = MeasurementTable(
        name="ShapeMeasurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            (
                {"slice_index": 0, "object_label": 1, "FormFactor": 0.5},
                {"slice_index": 0, "object_label": 2, "FormFactor": 0.7},
                {"slice_index": 1, "object_label": 1, "FormFactor": 0.9},
            ),
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("object_label", int),
                FieldSpec("FormFactor", float),
            ),
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
    )

    values = measurement_values_for_label_slices(
        (table,),
        "AreaShape_FormFactor",
        labels,
        plane_projector=RuntimePlaneProjection.stack(2),
        object_name="Cells",
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    assert tuple(value.tolist() for value in values) == ([0.5, 0.7], [0.9])


def test_projected_payload_measurement_lookup_uses_selected_runtime_slice() -> None:
    np = pytest.importorskip("numpy")
    labels = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(labels=np.array(((1, 2),), dtype="int32")),
        domain=ObjectLabelDomain.declared(
            scope=ObjectLabelDomainScope.PAYLOAD,
            declared_object_ids=(1, 2),
        ),
    )
    table = MeasurementTable(
        name="ShapeMeasurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            (
                {"slice_index": 0, "object_label": 1, "FormFactor": 0.5},
                {"slice_index": 0, "object_label": 2, "FormFactor": 0.7},
                {"slice_index": 1, "object_label": 1, "FormFactor": 0.9},
                {"slice_index": 1, "object_label": 2, "FormFactor": 1.1},
            ),
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("object_label", int),
                FieldSpec("FormFactor", float),
            ),
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
    )

    values = measurement_values_for_label_slices(
        (table,),
        "AreaShape_FormFactor",
        labels,
        plane_projector=RuntimePlaneProjection.selected(1, 2),
        object_name="Cells",
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    assert tuple(value.tolist() for value in values) == ([0.9, 1.1],)


def test_payload_measurement_projection_ignores_source_contributor_alias_count() -> (
    None
):
    np = pytest.importorskip("numpy")
    labels = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(labels=np.array(((1, 2),), dtype="int32")),
        domain=ObjectLabelDomain.declared(
            scope=ObjectLabelDomainScope.PAYLOAD,
            declared_object_ids=(1, 2),
        ),
        source_image_names=("Image", "Mask"),
    )
    table = MeasurementTable(
        name="ShapeMeasurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            (
                {"slice_index": 0, "object_label": 1, "FormFactor": 0.5},
                {"slice_index": 0, "object_label": 2, "FormFactor": 0.7},
            ),
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("object_label", int),
                FieldSpec("FormFactor", float),
            ),
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
    )

    values = measurement_values_for_label_slices(
        (table,),
        "AreaShape_FormFactor",
        labels,
        plane_projector=RuntimePlaneProjection.selected(0, 1),
        object_name="Cells",
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    assert tuple(value.tolist() for value in values) == ([0.5, 0.7],)


def test_payload_measurement_lookup_rejects_unselected_runtime_stack() -> None:
    np = pytest.importorskip("numpy")
    labels = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(labels=np.array(((1,),), dtype="int32")),
        domain=ObjectLabelDomain.declared(
            scope=ObjectLabelDomainScope.PAYLOAD,
            declared_object_ids=(1,),
        ),
    )

    with pytest.raises(ValueError, match="requires one selected runtime slice"):
        measurement_values_for_label_slices(
            (),
            "AreaShape_FormFactor",
            labels,
            plane_projector=RuntimePlaneProjection.stack(2),
            object_name="Cells",
            dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
        )


def test_payload_measurement_lookup_accepts_declared_singleton_runtime_stack() -> None:
    np = pytest.importorskip("numpy")
    labels = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(labels=np.array(((1,),), dtype="int32")),
        domain=ObjectLabelDomain.declared(
            scope=ObjectLabelDomainScope.PAYLOAD,
            declared_object_ids=(1,),
        ),
    )
    table = MeasurementTable(
        name="ShapeMeasurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            ({"slice_index": 0, "object_label": 1, "FormFactor": 0.5},),
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("object_label", int),
                FieldSpec("FormFactor", float),
            ),
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
    )

    values = measurement_values_for_label_slices(
        (table,),
        "AreaShape_FormFactor",
        labels,
        plane_projector=RuntimePlaneProjection.stack(1),
        object_name="Cells",
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    assert tuple(value.tolist() for value in values) == ([0.5],)


def test_batch_label_slice_measurement_lookup_scans_each_axis_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")
    MeasurementLabelSliceFeatureBatchQueryCache.process_cache().entries.clear()
    MeasurementObjectFeatureAxisBatchQueryCache.process_cache().entries.clear()
    table_scans: list[tuple[int, int | None, tuple[str, ...]]] = []
    original_table_value_indexes = (
        MeasurementObjectFeatureVectorBatchQuery.table_value_indexes
    )

    def counted_table_value_indexes(
        query: MeasurementObjectFeatureVectorBatchQuery,
        table: MeasurementTable,
        table_query: MeasurementFeatureQuery,
        table_object_names: tuple[str, ...],
        query_objects_by_requested_object: Mapping[str, str | None],
        *,
        projection: MeasurementAxisValueProjection | None = None,
    ) -> dict[str, MeasurementFeatureValueIndex]:
        table_scans.append(
            (
                id(table),
                None if projection is None else projection.value,
                table_object_names,
            )
        )
        return original_table_value_indexes(
            query,
            table,
            table_query,
            table_object_names,
            query_objects_by_requested_object,
            projection=projection,
        )

    monkeypatch.setattr(
        MeasurementObjectFeatureVectorBatchQuery,
        "table_value_indexes",
        counted_table_value_indexes,
    )
    cell_labels = object_labels(
        np.array(
            (
                ((1, 2),),
                ((1, 0),),
            ),
            dtype="int32",
        ),
        declared_object_id_domains=((1, 2), (1,)),
    )
    nucleus_labels = object_labels(
        np.array(
            (
                ((1,),),
                ((1,),),
            ),
            dtype="int32",
        ),
        declared_object_id_domains=((1,), (1,)),
    )
    table = MeasurementTable(
        name="MixedShapeMeasurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            (
                {
                    "slice_index": 0,
                    "object_name": "Cells",
                    "object_label": 1,
                    "FormFactor": 0.5,
                },
                {
                    "slice_index": 0,
                    "object_name": "Cells",
                    "object_label": 2,
                    "FormFactor": 0.7,
                },
                {
                    "slice_index": 1,
                    "object_name": "Cells",
                    "object_label": 1,
                    "FormFactor": 0.9,
                },
                {
                    "slice_index": 0,
                    "object_name": "Nuclei",
                    "object_label": 1,
                    "FormFactor": 1.5,
                },
                {
                    "slice_index": 1,
                    "object_name": "Nuclei",
                    "object_label": 1,
                    "FormFactor": 1.9,
                },
            ),
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("object_name", str),
                FieldSpec("object_label", int),
                FieldSpec("FormFactor", float),
            ),
        ),
        subject=MeasurementSubject(MeasurementScope.ARTIFACT, "MixedShapeMeasurements"),
    )

    batch_values = MeasurementLabelSliceFeatureBatchQuery(
        measurement_tables=(table,),
        feature_name="AreaShape_FormFactor",
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
        row_axis=MeasurementRowAxisField.SLICE_INDEX,
        plane_projector=RuntimePlaneProjection.stack(2),
        labels_by_object={
            "Cells": cell_labels,
            "Nuclei": nucleus_labels,
        },
    ).values_by_object()

    assert tuple(value.tolist() for value in batch_values["Cells"]) == (
        [0.5, 0.7],
        [0.9],
    )
    assert tuple(value.tolist() for value in batch_values["Nuclei"]) == (
        [1.5],
        [1.9],
    )
    assert [(table_id, axis) for table_id, axis, _ in table_scans] == [
        (id(table), 0),
        (id(table), 1),
    ]
    assert all(
        set(object_names) == {"Cells", "Nuclei"}
        for _, _, object_names in table_scans
    )


def test_label_slice_measurement_lookup_preserves_producer_runtime_slice_axis() -> None:
    np = pytest.importorskip("numpy")
    labels = object_labels(
        np.ones((2, 1, 3), dtype="int32"),
        declared_object_id_domains=((1,), (1,)),
    )
    first = MeasurementTable(
        name="FirstIntensityMeasurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            (
                {
                    "slice_index": 0,
                    "object_name": "Tiles",
                    "object_label": 1,
                    "source_image_name": "DF_image",
                    "std_intensity": 0.25,
                },
            ),
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("object_name", str),
                FieldSpec("object_label", int),
                FieldSpec("source_image_name", str),
                FieldSpec("std_intensity", float),
            ),
        ),
        subject=MeasurementSubject(
            MeasurementScope.ARTIFACT, "FirstIntensityMeasurements"
        ),
    )
    second = MeasurementTable(
        name="SecondIntensityMeasurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            (
                {
                    "slice_index": 1,
                    "object_name": "Tiles",
                    "object_label": 1,
                    "source_image_name": "DF_image",
                    "std_intensity": 0.0,
                },
            ),
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("object_name", str),
                FieldSpec("object_label", int),
                FieldSpec("source_image_name", str),
                FieldSpec("std_intensity", float),
            ),
        ),
        subject=MeasurementSubject(
            MeasurementScope.ARTIFACT, "SecondIntensityMeasurements"
        ),
    )

    values = measurement_values_for_label_slices(
        (first, second),
        "Intensity_StdIntensity_DF_image",
        labels,
        plane_projector=RuntimePlaneProjection.stack(2),
        object_name="Tiles",
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    assert tuple(value.tolist() for value in values) == ([0.25], [0.0])


def test_axis_batch_cache_rebuilds_partial_object_axes() -> None:
    np = pytest.importorskip("numpy")
    cache = MeasurementObjectFeatureAxisBatchQueryCache.process_cache()
    cache.entries.clear()
    table = MeasurementTable(
        name="IntensityMeasurements",
        rows=MeasurementProjectedColumnarRows(
            {
                "slice_index": (0, 0, 1, 1, 2, 2),
                "object_name": (
                    "Nuclei",
                    "Nuclei",
                    "Nuclei",
                    "Nuclei",
                    "Nuclei",
                    "Nuclei",
                ),
                "object_label": (1, 2, 1, 2, 1, 2),
                "mean_intensity": (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            },
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("object_name", str),
                FieldSpec("object_label", int),
                FieldSpec("mean_intensity", float),
            ),
        ),
        source_image_name="CropBlue",
        subject=MeasurementSubject(MeasurementScope.IMAGE, "CropBlue"),
    )
    batch_query = MeasurementObjectFeatureVectorBatchQuery(
        "Intensity_MeanIntensity_CropBlue",
        ("Nuclei",),
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )
    table_map = {"Nuclei": (table,)}
    cache.store_value(
        batch_query.axis_cache_key(table_map, MeasurementRowAxisField.SLICE_INDEX),
        (
            batch_query.table_owners(table_map),
            {
                0: {"Nuclei": ({1: 1.0, 2: 2.0}, [])},
                1: {},
                2: {},
            },
        ),
    )
    labels = object_labels(
        np.array(
            (
                ((1, 2),),
                ((1, 2),),
                ((1, 2),),
            ),
            dtype="int32",
        ),
        declared_object_id_domains=((1, 2), (1, 2), (1, 2)),
    )

    values = measurement_values_for_label_slices(
        (table,),
        "Intensity_MeanIntensity_CropBlue",
        labels,
        plane_projector=RuntimePlaneProjection.stack(3),
        object_name="Nuclei",
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    assert tuple(value.tolist() for value in values) == (
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    )


def test_axis_batch_projection_rejects_missing_object_axes() -> None:
    np = pytest.importorskip("numpy")
    table = MeasurementTable(
        name="IntensityMeasurements",
        rows=MeasurementProjectedColumnarRows(
            {
                "slice_index": (0, 0, 0, 1, 2),
                "object_name": (
                    "Nuclei",
                    "Nuclei",
                    "Cytoplasm",
                    "Cytoplasm",
                    "Cytoplasm",
                ),
                "object_label": (1, 2, 1, 1, 1),
                "mean_intensity": (1.0, 2.0, 10.0, 20.0, 30.0),
            },
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("object_name", str),
                FieldSpec("object_label", int),
                FieldSpec("mean_intensity", float),
            ),
        ),
        source_image_name="CropBlue",
        subject=MeasurementSubject(MeasurementScope.IMAGE, "CropBlue"),
    )
    labels = object_labels(
        np.array(
            (
                ((1, 2),),
                ((1, 2),),
                ((1, 2),),
            ),
            dtype="int32",
        ),
        declared_object_id_domains=((1, 2), (1, 2), (1, 2)),
    )

    with pytest.raises(ValueError, match="does not match the declared label domain"):
        measurement_values_for_label_slices(
            (table,),
            "Intensity_MeanIntensity_CropBlue",
            labels,
            plane_projector=RuntimePlaneProjection.stack(3),
            object_name="Nuclei",
            dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
        )


def test_mixed_wide_and_long_measurement_rows_resolve_explicit_feature_rows() -> None:
    table = MeasurementTable(
        name="RelationshipMeasurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            (
                {
                    "parent_object_count": 2,
                    "child_object_count": 2,
                    "mean_children_per_parent": 1.0,
                },
                {
                    "object_name": "Objects1",
                    "object_label": 1,
                    "feature_name": "Children_Objects2_Count",
                    "result_value": 2.0,
                },
                {
                    "object_name": "Objects1",
                    "object_label": 2,
                    "feature_name": "Children_Objects2_Count",
                    "result_value": 0.0,
                },
            ),
            fields=(
                FieldSpec("parent_object_count", int, required=False),
                FieldSpec("child_object_count", int, required=False),
                FieldSpec("mean_children_per_parent", float, required=False),
                FieldSpec("object_name", str, required=False),
                FieldSpec("object_label", int, required=False),
                FieldSpec("feature_name", str, required=False),
                FieldSpec("result_value", float, required=False),
            ),
        ),
        subject=MeasurementSubject(
            MeasurementScope.ARTIFACT, "RelationshipMeasurements"
        ),
    )

    values = measurement_values_for_feature(
        (table,),
        "Children_Objects2_Count",
        object_count=2,
        object_name="Objects1",
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    assert values.tolist() == [2.0, 0.0]


def test_row_sequence_feature_semantics_ignore_stale_partition_fields() -> None:
    table = MeasurementTable(
        name="RelationshipMeasurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            (
                {
                    "object_name": "Objects2",
                    "object_label": 1,
                    "Parent_Objects1": 1,
                },
            ),
            fields=(
                FieldSpec("object_name", str),
                FieldSpec("object_label", int),
                FieldSpec("Parent_Objects1", int),
            ),
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Objects2"),
    )

    semantics = MeasurementTableObjectFeatureSemantics.from_table(table)

    assert "Children_Objects2_Count" not in semantics.feature_names


def test_measurement_table_axis_query_projects_sequence_rows() -> None:
    table = MeasurementTable(
        name="ObjectMeasurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            (
                {"slice_index": 0, "object_label": 1, "area": 10.0},
                {"slice_index": 1, "object_label": 1, "area": 20.0},
            ),
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("object_label", int),
                FieldSpec("area", float),
            ),
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Objects"),
    )

    query = MeasurementTableAxisProjection(MeasurementRowAxisField.SLICE_INDEX, 1)
    projected = MeasurementTableAxisProjection(
        axis=query.axis,
        value=query.value,
        table=table,
    ).apply()

    assert query.axis is MeasurementRowAxisField.SLICE_INDEX
    assert query.value == 1
    assert projected.rows.row_mappings() == (
        {"slice_index": 1, "object_label": 1, "area": 20.0},
    )
    assert MeasurementTableObjectFeatureSemantics.from_table(
        table
    ).feature_names == frozenset({"area"})


def test_measurement_table_union_preserves_compatible_schema() -> None:
    subject = MeasurementSubject(MeasurementScope.OBJECT, "Cells", "cell_id")
    first = MeasurementTable(
        name="CellMeasurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            ({"cell_id": 1, "area": 10.0},),
            fields=(
                FieldSpec("cell_id", int),
                FieldSpec("area", float),
            ),
        ),
        subject=subject,
    )
    second = MeasurementTable(
        name="CellMeasurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            ({"cell_id": 2, "area": 20.0},),
            fields=(
                FieldSpec("cell_id", int),
                FieldSpec("area", float),
            ),
        ),
        subject=subject,
    )

    union = MeasurementTableUnion("CellMeasurements", (first, second)).as_table()

    assert union.rows.fields == (FieldSpec("cell_id", int), FieldSpec("area", float))
    assert union.subject.object_name == "Cells"
    assert union.subject.object_id_field == "cell_id"
    assert union.subject == subject
    assert union.rows.row_mappings() == (
        {"cell_id": 1, "area": 10.0},
        {"cell_id": 2, "area": 20.0},
    )


def test_measurement_table_union_composes_ordered_source_provenance() -> None:
    subject = MeasurementSubject(MeasurementScope.OBJECT, "Cells", "cell_id")
    paths = ("/plate/site-1.tif", "/plate/site-2.tif")
    tables = tuple(
        MeasurementTable(
            name="CellMeasurements",
            rows=MeasurementSparseColumnarRows.from_rows(
                ({"cell_id": index, "area": float(index * 10)},),
                fields=(FieldSpec("cell_id", int), FieldSpec("area", float)),
            ),
            subject=subject,
            source_image_provenance_planes=(
                SourceImageProvenancePlanes.from_components(paths=(path,))
            ),
        )
        for index, path in enumerate(paths, start=1)
    )

    union = MeasurementTableUnion("CellMeasurements", tables).as_table()

    assert union.rows.row_mappings() == (
        {"cell_id": 1, "area": 10.0},
        {"cell_id": 2, "area": 20.0},
    )
    assert union.source_provenance.source_plane_count == 2
    assert all(
        isinstance(plane, RuntimeSourceImageProvenancePlane)
        for plane in union.source_image_provenance_planes.planes
    )
    assert union.source_image_provenance_planes.paths == paths
    assert union.source_provenance.for_source_plane(0).source_path == paths[0]
    assert union.source_provenance.for_source_plane(1).source_path == paths[1]


def test_measurement_table_union_bundles_sources_per_declared_runtime_slice() -> None:
    subject = MeasurementSubject(MeasurementScope.IMAGE, "quality_metrics")
    tables = tuple(
        MeasurementTable(
            name="quality_metrics",
            rows=MeasurementSparseColumnarRows.from_rows(
                (
                    {"slice_index": 0, "focus_score": float(channel)},
                    {"slice_index": 1, "focus_score": float(channel + 1)},
                ),
                fields=(
                    FieldSpec("slice_index", int),
                    FieldSpec("focus_score", float),
                ),
            ),
            source_image_name="InputImages",
            subject=subject,
            source_image_provenance_planes=(
                SourceImageProvenancePlanes.from_components(
                    paths=(
                        f"/plate/A01_s001_w{channel}.tif",
                        f"/plate/A01_s002_w{channel}.tif",
                    ),
                    component_metadata=(
                        {
                            "well": "A01",
                            "site": "1",
                            "channel": str(channel),
                        },
                        {
                            "well": "A01",
                            "site": "2",
                            "channel": str(channel),
                        },
                    ),
                )
            ),
            source_image_names=("InputImages", "InputImages"),
        )
        for channel in (1, 2)
    )

    metadata = MeasurementTableUnion("quality_metrics", tables).source_metadata()

    assert metadata.source_provenance.source_plane_count == 2
    assert tuple(
        metadata.source_provenance.for_source_plane(index).source_component_metadata[
            "site"
        ]
        for index in range(2)
    ) == ("1", "2")
    assert tuple(
        tuple(
            contributor.path
            for contributor in metadata.source_provenance.for_source_plane(
                index
            ).source_image_provenance_planes.contributors
        )
        for index in range(2)
    ) == (
        ("/plate/A01_s001_w1.tif", "/plate/A01_s001_w2.tif"),
        ("/plate/A01_s002_w1.tif", "/plate/A01_s002_w2.tif"),
    )


def test_measurement_table_union_accepts_axisless_payload_domain() -> None:
    table = MeasurementTable(
        name="CellMeasurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            ({"cell_id": 1, "area": 10.0},),
            fields=(
                FieldSpec("cell_id", int),
                FieldSpec("area", float),
            ),
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells", "cell_id"),
    )

    assert (
        MeasurementTableUnion("CellMeasurements", (table,)).row_axis_domain(
            MeasurementRowAxisField.SLICE_INDEX
        )
        is None
    )


def test_measurement_table_union_preserves_payload_rows_in_axis_declaring_table() -> (
    None
):
    table = MeasurementTable(
        name="CellMeasurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            (
                {"slice_index": 0, "cell_id": 1, "area": 10.0},
                {"cell_id": 1, "Children_Nuclei_Count": 1},
            ),
            fields=(
                FieldSpec("slice_index", int, required=False),
                FieldSpec("cell_id", int),
                FieldSpec("area", float, required=False),
                FieldSpec("Children_Nuclei_Count", int, required=False),
            ),
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells", "cell_id"),
    )

    assert MeasurementTableUnion(
        "CellMeasurements",
        (table,),
    ).row_axis_domain(MeasurementRowAxisField.SLICE_INDEX) == (0,)


def test_measurement_table_union_rejects_mixed_slice_domains() -> None:
    tables = (
        MeasurementTable(
            name="CellMeasurements",
            rows=MeasurementSparseColumnarRows.from_rows(
                ({"slice_index": 0, "cell_id": 1, "area": 10.0},),
                fields=(
                    FieldSpec("slice_index", int),
                    FieldSpec("cell_id", int),
                    FieldSpec("area", float),
                ),
            ),
            subject=MeasurementSubject(MeasurementScope.ARTIFACT, "CellMeasurements"),
        ),
        MeasurementTable(
            name="CellMeasurements",
            rows=MeasurementSparseColumnarRows.from_rows(
                ({"cell_id": 2, "area": 20.0},),
                fields=(
                    FieldSpec("cell_id", int),
                    FieldSpec("area", float),
                ),
            ),
            subject=MeasurementSubject(MeasurementScope.ARTIFACT, "CellMeasurements"),
        ),
    )

    with pytest.raises(
        ValueError,
        match="mixes declared and axisless 'slice_index' row domains",
    ):
        MeasurementTableUnion("CellMeasurements", tables).row_axis_domain(
            MeasurementRowAxisField.SLICE_INDEX
        )


def test_measurement_table_union_drops_incompatible_schema_facts() -> None:
    first = MeasurementTable(
        name="MixedMeasurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            ({"object_label": 1, "area": 10.0},),
            fields=(
                FieldSpec("object_label", int),
                FieldSpec("area", float),
            ),
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
    )
    second = MeasurementTable(
        name="MixedMeasurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            ({"object_label": 1, "area": 20.0},),
            fields=(
                FieldSpec("object_label", int),
                FieldSpec("area", float),
            ),
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Nuclei"),
    )

    with pytest.raises(
        ValueError,
        match="require one exact nominal subject",
    ):
        MeasurementTableUnion("MixedMeasurements", (first, second)).as_table()


def test_measurement_table_axis_query_projects_table_sequences() -> None:
    first = MeasurementTable(
        name="FirstMeasurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            (
                {"slice_index": 0, "area": 10.0},
                {"slice_index": 1, "area": 20.0},
            ),
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("area", float),
            ),
        ),
        subject=MeasurementSubject(MeasurementScope.ARTIFACT, "FirstMeasurements"),
    )
    second = MeasurementTable(
        name="SecondMeasurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            ({"area": 99.0},), fields=(FieldSpec("area", float),)
        ),
        subject=MeasurementSubject(MeasurementScope.ARTIFACT, "SecondMeasurements"),
    )

    projected = MeasurementTableAxisProjection(
        MeasurementRowAxisField.SLICE_INDEX,
        1,
    ).tables((first, second))

    assert projected[0].rows.row_mappings() == ({"slice_index": 1, "area": 20.0},)
    assert projected[1] is second


def test_axis_specific_measurement_table_query_projects_runtime_slice_axis() -> None:
    runtime_table = MeasurementTable(
        name="RuntimeObjectMeasurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            (
                {"slice_index": 0, "area": 10.0},
                {"slice_index": 1, "area": 20.0},
            ),
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("area", float),
            ),
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Objects"),
    )

    assert tuple(
        MeasurementTableAxisProjection(
            MeasurementRowAxisField.SLICE_INDEX,
            1,
        )
        .tables((runtime_table,))[0]
        .rows.row_mappings()
    ) == ({"slice_index": 1, "area": 20.0},)


def test_runtime_relationship_query_reconstructs_typed_relationship() -> None:
    store = RuntimeValueStore()
    declaration = ObjectRelationshipDeclaration.parent_child(
        source=ArtifactSpec.output("Cells", ObjectLabelsArtifactType).ref(),
        target=ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType).ref(),
        producer_module_number=1,
    )
    _record_native(
        store,
        ObjectRelationship(
            name="ParentChild",
            declaration=declaration,
            payload=DirectedObjectRelationshipPayload(
                source_ids=(10, 11),
                target_ids=(1, 2),
                slice_indices=(),
                slice_count=None,
            ),
        ),
        RelationshipsArtifactType,
    )

    relationship = runtime_relationship(
        RuntimeArtifactQueryContext(store, AXIS_ID),
        "ParentChild",
    )

    assert relationship.declaration.source.name == "Cells"
    assert relationship.declaration.target.name == "Nuclei"
    assert relationship.payload.source_ids == (10, 11)
    assert relationship.payload.target_ids == (1, 2)
    assert relationship.declaration.relationship_type == "parent_child"


def test_runtime_artifact_ambiguity_reports_locations_without_payload_repr() -> None:
    class ReprMustNotRun:
        def __repr__(self) -> str:
            raise AssertionError("runtime artifact payload repr was evaluated")

    store = RuntimeValueStore()
    for group_key in ("first", "second"):
        _record_native(
            store,
            MeasurementTable(
                name="SharedMeasurements",
                rows=MeasurementSparseColumnarRows.from_rows(
                    ({"payload": ReprMustNotRun()},),
                    fields=(FieldSpec("payload", ReprMustNotRun),),
                ),
                subject=MeasurementSubject(
                    MeasurementScope.ARTIFACT, "SharedMeasurements"
                ),
            ),
            MeasurementsArtifactType,
            group_key=group_key,
        )

    with pytest.raises(RuntimeError, match="Ambiguous runtime artifact"):
        RuntimeArtifactQueryContext(store, AXIS_ID).resolve(
            name="SharedMeasurements",
            artifact_type=MeasurementsArtifactType,
        )


def _record_native(
    store: RuntimeValueStore,
    native_value: MeasurementTable | ObjectRelationship,
    kind: ArtifactType,
    *,
    group_key: str | None = None,
) -> None:
    value = RuntimeValue.normalize(
        ArtifactOutputPlan(
            name=native_value.name,
            path=f"/memory/{native_value.name}.pkl",
            artifact_type=kind,
            group_keys=(group_key,) if group_key is not None else (),
            group_component=(AllComponents.CHANNEL if group_key is not None else None),
        ),
        native_value,
        axis_id=AXIS_ID,
    )
    store.record(
        value,
        path=f"/memory/{native_value.name}.pkl",
        backend="memory",
    )
