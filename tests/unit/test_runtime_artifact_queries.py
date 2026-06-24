from __future__ import annotations

from dataclasses import dataclass

import pytest

from openhcs.core.artifacts import ArtifactKind, ArtifactOutputPlan
from openhcs.core.measurement_row_materialization import (
    ConcatenatedColumnarRows,
    MeasurementProjectedColumnarRows,
    MeasurementRowOwnership,
    MeasurementSparseColumnarRows,
    MEASUREMENT_SPARSE_CELL,
)
from openhcs.core.runtime_artifact_queries import (
    RuntimeArtifactQueryContext,
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
    MeasurementTableObjectFeatureSemantics,
    MeasurementTableObjectFeatureSemanticsCache,
    matching_measurement_field,
    measurement_feature_candidates,
    ordered_measurement_feature_candidates,
    measurement_values_for_feature,
)
from openhcs.core.runtime_semantics import (
    FieldSpec,
    MeasurementRowAxisField,
    MeasurementRowMappingCache,
    MeasurementScope,
    MeasurementSubject,
    RelationshipSemantics,
)
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.runtime_values import (
    MeasurementTable,
    ObjectRelationship,
    normalize_artifact_value,
)
from openhcs.interop.cellprofiler.measurement_dialect import (
    CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
)


AXIS_ID = "A01"


def measurement_values_for_label_slices(
    measurement_tables: tuple[MeasurementTable, ...],
    feature_name: str,
    labels: object,
    *,
    object_name: str | None = None,
    row_axis: MeasurementRowAxisField = MeasurementRowAxisField.SLICE_INDEX,
    row_axis_start: int | None = None,
    dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
) -> tuple[object, ...]:
    return MeasurementLabelSliceFeatureQuery(
        measurement_tables=measurement_tables,
        feature_name=feature_name,
        object_name=object_name,
        dialect=dialect,
        row_axis=row_axis,
    ).values_for_labels(labels, row_axis_start=row_axis_start)


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
            rows=({"object_label": 1, "area": 42.0},),
            object_name="Nuclei",
        ),
        ArtifactKind.MEASUREMENTS,
    )
    _record_native(
        store,
        MeasurementTable(
            name="MixedMeasurements",
            rows=(
                {"object_name": "Nuclei", "object_label": 1, "mean": 3.0},
                {"object_name": "Cells", "object_label": 1, "mean": 9.0},
            ),
        ),
        ArtifactKind.MEASUREMENTS,
    )
    _record_native(
        store,
        MeasurementTable(name="ImageMeasurements", rows=({"area": 100.0},)),
        ArtifactKind.MEASUREMENTS,
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
    assert MeasurementRowOwnership(object_name="Cells").annotate_row({"area": 42.0}) == {
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
        }
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
        }
    )

    assert rows.row_mappings() == (
        {"object_label": 1},
        {"object_label": 2, "optional_measurement": None},
    )


def test_projected_columnar_rows_omit_structural_missing_cells() -> None:
    rows = MeasurementProjectedColumnarRows(
        {
            "slice_index": (0, 0),
            "feature_name": ("Classify_Small_NumObjectsPerBin", "Classify_Small"),
            "result_value": (2, 1),
            "object_name": (MEASUREMENT_SPARSE_CELL, "Nuclei"),
            "object_label": (MEASUREMENT_SPARSE_CELL, 1),
        }
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


def test_measurement_table_axis_values_omit_sparse_columnar_axis_cells() -> None:
    table = MeasurementTable(
        name="SparseAxisMeasurements",
        rows=MeasurementProjectedColumnarRows(
            {
                "slice_index": (0, MEASUREMENT_SPARSE_CELL, 1, None),
                "feature_name": ("first", "missing", "second", "absent"),
                "result_value": (2.0, MEASUREMENT_SPARSE_CELL, 4.0, 6.0),
            }
        ),
    )

    assert measurement_table_axis_values(
        table,
        MeasurementRowAxisField.SLICE_INDEX,
    ) == {0, 1}


def test_axis_projection_returns_original_columnar_table_when_filter_keeps_all_rows() -> None:
    table = MeasurementTable(
        name="Measurements",
        rows=MeasurementProjectedColumnarRows(
            {
                "image_number": (1, 1),
                "object_name": ("Nuclei", "Cells"),
                "object_label": (1, 1),
                "Intensity_MeanIntensity_CorrProtein": (4.0, 5.0),
            }
        ),
        object_id_field="object_label",
    )

    projected = MeasurementTableAxisProjection(
        axis=MeasurementRowAxisField.IMAGE_NUMBER,
        value=1,
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
                }
            ),
            MeasurementSparseColumnarRows(
                {
                    "object_label": (2,),
                    "area": (MEASUREMENT_SPARSE_CELL,),
                    "mean": (8.0,),
                }
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
                object_id_field="object_label",
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
        rows=(
            {"cell_id": 2, "area": 20.0},
            {"cell_id": 1, "area": 10.0},
        ),
        object_name="Cells",
        object_id_field="cell_id",
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
        rows=(
            {"object_name": "Cells", "object_label": 1, "area": 10.0},
            {"object_name": "Cells", "object_label": 2, "area": 20.0},
        ),
        fields=(
            FieldSpec("object_name"),
            FieldSpec("object_label"),
            FieldSpec("area"),
        ),
    )

    first = MeasurementTableObjectFeatureSemantics.from_table(table)
    second = MeasurementTableObjectFeatureSemantics.from_table(table)

    assert first is second
    assert first.object_names == ("Cells",)
    assert "area" in first.feature_names


def test_measurement_feature_candidates_match_cellprofiler_compact_metric_names() -> None:
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
        rows=(
            {"image_number": 1, "object_label": 1, "area": "", "volume": 12.0},
            {"image_number": 1, "object_label": 2, "area": None, "volume": 24.0},
        ),
        object_name="Cells",
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
        rows=(
            {"slice_index": 0, "object_name": "Cells", "object_label": 1, "volume": 7.0},
            {"slice_index": 0, "object_name": "Nuclei", "object_label": 1, "area": 3.0},
            {"slice_index": 0, "object_name": "Nuclei", "object_label": 2, "area": 5.0},
        ),
    )
    labels = pytest.importorskip("numpy").array([[1, 2]], dtype="int32")

    values = measurement_values_for_label_slices(
        (table,),
        "AreaShape_Area",
        labels,
        object_name="Nuclei",
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    assert tuple(value.tolist() for value in values) == ([3.0, 5.0],)


def test_label_slice_measurement_lookup_uses_declared_image_number_axis() -> None:
    labels = pytest.importorskip("numpy").array(
        (
            ((1, 2),),
            ((1, 0),),
        ),
        dtype="int32",
    )
    table = MeasurementTable(
        name="ShapeMeasurements",
        rows=(
            {"image_number": 1, "object_label": 1, "FormFactor": 0.5},
            {"image_number": 1, "object_label": 2, "FormFactor": 0.7},
            {"image_number": 2, "object_label": 1, "FormFactor": 0.9},
        ),
        object_name="Cells",
    )

    values = measurement_values_for_label_slices(
        (table,),
        "AreaShape_FormFactor",
        labels,
        object_name="Cells",
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    assert tuple(value.tolist() for value in values) == ([0.5, 0.7], [0.9])


def test_batch_label_slice_measurement_lookup_uses_per_object_axis_semantics() -> None:
    np = pytest.importorskip("numpy")
    cell_labels = np.array(
        (
            ((1, 2),),
            ((1, 0),),
        ),
        dtype="int32",
    )
    nucleus_labels = np.array(
        (
            ((1,),),
            ((1,),),
        ),
        dtype="int32",
    )
    table = MeasurementTable(
        name="MixedShapeMeasurements",
        rows=(
            {"image_number": 1, "object_name": "Cells", "object_label": 1, "FormFactor": 0.5},
            {"image_number": 1, "object_name": "Cells", "object_label": 2, "FormFactor": 0.7},
            {"image_number": 2, "object_name": "Cells", "object_label": 1, "FormFactor": 0.9},
            {"image_number": 1, "object_name": "Nuclei", "object_label": 1, "FormFactor": 1.5},
            {"image_number": 2, "object_name": "Nuclei", "object_label": 1, "FormFactor": 1.9},
        ),
    )

    batch_values = MeasurementLabelSliceFeatureBatchQuery(
        measurement_tables=(table,),
        feature_name="AreaShape_FormFactor",
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
        row_axis=MeasurementRowAxisField.IMAGE_NUMBER,
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


def test_mixed_wide_and_long_measurement_rows_resolve_explicit_feature_rows() -> None:
    table = MeasurementTable(
        name="RelationshipMeasurements",
        rows=(
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
        rows=(
            {
                "object_name": "Objects2",
                "object_label": 1,
                "Parent_Objects1": 1,
            },
        ),
        object_name="Objects2",
        fields=(
            FieldSpec("object_name"),
            FieldSpec("object_label"),
            FieldSpec("Children_Objects2_Count"),
        ),
        validated_runtime_schema=True,
    )

    semantics = MeasurementTableObjectFeatureSemantics.from_table(table)

    assert "Children_Objects2_Count" not in semantics.feature_names


def test_measurement_table_axis_query_projects_sequence_rows() -> None:
    table = MeasurementTable(
        name="ObjectMeasurements",
        rows=(
            {"slice_index": 0, "object_label": 1, "area": 10.0},
            {"slice_index": 1, "object_label": 1, "area": 20.0},
        ),
        object_name="Objects",
    )

    query = MeasurementTableAxisProjection(MeasurementRowAxisField.SLICE_INDEX, 1)
    projected = MeasurementTableAxisProjection(
        axis=query.axis,
        value=query.value,
        table=table,
    ).apply()

    assert query.axis is MeasurementRowAxisField.SLICE_INDEX
    assert query.value == 1
    assert tuple(projected.rows) == (
        {"slice_index": 1, "object_label": 1, "area": 20.0},
    )


def test_measurement_table_union_preserves_compatible_schema() -> None:
    subject = MeasurementSubject(MeasurementScope.OBJECT, "Cells", "cell_id")
    first = MeasurementTable(
        name="CellMeasurements",
        rows=({"cell_id": 1, "area": 10.0},),
        fields=(FieldSpec("cell_id"), FieldSpec("area")),
        object_name="Cells",
        object_id_field="cell_id",
        subject=subject,
        validated_runtime_schema=True,
    )
    second = MeasurementTable(
        name="CellMeasurements",
        rows=({"cell_id": 2, "area": 20.0},),
        fields=(FieldSpec("cell_id"), FieldSpec("area")),
        object_name="Cells",
        object_id_field="cell_id",
        subject=subject,
        validated_runtime_schema=True,
    )

    union = MeasurementTableUnion("CellMeasurements", (first, second)).as_table()

    assert union.fields == (FieldSpec("cell_id"), FieldSpec("area"))
    assert union.object_name == "Cells"
    assert union.object_id_field == "cell_id"
    assert union.subject == subject
    assert union.validated_runtime_schema is True
    assert union.schema_loss_reasons == frozenset()
    assert tuple(union.rows) == (
        {"cell_id": 1, "area": 10.0},
        {"cell_id": 2, "area": 20.0},
    )


def test_measurement_table_union_drops_incompatible_schema_facts() -> None:
    first = MeasurementTable(
        name="MixedMeasurements",
        rows=({"object_label": 1, "area": 10.0},),
        fields=(FieldSpec("object_label"), FieldSpec("area")),
        object_name="Cells",
    )
    second = MeasurementTable(
        name="MixedMeasurements",
        rows=({"object_label": 1, "area": 20.0},),
        fields=(FieldSpec("object_label"), FieldSpec("area")),
        object_name="Nuclei",
    )

    union = MeasurementTableUnion("MixedMeasurements", (first, second)).as_table()

    assert union.fields == (FieldSpec("object_label"), FieldSpec("area"))
    assert union.object_name is None
    assert union.subject.scope is MeasurementScope.ARTIFACT
    assert union.validated_runtime_schema is False
    assert "object_name" in union.schema_loss_reasons
    assert "subject" in union.schema_loss_reasons


def test_measurement_table_axis_query_projects_table_sequences() -> None:
    first = MeasurementTable(
        name="FirstMeasurements",
        rows=(
            {"image_number": 1, "area": 10.0},
            {"image_number": 2, "area": 20.0},
        ),
    )
    second = MeasurementTable(
        name="SecondMeasurements",
        rows=({"area": 99.0},),
    )

    projected = MeasurementTableAxisProjection(
        MeasurementRowAxisField.IMAGE_NUMBER,
        2,
    ).tables((first, second))

    assert tuple(projected[0].rows) == ({"image_number": 2, "area": 20.0},)
    assert projected[1] is second


def test_axis_specific_measurement_table_query_projects_declared_axes() -> None:
    table = MeasurementTable(
        name="ObjectMeasurements",
        rows=(
            {"slice_index": 0, "image_number": 1, "area": 10.0},
            {"slice_index": 1, "image_number": 2, "area": 20.0},
        ),
        object_name="Objects",
    )

    assert tuple(
        MeasurementTableAxisProjection(
            MeasurementRowAxisField.SLICE_INDEX,
            1,
        ).tables((table,))[0].rows
    ) == (
        {"slice_index": 1, "image_number": 2, "area": 20.0},
    )
    assert tuple(
        MeasurementTableAxisProjection(
            MeasurementRowAxisField.IMAGE_NUMBER,
            2,
        ).tables((table,))[0].rows
    ) == (
        {"slice_index": 1, "image_number": 2, "area": 20.0},
    )


def test_runtime_relationship_query_reconstructs_typed_relationship() -> None:
    store = RuntimeValueStore()
    semantics = RelationshipSemantics.parent_child("Cells", "Nuclei")
    _record_native(
        store,
        ObjectRelationship(
            name="ParentChild",
            source=semantics.source,
            target=semantics.target,
            source_ids=(10, 11),
            target_ids=(1, 2),
            relationship_type=semantics.relationship_type,
        ),
        ArtifactKind.RELATIONSHIPS,
    )

    relationship = runtime_relationship(
        RuntimeArtifactQueryContext(store, AXIS_ID),
        "ParentChild",
    )

    assert relationship.source.name == "Cells"
    assert relationship.target.name == "Nuclei"
    assert relationship.source_ids == (10, 11)
    assert relationship.target_ids == (1, 2)
    assert relationship.relationship_type == "parent_child"


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
                rows=({"payload": ReprMustNotRun()},),
            ),
            ArtifactKind.MEASUREMENTS,
            group_key=group_key,
        )

    with pytest.raises(RuntimeError, match="Ambiguous runtime artifact"):
        RuntimeArtifactQueryContext(store, AXIS_ID).resolve(
            name="SharedMeasurements",
            kind=ArtifactKind.MEASUREMENTS,
        )


def _record_native(
    store: RuntimeValueStore,
    native_value: MeasurementTable | ObjectRelationship,
    kind: ArtifactKind,
    *,
    group_key: str | None = None,
) -> None:
    value = normalize_artifact_value(
        ArtifactOutputPlan(
            name=native_value.name,
            path=f"/memory/{native_value.name}.pkl",
            kind=kind,
            group_keys=(group_key,) if group_key is not None else (),
        ),
        native_value,
        axis_id=AXIS_ID,
    )
    store.record(
        value,
        path=f"/memory/{native_value.name}.pkl",
        backend="memory",
    )
