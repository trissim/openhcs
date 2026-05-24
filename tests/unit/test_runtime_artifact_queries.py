from __future__ import annotations

from dataclasses import dataclass

import pytest

from openhcs.core.artifacts import ArtifactKind, ArtifactOutputPlan
from openhcs.core.runtime_artifact_queries import (
    RuntimeArtifactQueryContext,
    MeasurementTableAxisProjection,
    MeasurementTableAxisQuery,
    MeasurementTableObjectFeatureSemantics,
    MeasurementTableObjectFeatureSemanticsCache,
    annotate_measurement_row_object,
    matching_measurement_field,
    measurement_feature_candidates,
    ordered_measurement_feature_candidates,
    measurement_row_mapping,
    measurement_values_for_feature,
    measurement_values_for_label_slices,
    runtime_measurement_tables_for_object,
    runtime_relationship,
)
from openhcs.core.runtime_semantics import (
    FieldSpec,
    MeasurementRowAxisField,
    MeasurementRowMappingCache,
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
    assert annotate_measurement_row_object({"area": 42.0}, "Cells") == {
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

    query = MeasurementTableAxisQuery(MeasurementRowAxisField.SLICE_INDEX, 1)
    projected = MeasurementTableAxisProjection(
        table,
        query.axis,
        query.value,
    ).apply()

    assert query.axis is MeasurementRowAxisField.SLICE_INDEX
    assert query.value == 1
    assert tuple(projected.rows) == (
        {"slice_index": 1, "object_label": 1, "area": 20.0},
    )


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

    projected = MeasurementTableAxisQuery(
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
        MeasurementTableAxisQuery(
            MeasurementRowAxisField.SLICE_INDEX,
            1,
        ).tables((table,))[0].rows
    ) == (
        {"slice_index": 1, "image_number": 2, "area": 20.0},
    )
    assert tuple(
        MeasurementTableAxisQuery(
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
