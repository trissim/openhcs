from __future__ import annotations

from dataclasses import dataclass

import pytest

from openhcs.core.artifacts import ArtifactKind, ArtifactOutputPlan
from openhcs.core.runtime_artifact_queries import (
    RuntimeArtifactQueryContext,
    MeasurementTableObjectFeatureSemantics,
    MeasurementTableObjectFeatureSemanticsCache,
    annotate_measurement_row_object,
    matching_measurement_field,
    measurement_feature_candidates,
    ordered_measurement_feature_candidates,
    measurement_row_mapping,
    measurement_values_for_feature,
    runtime_measurement_tables_for_object,
    runtime_relationship,
)
from openhcs.core.runtime_semantics import (
    FieldSpec,
    MeasurementRowMappingCache,
    RelationshipSemantics,
)
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.runtime_values import (
    MeasurementTable,
    ObjectRelationship,
    normalize_artifact_value,
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
