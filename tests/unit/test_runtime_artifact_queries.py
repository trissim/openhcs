from __future__ import annotations

from dataclasses import dataclass

from openhcs.core.artifacts import ArtifactKind, ArtifactOutputPlan
from openhcs.core.runtime_artifact_queries import (
    RuntimeArtifactQueryContext,
    annotate_measurement_row_object,
    measurement_row_mapping,
    measurement_values_for_feature,
    runtime_measurement_tables_for_object,
    runtime_relationship,
)
from openhcs.core.runtime_semantics import RelationshipSemantics
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
    row = MeasurementRow(object_name="Nuclei", object_label=1)

    assert measurement_row_mapping(row) == {
        "object_name": "Nuclei",
        "object_label": 1,
    }
    assert annotate_measurement_row_object({"area": 42.0}, "Cells") == {
        "area": 42.0,
        "object_name": "Cells",
    }


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


def _record_native(
    store: RuntimeValueStore,
    native_value: MeasurementTable | ObjectRelationship,
    kind: ArtifactKind,
) -> None:
    value = normalize_artifact_value(
        ArtifactOutputPlan(
            name=native_value.name,
            path=f"/memory/{native_value.name}.pkl",
            kind=kind,
        ),
        native_value,
        axis_id=AXIS_ID,
    )
    store.record(
        value,
        path=f"/memory/{native_value.name}.pkl",
        backend="memory",
    )
