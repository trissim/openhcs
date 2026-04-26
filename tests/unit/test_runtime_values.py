import pytest

from openhcs.core.artifacts import ArtifactKind, ArtifactOutputPlan
from openhcs.core.runtime_values import (
    FieldSpec,
    MeasurementTable,
    MeasurementScope,
    MeasurementSubject,
    NamedImage,
    ObjectLabelSet,
    ObjectLabelRepresentation,
    RelationshipEndpoint,
    ObjectRelationship,
    RuntimeStoragePolicy,
    normalize_artifact_value,
)


class ArrayLike:
    shape = (3, 3)


def test_normalize_artifact_value_builds_key_schema_and_storage_policy():
    output_plan = ArtifactOutputPlan(
        name="measurements",
        path="/memory/measurements.pkl",
        kind=ArtifactKind.MEASUREMENTS,
        group_keys=("DAPI",),
    )

    value = normalize_artifact_value(
        output_plan,
        [{"object_id": 1, "area": 12.0}],
        axis_id="A01",
    )

    assert value.name == "measurements"
    assert value.kind is ArtifactKind.MEASUREMENTS
    assert value.key.scope.axis_id == "A01"
    assert value.key.scope.group_key == "DAPI"
    assert value.schema.kind is ArtifactKind.MEASUREMENTS
    assert value.storage == RuntimeStoragePolicy(
        backend="memory",
        path="/memory/measurements.pkl",
        materialize=False,
    )


def test_normalize_artifact_value_rejects_metadata_payload_mismatch():
    output_plan = ArtifactOutputPlan(
        name="metadata",
        path="/memory/metadata.pkl",
        kind=ArtifactKind.METADATA,
    )

    with pytest.raises(TypeError, match="expected metadata mapping"):
        normalize_artifact_value(output_plan, ["not", "metadata"], axis_id="A01")


def test_normalize_artifact_value_rejects_object_label_payload_mismatch():
    output_plan = ArtifactOutputPlan(
        name="nuclei",
        path="/memory/nuclei.pkl",
        kind=ArtifactKind.OBJECT_LABELS,
    )

    with pytest.raises(TypeError, match="expected object_labels payload"):
        normalize_artifact_value(output_plan, {"not": "labels"}, axis_id="A01")


def test_normalize_artifact_value_accepts_object_label_arrays():
    output_plan = ArtifactOutputPlan(
        name="nuclei",
        path="/memory/nuclei.pkl",
        kind=ArtifactKind.OBJECT_LABELS,
    )

    value = normalize_artifact_value(output_plan, ArrayLike(), axis_id="A01")

    assert value.kind is ArtifactKind.OBJECT_LABELS


def test_normalize_named_image_preserves_raw_payload_and_schema():
    output_plan = ArtifactOutputPlan(
        name="DNA",
        path="/memory/DNA.pkl",
        kind=ArtifactKind.IMAGE,
    )
    image = ArrayLike()

    value = normalize_artifact_value(
        output_plan,
        NamedImage(
            name="DNA",
            data=image,
            dimensions=("z", "y", "x"),
            source_image_name="raw_DNA",
        ),
        axis_id="A01",
    )

    assert value.data is image
    assert value.schema.kind is ArtifactKind.IMAGE
    assert value.schema.dimensions == ("z", "y", "x")
    assert value.schema.source_image_name == "raw_DNA"


def test_normalize_object_label_set_adds_object_schema():
    output_plan = ArtifactOutputPlan(
        name="Nuclei",
        path="/memory/Nuclei.pkl",
        kind=ArtifactKind.OBJECT_LABELS,
    )
    labels = ArrayLike()

    value = normalize_artifact_value(
        output_plan,
        ObjectLabelSet(
            name="Nuclei",
            labels=labels,
            source_image_name="DNA",
            dimensions=("y", "x"),
        ),
        axis_id="A01",
    )

    assert value.data is labels
    assert value.schema.object_name == "Nuclei"
    assert value.schema.source_image_name == "DNA"
    assert value.schema.dimensions == ("y", "x")
    assert value.schema.label_representation is ObjectLabelRepresentation.DENSE_LABELS


def test_normalize_object_label_set_accepts_sparse_ijv_representation():
    output_plan = ArtifactOutputPlan(
        name="Nuclei",
        path="/memory/Nuclei.pkl",
        kind=ArtifactKind.OBJECT_LABELS,
    )
    labels = [{"i": 0, "j": 1, "label": 7}]

    value = normalize_artifact_value(
        output_plan,
        ObjectLabelSet(
            name="Nuclei",
            labels=labels,
            representation=ObjectLabelRepresentation.SPARSE_IJV,
        ),
        axis_id="A01",
    )

    assert value.data is labels
    assert value.schema.label_representation is ObjectLabelRepresentation.SPARSE_IJV


def test_normalize_measurement_table_infers_fields_and_object_schema():
    output_plan = ArtifactOutputPlan(
        name="NucleiMeasurements",
        path="/memory/NucleiMeasurements.pkl",
        kind=ArtifactKind.MEASUREMENTS,
    )
    rows = [{"object_id": 1, "area": 12.0}]

    value = normalize_artifact_value(
        output_plan,
        MeasurementTable(
            name="NucleiMeasurements",
            rows=rows,
            object_name="Nuclei",
            object_id_field="object_id",
        ),
        axis_id="A01",
    )

    assert value.data is rows
    assert value.schema.object_name == "Nuclei"
    assert value.schema.object_id_field == "object_id"
    assert value.schema.measurement_subject == MeasurementSubject(
        MeasurementScope.OBJECT,
        "Nuclei",
        "object_id",
    )
    assert value.schema.fields == (FieldSpec("object_id"), FieldSpec("area"))


def test_normalize_measurement_table_accepts_generic_subject():
    output_plan = ArtifactOutputPlan(
        name="ImageMeasurements",
        path="/memory/ImageMeasurements.pkl",
        kind=ArtifactKind.MEASUREMENTS,
    )
    rows = [{"mean_intensity": 12.0}]

    value = normalize_artifact_value(
        output_plan,
        MeasurementTable(
            name="ImageMeasurements",
            rows=rows,
            subject=MeasurementSubject(MeasurementScope.IMAGE, "DNA"),
        ),
        axis_id="A01",
    )

    assert value.schema.measurement_subject == MeasurementSubject(
        MeasurementScope.IMAGE,
        "DNA",
    )
    assert value.schema.source_image_name == "DNA"
    assert value.schema.object_name is None


def test_object_measurement_subject_allows_implicit_object_ids():
    subject = MeasurementSubject(MeasurementScope.OBJECT, "Nuclei")

    assert subject.id_field is None


def test_normalize_object_relationship_materializes_table_columns():
    output_plan = ArtifactOutputPlan(
        name="ParentChild",
        path="/memory/ParentChild.pkl",
        kind=ArtifactKind.RELATIONSHIPS,
    )

    value = normalize_artifact_value(
        output_plan,
        ObjectRelationship(
            name="ParentChild",
            source=RelationshipEndpoint(
                "Cells",
                role="parent",
                id_field="parent_id",
            ),
            target=RelationshipEndpoint(
                "Nuclei",
                role="child",
                id_field="child_id",
            ),
            source_ids=[10, 11],
            target_ids=[1, 2],
            relationship_type="parent_child",
        ),
        axis_id="A01",
    )

    assert value.data == {
        "relationship_type": "parent_child",
        "source_role": "parent",
        "target_role": "child",
        "source_object": "Cells",
        "target_object": "Nuclei",
        "parent_id": [10, 11],
        "child_id": [1, 2],
    }
    assert value.schema.relationship is not None
    assert value.schema.relationship.source_name == "Cells"
    assert value.schema.relationship.target_name == "Nuclei"
    assert value.schema.parent_object_name == "Cells"
    assert value.schema.child_object_name == "Nuclei"


def test_native_runtime_value_name_must_match_output_plan():
    output_plan = ArtifactOutputPlan(
        name="Nuclei",
        path="/memory/Nuclei.pkl",
        kind=ArtifactKind.OBJECT_LABELS,
    )

    with pytest.raises(ValueError, match="does not match planned artifact"):
        normalize_artifact_value(
            output_plan,
            ObjectLabelSet(name="Cells", labels=ArrayLike()),
            axis_id="A01",
        )


def test_object_relationship_rejects_mismatched_id_lengths():
    with pytest.raises(ValueError, match="equal length"):
        ObjectRelationship(
            name="ParentChild",
            source=RelationshipEndpoint(
                "Cells",
                role="parent",
                id_field="parent_id",
            ),
            target=RelationshipEndpoint(
                "Nuclei",
                role="child",
                id_field="child_id",
            ),
            source_ids=[1],
            target_ids=[1, 2],
        )
