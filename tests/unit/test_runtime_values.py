import pytest
import numpy as np
import pandas as pd

from openhcs.core.artifacts import ArtifactKind, ArtifactOutputPlan
from openhcs.core.runtime_values import (
    FieldSpec,
    ImagePayloadMetadata,
    ImageMetadataPayload,
    MeasurementTable,
    MeasurementScope,
    MeasurementSubject,
    MaskedImagePayload,
    NamedImage,
    ObjectLabelPayload,
    ObjectLabelSet,
    ObjectLabelRepresentation,
    RelationshipEndpoint,
    ObjectRelationship,
    RuntimeArrayPayload,
    RuntimeStoragePolicy,
    SpatialGrid,
    compose_image_payload_metadata,
    image_payload_data,
    image_payload_metadata,
    image_payload_with_context,
    normalize_image_payload_intensity,
    normalize_artifact_value,
)
from openhcs.core.runtime_semantics import ObjectLabelVariant, SpatialGridOrdering
from openhcs.processing.backends.lib_registry.unified_registry import (
    _aggregate_pure_2d_auxiliary_output,
)


class ArrayLike(RuntimeArrayPayload):
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


def test_spatial_grid_normalizes_to_mapping_runtime_value():
    output_plan = ArtifactOutputPlan(
        name="Grid",
        path="/memory/Grid.pkl",
        kind=ArtifactKind.SPATIAL_GRID,
    )
    grid = SpatialGrid(
        name="Grid",
        rows=30,
        columns=30,
        x_spacing=55.0,
        y_spacing=55.0,
        x_origin=27.0,
        y_origin=27.0,
    )

    value = normalize_artifact_value(output_plan, grid, axis_id="A01")

    assert value.kind is ArtifactKind.SPATIAL_GRID
    assert value.schema.kind is ArtifactKind.SPATIAL_GRID
    assert value.data["rows"] == 30
    assert value.data["x_location_of_lowest_x_spot"] == 27.0
    assert value.data["ordering"] == SpatialGridOrdering.BY_ROWS.value
    assert SpatialGrid.from_runtime_value(value) == grid


def test_spatial_grid_preserves_column_ordering():
    output_plan = ArtifactOutputPlan(
        name="Grid",
        path="/memory/Grid.pkl",
        kind=ArtifactKind.SPATIAL_GRID,
    )
    grid = SpatialGrid(
        name="Grid",
        rows=2,
        columns=3,
        x_spacing=55.0,
        y_spacing=55.0,
        x_origin=27.0,
        y_origin=27.0,
        ordering=SpatialGridOrdering.BY_COLUMNS,
    )

    value = normalize_artifact_value(output_plan, grid, axis_id="A01")

    assert value.data["ordering"] == SpatialGridOrdering.BY_COLUMNS.value
    assert SpatialGrid.from_runtime_value(value).ordering is SpatialGridOrdering.BY_COLUMNS


def test_normalize_artifact_value_accepts_object_label_arrays():
    output_plan = ArtifactOutputPlan(
        name="nuclei",
        path="/memory/nuclei.pkl",
        kind=ArtifactKind.OBJECT_LABELS,
    )

    value = normalize_artifact_value(output_plan, ArrayLike(), axis_id="A01")

    assert value.kind is ArtifactKind.OBJECT_LABELS


def test_normalize_artifact_value_accepts_registered_external_arrays():
    output_plan = ArtifactOutputPlan(
        name="nuclei",
        path="/memory/nuclei.pkl",
        kind=ArtifactKind.OBJECT_LABELS,
    )
    labels = np.zeros((3, 3), dtype=np.uint16)

    value = normalize_artifact_value(output_plan, labels, axis_id="A01")

    assert value.data is labels
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


def test_masked_image_payload_behaves_like_array_with_mask() -> None:
    image = np.arange(6, dtype=np.float32).reshape(2, 3)
    mask = np.array([[True, False, True], [True, True, False]])

    payload = MaskedImagePayload(data=image, mask=mask)

    assert payload.shape == image.shape
    assert payload.ndim == 2
    assert payload.dtype == image.dtype
    np.testing.assert_array_equal(np.asarray(payload), image)
    np.testing.assert_array_equal(payload.mask, mask)


def test_image_metadata_payload_carries_source_intensity_scale() -> None:
    image = np.zeros((2, 3), dtype=np.uint16)
    metadata = ImagePayloadMetadata.for_array(
        image,
        source_path="/plate/A01_s001_w1.png",
    )

    payload = ImageMetadataPayload(data=image, metadata=metadata)

    assert payload.shape == image.shape
    assert payload.metadata.intensity_scale == 65535.0
    assert payload.metadata.source_dtype == "uint16"
    assert payload.metadata.source_path == "/plate/A01_s001_w1.png"
    np.testing.assert_array_equal(np.asarray(payload), image)


def test_image_metadata_payload_exposes_array_methods() -> None:
    image = np.arange(6, dtype=np.float32).reshape(2, 3)
    payload = ImageMetadataPayload(
        data=image,
        metadata=ImagePayloadMetadata(source_dtype="float32"),
    )

    copied = payload.copy()

    np.testing.assert_array_equal(copied, image)
    assert copied is not image
    np.testing.assert_array_equal(payload.astype(np.float64), image.astype(np.float64))


def test_compose_image_payload_metadata_tracks_per_channel_sources() -> None:
    first = image_payload_with_context(
        np.zeros((2, 3), dtype=np.float32),
        metadata=ImagePayloadMetadata(intensity_scale=65535.0, source_dtype="uint16"),
    )
    second = image_payload_with_context(
        np.zeros((2, 3), dtype=np.float32),
        metadata=ImagePayloadMetadata(intensity_scale=255.0, source_dtype="uint8"),
    )

    metadata = compose_image_payload_metadata((first, second))

    assert metadata.channel_intensity_scales == (65535.0, 255.0)
    assert metadata.for_channel(0).intensity_scale == 65535.0
    assert metadata.for_channel(1).source_dtype == "uint8"


def test_compose_image_payload_metadata_tracks_unit_interval_proof() -> None:
    first = image_payload_with_context(
        np.zeros((2, 3), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            intensity_scale=65535.0,
            source_dtype="uint16",
            unit_interval_intensity_scale=65535,
        ),
    )
    second = image_payload_with_context(
        np.zeros((2, 3), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            intensity_scale=255.0,
            source_dtype="uint8",
            unit_interval_intensity_scale=255,
        ),
    )

    metadata = compose_image_payload_metadata((first, second))

    assert metadata.channel_unit_interval_intensity_scales == (65535, 255)
    assert metadata.for_channel(0).unit_interval_intensity_scale == 65535
    assert metadata.for_channel(1).unit_interval_intensity_scale == 255


def test_image_payload_metadata_tracks_spatial_crop_edges() -> None:
    metadata = ImagePayloadMetadata(source_dtype="float32").with_spatial_crop(
        input_shape_yx=(10, 12),
        output_shape_yx=(4, 5),
        offset_yx=(3, 2),
        physical_border_edges_yx=(False, False, False, False),
    )

    assert metadata.spatial_origin_yx == (3, 2)
    assert metadata.source_spatial_shape_yx == (10, 12)
    assert metadata.physical_border_edges_for_shape((4, 5)) == (
        False,
        False,
        False,
        False,
    )
    assert metadata.for_channel(0).spatial_origin_yx == (3, 2)


def test_compose_image_payload_metadata_preserves_shared_spatial_context() -> None:
    metadata = ImagePayloadMetadata(source_dtype="float32").with_spatial_crop(
        input_shape_yx=(10, 12),
        output_shape_yx=(4, 5),
        offset_yx=(0, 2),
        physical_border_edges_yx=(True, False, False, False),
    )
    first = image_payload_with_context(
        np.zeros((4, 5), dtype=np.float32),
        metadata=metadata,
    )
    second = image_payload_with_context(
        np.zeros((4, 5), dtype=np.float32),
        metadata=metadata,
    )

    composed = compose_image_payload_metadata((first, second))

    assert composed.spatial_origin_yx == (0, 2)
    assert composed.source_spatial_shape_yx == (10, 12)
    assert composed.physical_border_edges_for_shape((4, 5)) == (
        True,
        False,
        False,
        False,
    )


def test_aggregate_pure_2d_auxiliary_output_preserves_image_payload_metadata() -> None:
    first = image_payload_with_context(
        np.zeros((2, 3), dtype=np.float32),
        metadata=ImagePayloadMetadata(intensity_scale=65535.0, source_dtype="uint16"),
    )
    second = image_payload_with_context(
        np.ones((2, 3), dtype=np.float32),
        metadata=ImagePayloadMetadata(intensity_scale=255.0, source_dtype="uint8"),
    )

    stacked = _aggregate_pure_2d_auxiliary_output([first, second], "numpy")

    assert isinstance(stacked, ImageMetadataPayload)
    assert image_payload_data(stacked).shape == (2, 2, 3)
    assert image_payload_metadata(stacked).for_channel(0).intensity_scale == 65535.0
    assert image_payload_metadata(stacked).for_channel(1).source_dtype == "uint8"


def test_aggregate_pure_2d_auxiliary_output_preserves_stacked_object_labels() -> None:
    first = ObjectLabelPayload(
        labels=np.ones((2, 3, 4), dtype=np.int32),
        unedited_labels=np.ones((2, 3, 4), dtype=np.int32) * 2,
    )
    second = ObjectLabelPayload(
        labels=np.ones((2, 3, 4), dtype=np.int32) * 3,
        unedited_labels=np.ones((2, 3, 4), dtype=np.int32) * 4,
    )

    stacked = _aggregate_pure_2d_auxiliary_output([first, second], "numpy")

    assert isinstance(stacked, ObjectLabelPayload)
    assert stacked.labels.shape == (2, 2, 3, 4)
    assert stacked.unedited_labels is not None
    assert stacked.unedited_labels.shape == (2, 2, 3, 4)
    np.testing.assert_array_equal(stacked.labels[0], first.labels)
    np.testing.assert_array_equal(stacked.labels[1], second.labels)


def test_normalize_image_payload_intensity_uses_semantic_scale() -> None:
    image = np.array([[0, 4095]], dtype=np.uint16)
    payload = image_payload_with_context(
        image,
        metadata=ImagePayloadMetadata(intensity_scale=4095.0, source_dtype="uint16"),
    )

    normalized = normalize_image_payload_intensity(payload)

    assert image_payload_metadata(normalized).intensity_scale == 4095.0
    assert image_payload_metadata(normalized).unit_interval_intensity_scale == 4095
    assert image_payload_data(normalized).dtype == np.float32
    np.testing.assert_allclose(image_payload_data(normalized), [[0.0, 1.0]])


def test_normalize_image_payload_intensity_falls_back_to_dtype_scale() -> None:
    image = np.array([[0, 255]], dtype=np.uint8)

    normalized = normalize_image_payload_intensity(image)

    assert normalized.dtype == np.float32
    np.testing.assert_allclose(normalized, [[0.0, 1.0]])


def test_masked_image_payload_rejects_unaligned_mask_shape() -> None:
    image = np.zeros((2, 3), dtype=np.float32)
    mask = np.ones((4, 4), dtype=bool)

    with pytest.raises(ValueError, match="mask shape"):
        MaskedImagePayload(data=image, mask=mask)


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


def test_normalize_object_label_set_preserves_dense_label_variants():
    output_plan = ArtifactOutputPlan(
        name="Nuclei",
        path="/memory/Nuclei.pkl",
        kind=ArtifactKind.OBJECT_LABELS,
    )
    labels = np.array([[0, 1], [2, 0]], dtype=np.int32)
    unedited_labels = np.array([[3, 1], [2, 0]], dtype=np.int32)
    small_removed_labels = np.array([[0, 1], [2, 4]], dtype=np.int32)

    value = normalize_artifact_value(
        output_plan,
        ObjectLabelSet(
            name="Nuclei",
            labels=labels,
            unedited_labels=unedited_labels,
            small_removed_labels=small_removed_labels,
        ),
        axis_id="A01",
    )

    assert isinstance(value.data, ObjectLabelPayload)
    assert value.schema.label_variants == (
        ObjectLabelVariant.FINAL,
        ObjectLabelVariant.UNEDITED,
        ObjectLabelVariant.SMALL_REMOVED,
    )
    restored = ObjectLabelSet.from_runtime_value(value)
    np.testing.assert_array_equal(restored.labels_for_variant("final"), labels)
    np.testing.assert_array_equal(restored.labels_for_variant("unedited"), unedited_labels)
    np.testing.assert_array_equal(
        restored.labels_for_variant("small_removed"),
        small_removed_labels,
    )


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


def test_normalize_measurement_table_accepts_registered_columnar_rows():
    output_plan = ArtifactOutputPlan(
        name="NucleiMeasurements",
        path="/memory/NucleiMeasurements.pkl",
        kind=ArtifactKind.MEASUREMENTS,
    )
    rows = pd.DataFrame({"object_id": [1], "area": [12.0]})

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
    assert value.schema.relationship.source.name == "Cells"
    assert value.schema.relationship.target.name == "Nuclei"


def test_normalize_object_relationship_preserves_slice_metadata():
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
            source_ids=(10, 11),
            target_ids=(1, 2),
            relationship_type="parent_child",
            slice_indices=(0, 1),
            slice_count=2,
        ),
        axis_id="A01",
    )

    relationship = ObjectRelationship.from_runtime_value(value)

    assert value.data["slice_index"] == (0, 1)
    assert value.data["slice_count"] == 2
    assert relationship.slice_indices == (0, 1)
    assert relationship.slice_count == 2


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
