import pytest
import numpy as np
import pandas as pd

from openhcs.core.artifacts import ArtifactKey, ArtifactKind, ArtifactOutputPlan, ArtifactScope
from openhcs.core.runtime_invocation import RuntimeSliceAlignedValues
from openhcs.core.runtime_values import (
    FieldSpec,
    DerivedImagePayloadContext,
    DenseObjectLabelSliceStack,
    ImagePayloadChannelProjection,
    ImagePayloadMetadata,
    ImageMetadataPayload,
    MeasurementTable,
    MeasurementScope,
    MeasurementSubject,
    MaskedImagePayload,
    NamedImage,
    ObjectLabelPayload,
    ObjectLabelPayloadBuilderStrategy,
    ObjectLabelDomainScope,
    ObjectLabelSet,
    ObjectLabelDenseDataStrategy,
    ObjectLabelMeasurementPayloadStrategy,
    ObjectLabelRepresentation,
    ObjectLabelPure2DSliceAggregator,
    ObjectLabelVariantCompatibilityStrategy,
    RuntimePayloadDataStrategy,
    RelationshipEndpoint,
    ObjectRelationship,
    RuntimeArrayPayload,
    RuntimeStoragePolicy,
    RuntimeValue,
    RuntimeValueSchema,
    SparseIJVLabelRows,
    SpatialGrid,
    SingletonObjectLabelStackCollapseStrategy,
    compose_image_payload_metadata,
    image_payload_data,
    image_payload_metadata,
    image_payload_with_context,
    normalize_image_payload_intensity,
    normalize_artifact_value,
    object_label_dense_array,
    object_label_payload_from_source_image,
    object_label_payload_with_dense_labels,
    object_label_set_from_source_image,
    object_label_set_with_replacement_labels,
)
from openhcs.core.runtime_semantics import (
    ExplicitObjectLabelDomainDeclaration,
    ObjectFeatureArrayDomain,
    ObjectFeatureArrayDomainStrategy,
    ObjectFeatureMissingValue,
    ObjectFeatureMissingValueStrategy,
    ObjectLabelDomain,
    ObjectLabelDomainMetadata,
    ObjectLabelDomainMetadataStrategy,
    ObjectFeatureValueTable,
    ObjectLabelVariant,
    ObjectShapeMeasurementFeature,
    RuntimePlaneAxis,
    ShapeObjectFeatureValueTable,
    SpatialGridOrdering,
)
from openhcs.processing.backends.lib_registry.unified_registry import (
    Pure2DAuxiliaryOutputAggregator,
)


class ArrayLike(RuntimeArrayPayload):
    shape = (3, 3)

    def array_payload_data(self):
        return np.zeros(self.shape, dtype=np.int32)

    def with_data(self, data):
        return data


class NominalObjectLabelDomainCarrier(ObjectLabelDomainMetadata):
    def __init__(self, domain: ObjectLabelDomain) -> None:
        self._domain = domain

    def object_label_domain(self) -> ObjectLabelDomain:
        return self._domain


class SpecificNominalObjectLabelDomainCarrier(NominalObjectLabelDomainCarrier):
    pass


class SpecificNominalObjectLabelDomainStrategy(ObjectLabelDomainMetadataStrategy):
    value_type = SpecificNominalObjectLabelDomainCarrier

    def object_label_domain(self, value: object) -> ObjectLabelDomain:
        return ObjectLabelDomain(declared_object_ids=(8,))


class StructuralObjectLabelDomainLookalike:
    def object_label_domain(self) -> ObjectLabelDomain:
        return ObjectLabelDomain(declared_object_count=99)


def test_object_label_dense_data_uses_nominal_payload_registry() -> None:
    labels = np.array([[0, 1], [2, 0]], dtype=np.int16)
    payload = ObjectLabelPayload(labels=labels)
    label_set = ObjectLabelSet(name="Cells", labels=labels)

    assert issubclass(ObjectLabelDenseDataStrategy, RuntimePayloadDataStrategy)
    assert ObjectLabelDenseDataStrategy.for_payload(payload).data(payload) is labels
    assert ObjectLabelDenseDataStrategy.for_payload(label_set).data(label_set) is labels
    assert ObjectLabelDenseDataStrategy.for_payload(labels).data(labels) is labels
    assert object_label_dense_array(payload, dtype=np.int32).dtype == np.int32


def test_sparse_ijv_object_label_dense_data_uses_source_shape() -> None:
    sparse_rows = SparseIJVLabelRows(
        np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int32)
    )
    label_set = ObjectLabelSet(
        name="Cells",
        labels=sparse_rows,
        representation=ObjectLabelRepresentation.SPARSE_IJV,
        source_spatial_shape_yx=(5, 6),
    )

    dense = object_label_dense_array(label_set, dtype=np.int32)

    assert dense.shape == (5, 6)
    assert dense[0, 1] == 2
    assert dense[2, 3] == 4


def test_sparse_ijv_object_label_dense_data_preserves_runtime_slices() -> None:
    sparse_rows = SparseIJVLabelRows(
        np.array([[0, 1, 2, 3], [2, 3, 4, 5]], dtype=np.int32)
    )
    payload = ObjectLabelPayload(
        labels=sparse_rows,
        source_spatial_shape_yx=(6, 7),
    )

    dense = object_label_dense_array(payload, dtype=np.int32)

    assert dense.shape == (3, 6, 7)
    assert dense[0, 1, 2] == 3
    assert dense[2, 3, 4] == 5


def test_object_label_domain_preservation_uses_nominal_metadata_contract() -> None:
    labels = np.array([[0, 1], [2, 0]], dtype=np.int16)
    source = NominalObjectLabelDomainCarrier(
        ObjectLabelDomain(declared_object_ids=(4, 7))
    )

    rebuilt = object_label_payload_with_dense_labels(source, labels)

    assert rebuilt.declared_object_ids == (4, 7)


def test_object_label_domain_preservation_uses_mro_specific_provider() -> None:
    labels = np.array([[0, 1], [2, 0]], dtype=np.int16)
    source = SpecificNominalObjectLabelDomainCarrier(
        ObjectLabelDomain(declared_object_ids=(4, 7))
    )

    rebuilt = object_label_payload_with_dense_labels(source, labels)

    assert rebuilt.declared_object_ids == (8,)


def test_object_label_domain_preservation_rejects_structural_lookalikes() -> None:
    labels = np.array([[0, 1], [2, 0]], dtype=np.int16)

    rebuilt = object_label_payload_with_dense_labels(
        StructuralObjectLabelDomainLookalike(),
        labels,
    )

    assert rebuilt.declared_object_count is None
    assert rebuilt.declared_object_ids == ()


def test_object_label_pure_2d_aggregator_preserves_dense_payload_domains() -> None:
    first = ObjectLabelPayload(
        labels=np.asarray([[0, 1], [0, 0]], dtype=np.int32),
        declared_object_ids=(1,),
        domain_scope=ObjectLabelDomainScope.PAYLOAD,
    )
    second = ObjectLabelPayload(
        labels=np.asarray([[0, 2], [0, 0]], dtype=np.int32),
        declared_object_ids=(2,),
        domain_scope=ObjectLabelDomainScope.PAYLOAD,
    )

    aggregated = ObjectLabelPure2DSliceAggregator.aggregate(
        (first, second),
        "numpy",
    )

    assert isinstance(aggregated, ObjectLabelPayload)
    assert aggregated.domain_scope is ObjectLabelDomainScope.PLANE
    assert aggregated.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    np.testing.assert_array_equal(
        aggregated.labels,
        np.asarray(
            [
                [[0, 1], [0, 0]],
                [[0, 2], [0, 0]],
            ],
            dtype=np.int32,
        ),
    )
    assert aggregated.declared_object_id_domains == ((1,), (2,))


def test_object_label_pure_2d_aggregator_preserves_sparse_ijv_sets() -> None:
    first = ObjectLabelSet(
        name="Cells",
        labels=SparseIJVLabelRows.from_dense_labels(
            np.asarray([[0, 1], [0, 0]], dtype=np.int32)
        ),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
    )
    second = ObjectLabelSet(
        name="Cells",
        labels=SparseIJVLabelRows.from_dense_labels(
            np.asarray([[0, 2], [0, 0]], dtype=np.int32)
        ),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
    )

    aggregated = ObjectLabelPure2DSliceAggregator.aggregate(
        (first, second),
        "numpy",
    )

    assert isinstance(aggregated, ObjectLabelSet)
    assert aggregated.representation is ObjectLabelRepresentation.SPARSE_IJV
    assert isinstance(aggregated.labels, SparseIJVLabelRows)
    assert aggregated.labels.has_slice_index
    assert aggregated.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE


def test_shape_object_feature_table_uses_registered_nominal_contract() -> None:
    table = ShapeObjectFeatureValueTable.from_feature_arrays(
        {
            ObjectShapeMeasurementFeature.AREA.value: np.asarray([10.0]),
            ObjectShapeMeasurementFeature.MAXIMUM_RADIUS.value: np.asarray([2.0]),
        },
        measured_object_ids=(2,),
        object_domain=(1, 2),
    )

    assert issubclass(ShapeObjectFeatureValueTable, ObjectFeatureValueTable)
    assert ShapeObjectFeatureValueTable in ObjectFeatureValueTable.registered_strategy_types()

    rows = table.rows()
    assert rows[0]["object_label"] == 1
    assert np.isnan(rows[0][ObjectShapeMeasurementFeature.AREA.value])
    assert rows[0][ObjectShapeMeasurementFeature.MAXIMUM_RADIUS.value] == 0.0
    assert rows[0][ObjectShapeMeasurementFeature.CENTER_Z.value] == 0.0
    assert rows[1]["object_label"] == 2
    assert rows[1][ObjectShapeMeasurementFeature.AREA.value] == 10.0


def test_shape_object_feature_table_rejects_undeclared_dense_feature_domain() -> None:
    table = ShapeObjectFeatureValueTable.from_feature_arrays(
        {
            ObjectShapeMeasurementFeature.ZERNIKE.value: np.asarray(
                [0.1, 0.2, 0.3],
            ),
        },
        measured_object_ids=(1, 3),
        object_domain=(1, 2, 3),
    )

    with pytest.raises(ValueError, match="feature-array domain"):
        table.rows()


def test_shape_descriptor_row_ordinal_domain_is_registered_nominally() -> None:
    table = ShapeObjectFeatureValueTable.from_feature_arrays(
        {
            ObjectShapeMeasurementFeature.MAX_FERET_DIAMETER.value: np.asarray(
                [0.0, 20.0],
            ),
            "Zernike_0_0": np.asarray([0.1, 0.2]),
        },
        measured_object_ids=(1, 3),
        object_domain=(1, 2, 3),
    )

    assert (
        ObjectFeatureArrayDomainStrategy.for_enum_member(
            ObjectFeatureArrayDomain.ROW_ORDINAL
        ).domain
        is ObjectFeatureArrayDomain.ROW_ORDINAL
    )
    assert (
        ObjectFeatureMissingValueStrategy.for_enum_member(
            ObjectFeatureMissingValue.ZERO
        ).missing_value
        is ObjectFeatureMissingValue.ZERO
    )

    rows = table.rows()
    assert rows[1][ObjectShapeMeasurementFeature.MAX_FERET_DIAMETER.value] == 20.0
    assert rows[2][ObjectShapeMeasurementFeature.MAX_FERET_DIAMETER.value] == 0.0
    assert np.isnan(rows[2]["Zernike_0_0"])


def test_object_label_payload_builder_uses_nominal_payload_registry() -> None:
    source_labels = np.array([[0, 1], [2, 0]], dtype=np.int16)
    transformed_labels = np.array([[0, 2], [1, 0]], dtype=np.float32)
    payload = ObjectLabelPayload(
        labels=source_labels,
        declared_object_ids=(1, 2),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        spatial_origin_yx=(3, 5),
        source_spatial_shape_yx=(20, 30),
    )

    rebuilt = object_label_payload_with_dense_labels(
        payload,
        transformed_labels,
        domain_declaration=ExplicitObjectLabelDomainDeclaration(
            ObjectLabelDomain(
                declared_object_count=1,
                declared_object_ids=(2,),
            )
        ),
    )

    assert isinstance(
        ObjectLabelPayloadBuilderStrategy.for_source(payload),
        ObjectLabelPayloadBuilderStrategy,
    )
    assert rebuilt.labels is transformed_labels
    assert rebuilt.declared_object_count == 1
    assert rebuilt.declared_object_ids == (2,)
    assert rebuilt.plane_axis is RuntimePlaneAxis.SOURCE_BINDING
    assert rebuilt.spatial_origin_yx == (3, 5)
    assert rebuilt.source_spatial_shape_yx == (20, 30)


def test_object_label_payload_from_source_image_uses_image_metadata() -> None:
    image = ImageMetadataPayload(
        data=np.zeros((4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            spatial_origin_yx=(2, 3),
            source_spatial_shape_yx=(10, 12),
        ),
    )
    labels = np.array([[0, 1], [2, 0]], dtype=np.int32)

    payload = object_label_payload_from_source_image(
        image,
        labels,
        declared_object_count=2,
    )

    assert payload.labels is labels
    assert payload.declared_object_count == 2
    assert payload.spatial_origin_yx == (2, 3)
    assert payload.source_spatial_shape_yx == (10, 12)


def test_object_label_set_from_source_image_uses_image_metadata() -> None:
    image = ImageMetadataPayload(
        data=np.zeros((4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            spatial_origin_yx=(2, 3),
            source_spatial_shape_yx=(10, 12),
        ),
    )
    sparse_rows = SparseIJVLabelRows(
        np.array([[0, 0, 1], [1, 1, 2]], dtype=np.int32)
    )

    label_set = object_label_set_from_source_image(
        image,
        name="OverlappingWorms",
        labels=sparse_rows,
        representation=ObjectLabelRepresentation.SPARSE_IJV,
        declared_object_count=2,
    )

    assert label_set.labels is sparse_rows
    assert label_set.representation is ObjectLabelRepresentation.SPARSE_IJV
    assert label_set.declared_object_count == 2
    assert label_set.spatial_origin_yx == (2, 3)
    assert label_set.source_spatial_shape_yx == (10, 12)


def test_object_label_set_replacement_preserves_sparse_ijv_representation() -> None:
    source_rows = SparseIJVLabelRows(
        np.array([[0, 0, 1], [1, 1, 2]], dtype=np.int32)
    )
    replacement_rows = SparseIJVLabelRows(
        np.array([[0, 1, 1]], dtype=np.int32)
    )
    source = ObjectLabelSet(
        name="OverlappingWorms",
        labels=source_rows,
        representation=ObjectLabelRepresentation.SPARSE_IJV,
    )
    replacement = ObjectLabelSet(
        name="OverlappingWorms",
        labels=replacement_rows,
        representation=ObjectLabelRepresentation.SPARSE_IJV,
    )

    rebuilt = object_label_set_with_replacement_labels(source, replacement)

    assert rebuilt.representation is ObjectLabelRepresentation.SPARSE_IJV
    assert rebuilt.labels is replacement_rows


def test_sparse_ijv_object_label_replacement_converts_dense_labels() -> None:
    source = ObjectLabelSet(
        name="OverlappingWorms",
        labels=SparseIJVLabelRows(np.array([[0, 0, 1]], dtype=np.int32)),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
    )
    dense_replacement = np.array([[0, 2], [3, 0]], dtype=np.int32)

    rebuilt = object_label_set_with_replacement_labels(source, dense_replacement)

    assert rebuilt.representation is ObjectLabelRepresentation.SPARSE_IJV
    assert isinstance(rebuilt.labels, SparseIJVLabelRows)
    np.testing.assert_array_equal(
        rebuilt.labels.as_array(),
        np.array([[0, 1, 2], [1, 0, 3]], dtype=np.int32),
    )


def test_object_label_payload_with_measurement_labels_preserves_domain_and_variants() -> None:
    labels = np.zeros((1, 2, 2), dtype=np.int32)
    unedited = np.ones_like(labels)
    small_removed = np.full_like(labels, 2)
    payload = ObjectLabelPayload(
        labels=labels,
        unedited_labels=unedited,
        small_removed_labels=small_removed,
        declared_object_count=2,
        declared_object_ids=(1, 2),
        spatial_origin_yx=(4, 5),
        source_spatial_shape_yx=(10, 11),
    )
    selected = labels[0]

    rebuilt = ObjectLabelMeasurementPayloadStrategy.for_source(payload).with_labels(
        payload,
        selected,
    )

    assert isinstance(
        ObjectLabelMeasurementPayloadStrategy.for_source(payload),
        ObjectLabelMeasurementPayloadStrategy,
    )
    assert isinstance(rebuilt, ObjectLabelPayload)
    assert rebuilt.labels is selected
    assert rebuilt.unedited_labels is None
    assert rebuilt.small_removed_labels is None
    assert rebuilt.declared_object_count == 2
    assert rebuilt.declared_object_ids == (1, 2)
    assert rebuilt.spatial_origin_yx == (4, 5)
    assert rebuilt.source_spatial_shape_yx == (10, 11)


def test_object_label_variant_compatibility_uses_nominal_registry() -> None:
    variant = np.ones((1, 2, 2), dtype=np.int32)
    matching_labels = np.zeros((1, 2, 2), dtype=np.int32)
    selected_labels = matching_labels[0]

    assert isinstance(
        ObjectLabelVariantCompatibilityStrategy.for_variant(variant),
        ObjectLabelVariantCompatibilityStrategy,
    )
    assert (
        ObjectLabelVariantCompatibilityStrategy.for_variant(variant).matching_labels(
            variant,
            matching_labels,
        )
        is variant
    )
    assert (
        ObjectLabelVariantCompatibilityStrategy.for_variant(variant).matching_labels(
            variant,
            selected_labels,
        )
        is None
    )


def test_singleton_object_label_stack_collapse_uses_nominal_registry() -> None:
    labels = np.arange(4, dtype=np.int32).reshape(1, 2, 2)
    payload = ObjectLabelPayload(
        labels=labels,
        unedited_labels=labels.copy(),
        small_removed_labels=labels.copy(),
        spatial_origin_yx=(1, 2),
        source_spatial_shape_yx=(5, 6),
    )

    collapsed_array = SingletonObjectLabelStackCollapseStrategy.for_labels(labels).collapse(
        labels
    )
    collapsed_payload = SingletonObjectLabelStackCollapseStrategy.for_labels(
        payload
    ).collapse(payload)

    assert isinstance(
        SingletonObjectLabelStackCollapseStrategy.for_labels(labels),
        SingletonObjectLabelStackCollapseStrategy,
    )
    np.testing.assert_array_equal(collapsed_array, labels[0])
    assert isinstance(collapsed_payload, ObjectLabelPayload)
    np.testing.assert_array_equal(collapsed_payload.labels, labels[0])
    np.testing.assert_array_equal(collapsed_payload.unedited_labels, labels[0])
    np.testing.assert_array_equal(collapsed_payload.small_removed_labels, labels[0])
    assert collapsed_payload.spatial_origin_yx == (1, 2)
    assert collapsed_payload.source_spatial_shape_yx == (5, 6)


def test_dense_object_label_slice_stack_projects_payload_labels() -> None:
    labels = np.array([[0, 1], [2, 0]], dtype=np.int16)
    payload = ObjectLabelPayload(labels=labels)

    stack = DenseObjectLabelSliceStack.from_payload(
        payload,
        slice_count=3,
        dtype=np.int32,
    )

    assert stack is not None
    assert stack.labels.shape == (3, 2, 2)
    assert stack.labels.dtype == np.int32
    np.testing.assert_array_equal(stack.slice(2), labels)


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


def test_normalize_artifact_value_aggregates_slice_aligned_object_label_domains():
    output_plan = ArtifactOutputPlan(
        name="GridObjects",
        path="/memory/GridObjects.pkl",
        kind=ArtifactKind.OBJECT_LABELS,
    )
    first = ObjectLabelPayload(
        labels=np.array([[0, 1], [0, 3]], dtype=np.int32),
        declared_object_count=4,
    )
    second = ObjectLabelPayload(
        labels=np.array([[0, 2], [4, 0]], dtype=np.int32),
        declared_object_count=4,
    )

    value = normalize_artifact_value(
        output_plan,
        RuntimeSliceAlignedValues((first, second)),
        axis_id="A01",
    )
    payload = value.data

    assert isinstance(payload, ObjectLabelPayload)
    assert value.schema.slice_aligned is False
    assert payload.declared_object_count == 4
    assert payload.domain_scope is ObjectLabelDomainScope.PLANE
    np.testing.assert_array_equal(
        payload.labels,
        np.array(
            [
                [[0, 1], [0, 3]],
                [[0, 2], [4, 0]],
            ],
            dtype=np.int32,
        ),
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


def test_object_label_payload_validator_accepts_nominal_slice_aggregate():
    output_plan = ArtifactOutputPlan(
        name="nuclei",
        path="/memory/nuclei.pkl",
        kind=ArtifactKind.OBJECT_LABELS,
    )
    payload = ObjectLabelPayload(
        labels=np.array(
            [
                [[0, 1], [0, 2]],
                [[3, 0], [4, 0]],
            ],
            dtype=np.int32,
        ),
        declared_object_id_domains=((1, 2), (3, 4)),
        domain_scope=ObjectLabelDomainScope.PLANE,
    )
    value = RuntimeValue(
        key=ArtifactKey(
            name="nuclei",
            kind=ArtifactKind.OBJECT_LABELS,
            scope=ArtifactScope(axis_id="A01"),
        ),
        data=payload,
        schema=RuntimeValueSchema(
            kind=ArtifactKind.OBJECT_LABELS,
            slice_aligned=True,
            object_name="nuclei",
        ),
    )

    normalized = normalize_artifact_value(output_plan, value, axis_id="A01")

    assert normalized.data is payload


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


def test_slice_aligned_spatial_grid_normalizes_to_validated_mapping_sequence():
    output_plan = ArtifactOutputPlan(
        name="Grid",
        path="/memory/Grid.pkl",
        kind=ArtifactKind.SPATIAL_GRID,
    )
    grids = RuntimeSliceAlignedValues(
        slices=(
            SpatialGrid(
                name="Grid",
                rows=2,
                columns=2,
                x_spacing=8.0,
                y_spacing=8.0,
                x_origin=1.0,
                y_origin=4.0,
            ),
            SpatialGrid(
                name="Grid",
                rows=2,
                columns=2,
                x_spacing=8.0,
                y_spacing=8.0,
                x_origin=2.0,
                y_origin=4.0,
            ),
        )
    )

    value = normalize_artifact_value(output_plan, grids, axis_id="A01")

    assert value.kind is ArtifactKind.SPATIAL_GRID
    assert value.schema.slice_aligned is True
    assert [grid["x_origin"] for grid in value.data] == [1.0, 2.0]


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


def test_derived_image_payload_context_projects_bundle_mask_to_single_output() -> None:
    image = np.zeros((3, 4, 5), dtype=np.float32)
    mask = np.ones((2, 3, 4, 5), dtype=bool)
    mask[1, :, 0, 0] = False
    source = MaskedImagePayload(data=np.stack((image, image)), mask=mask)

    result = DerivedImagePayloadContext(source, image).payload()

    assert isinstance(result, MaskedImagePayload)
    np.testing.assert_array_equal(result.mask, np.all(mask, axis=0))


def test_image_payload_channel_projection_preserves_channel_mask_and_metadata() -> None:
    data = np.zeros((2, 3, 4), dtype=np.float32)
    mask = np.ones_like(data, dtype=bool)
    mask[1, 0, 0] = False
    payload = MaskedImagePayload(
        data=data,
        mask=mask,
        metadata=ImagePayloadMetadata(
            channel_intensity_scales=(255.0, 65535.0),
            channel_source_dtypes=("uint8", "uint16"),
        ),
    )

    result = ImagePayloadChannelProjection.from_channel(payload, data, 1).payload()

    assert isinstance(result, MaskedImagePayload)
    assert result.data.shape == (1, 3, 4)
    np.testing.assert_array_equal(result.mask, mask[1:2])
    assert result.metadata.intensity_scale == 65535.0
    assert result.metadata.source_dtype == "uint16"


def test_masked_image_payload_accepts_grayscale_volume_stack_mask_domains() -> None:
    data = np.zeros((1, 3, 4, 5), dtype=np.float32)

    for mask_shape in ((3, 4, 5), (1, 4, 5), (4, 5)):
        payload = MaskedImagePayload(
            data=data,
            mask=np.ones(mask_shape, dtype=bool),
        )

        assert payload.mask.shape == mask_shape


def test_masked_image_payload_accepts_color_stack_mask_domains() -> None:
    data = np.zeros((2, 4, 5, 3), dtype=np.float32)

    for mask_shape in ((2, 4, 5), (4, 5)):
        payload = MaskedImagePayload(
            data=data,
            mask=np.ones(mask_shape, dtype=bool),
        )

        assert payload.mask.shape == mask_shape


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


def test_image_metadata_payload_supports_nominal_array_comparison() -> None:
    image = np.arange(6, dtype=np.float32).reshape(2, 3)
    payload = ImageMetadataPayload(
        data=image,
        metadata=ImagePayloadMetadata(source_dtype="float32"),
    )

    np.testing.assert_array_equal(payload > 2, image > 2)


def test_image_metadata_payload_ufunc_preserves_context_for_numeric_results() -> None:
    image = np.arange(6, dtype=np.float32).reshape(2, 3)
    metadata = ImagePayloadMetadata(source_dtype="float32")
    payload = ImageMetadataPayload(data=image, metadata=metadata)

    result = np.add(payload, 1.0)

    assert isinstance(result, ImageMetadataPayload)
    assert result.metadata == metadata
    np.testing.assert_array_equal(result.data, image + 1.0)


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


def test_pure_2d_auxiliary_aggregator_preserves_image_payload_metadata() -> None:
    first = image_payload_with_context(
        np.zeros((2, 3), dtype=np.float32),
        metadata=ImagePayloadMetadata(intensity_scale=65535.0, source_dtype="uint16"),
    )
    second = image_payload_with_context(
        np.ones((2, 3), dtype=np.float32),
        metadata=ImagePayloadMetadata(intensity_scale=255.0, source_dtype="uint8"),
    )

    stacked = Pure2DAuxiliaryOutputAggregator.aggregate([first, second], "numpy")

    assert isinstance(stacked, ImageMetadataPayload)
    assert image_payload_data(stacked).shape == (2, 2, 3)
    assert image_payload_metadata(stacked).for_channel(0).intensity_scale == 65535.0
    assert image_payload_metadata(stacked).for_channel(1).source_dtype == "uint8"


def test_pure_2d_auxiliary_aggregator_preserves_stacked_object_labels() -> None:
    first = ObjectLabelPayload(
        labels=np.ones((2, 3, 4), dtype=np.int32),
        unedited_labels=np.ones((2, 3, 4), dtype=np.int32) * 2,
    )
    second = ObjectLabelPayload(
        labels=np.ones((2, 3, 4), dtype=np.int32) * 3,
        unedited_labels=np.ones((2, 3, 4), dtype=np.int32) * 4,
    )

    stacked = Pure2DAuxiliaryOutputAggregator.aggregate([first, second], "numpy")

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


def test_measurement_table_normalizes_mixed_long_and_wide_rows():
    table = MeasurementTable(
        name="NucleiMeasurements",
        rows=[
            {"object_label": 1, "area": 12.0},
            {
                "object_label": 1,
                "feature_name": "Perimeter",
                "result_value": 8.0,
            },
        ],
        object_name="Nuclei",
    )

    assert table.fields == ()
    assert table.rows == [
        {
            "object_label": 1,
            "feature_name": "area",
            "result_value": 12.0,
        },
        {
            "object_label": 1,
            "feature_name": "Perimeter",
            "result_value": 8.0,
        },
    ]


def test_measurement_table_treats_value_named_columns_as_wide_without_feature_axis():
    rows = [{"image_number": 1, "mean_value": 0.5, "min_value": 0.1}]

    table = MeasurementTable(name="ImageMeasurements", rows=rows)

    assert table.rows is rows


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
