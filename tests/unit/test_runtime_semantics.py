from __future__ import annotations

import numpy as np
import pytest

from openhcs.core.aligned_image_payload import (
    AlignedImageStack,
    AlignedImageStackKwargResolver,
    ImagePayloadBundleContext,
    ObjectLabelPayloadSourceSpatialDomainAdapter,
)
from openhcs.core.source_spatial_domain import SourceSpatialDomain
from openhcs.core.runtime_object_label_domains import (
    ObjectLabelDomain,
    ObjectLabelDomainScope,
    DenseObjectLabelConsecutiveRelabelingStrategy,
    DenseObjectLabelExtentDomainDeclaration,
    ObjectLabelIdDomainStrategy,
    PresentObjectLabelIdsDomainDeclaration,
    dense_object_label_id_domain,
    dense_object_label_identity_domains,
    dense_object_label_measurement_row_domain,
    dense_object_label_plane_id_domains,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelRepresentation,
)
from openhcs.core.runtime_relationships import (
    ObjectInstanceKey,
    DirectedObjectRelationshipPayload,
    object_label_identity_lineage_payload,
    object_label_parent_child_payload,
)
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxis,
    RuntimePlaneAxisValueProjection,
    RuntimePlaneProjection,
)
from openhcs.core.source_spatial_domain import (
    SourceSpatialDomainAdapter,
)
from openhcs.core.runtime_image_values import (
    ImageMetadataPayload,
    ImagePayloadMetadata,
    MaskedImagePayload,
    image_payload_data,
    image_payload_metadata,
    preserve_declared_image_payload_axis,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelVariantData,
    ObjectLabelPayload,
    ObjectLabelSet,
)
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValues
from openhcs.core.runtime_sparse_labels import SparseIJVLabelRows


def declared_label_payload(labels: object) -> ObjectLabelPayload:
    """Build a payload through the explicit material-domain producer contract."""
    return ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=PresentObjectLabelIdsDomainDeclaration().declared_domain(None, labels),
    )


def test_runtime_plane_projection_is_nominal_and_validated() -> None:
    stack_projection = RuntimePlaneProjection.stack(3)
    selected_projection = RuntimePlaneProjection.selected(1, 3)

    assert isinstance(stack_projection, RuntimePlaneProjection)
    assert stack_projection.runtime_slice_plane_index() is None
    assert stack_projection.runtime_slice_axis_size() == 3
    assert selected_projection.runtime_slice_plane_index() == 1
    assert selected_projection.runtime_slice_axis_size() == 3

    with pytest.raises(ValueError, match="cannot be negative"):
        RuntimePlaneProjection.selected(-1)


def test_runtime_plane_projection_preserves_payload_declared_axis() -> None:
    payload = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(np.ones((2, 4, 5), dtype=np.float32), None)

    projection = preserve_declared_image_payload_axis(
        RuntimePlaneProjection.stack(2),
        payload,
    )

    assert projection == RuntimePlaneAxisValueProjection.preserve(
        axis=RuntimePlaneAxis.RUNTIME_SLICE,
        axis_size=2,
    )


def test_runtime_plane_projection_preserves_payload_owned_axis_cardinality() -> None:
    payload = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(np.ones((1, 4, 5), dtype=np.float32), None)

    projection = preserve_declared_image_payload_axis(
        RuntimePlaneProjection.stack(2),
        payload,
    )

    assert projection == RuntimePlaneAxisValueProjection.preserve(
        axis=RuntimePlaneAxis.RUNTIME_SLICE,
        axis_size=1,
    )


def test_runtime_plane_projection_prefers_declared_output_axis_cardinality() -> None:
    source = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(np.ones((2, 4, 5), dtype=np.float32), None)
    output = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(np.ones((1, 4, 5), dtype=np.float32), None)

    projection = preserve_declared_image_payload_axis(
        RuntimePlaneProjection.stack(2),
        output,
        source_payload=source,
    )

    assert projection == RuntimePlaneAxisValueProjection.preserve(
        axis=RuntimePlaneAxis.RUNTIME_SLICE,
        axis_size=1,
    )


def test_runtime_plane_projection_preserves_nominal_aligned_output_axis() -> None:
    output = RuntimeSliceAlignedValues(
        (
            np.ones((4, 5), dtype=np.float32),
            np.zeros((4, 5), dtype=np.float32),
        )
    )

    projection = preserve_declared_image_payload_axis(
        RuntimePlaneProjection.stack(2),
        output,
    )

    assert projection == RuntimePlaneAxisValueProjection.preserve(
        axis=RuntimePlaneAxis.RUNTIME_SLICE,
        axis_size=2,
    )


def test_runtime_plane_projection_does_not_preserve_selected_payload_axis() -> None:
    payload = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(np.ones((2, 4, 5), dtype=np.float32), None)

    projection = preserve_declared_image_payload_axis(
        RuntimePlaneProjection.selected(0, 2),
        payload,
    )

    assert projection is None


def test_runtime_plane_projection_preserves_source_binding_axis_from_payload() -> None:
    payload = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        source_image_names=("DNA", "RNA"),
    ).payload_with(np.ones((2, 4, 5), dtype=np.float32), None)

    projection = preserve_declared_image_payload_axis(
        RuntimePlaneProjection.stack(),
        payload,
    )

    assert projection == RuntimePlaneAxisValueProjection.preserve(
        axis=RuntimePlaneAxis.SOURCE_BINDING,
        axis_size=2,
    )


def test_runtime_plane_projection_preserves_output_axis_over_source_axis() -> None:
    source = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(np.ones((2, 4, 5), dtype=np.float32), None)
    output = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
    ).payload_with(np.ones((2, 4, 5), dtype=np.float32), None)

    projection = preserve_declared_image_payload_axis(
        RuntimePlaneProjection.stack(2),
        output,
        source_payload=source,
    )

    assert projection == RuntimePlaneAxisValueProjection.preserve(
        axis=RuntimePlaneAxis.SOURCE_BINDING,
        axis_size=2,
    )


def test_object_instance_key_uses_declared_runtime_slice_field_only() -> None:
    key = ObjectInstanceKey.from_measurement_row(
        {
            "slice_index": 1,
            "image_number": 1,
            "object_label": 7,
        },
        7,
    )

    assert key == ObjectInstanceKey(7, slice_index=1)


def test_aligned_dense_object_label_arrays_rejects_implicit_stack_projection() -> None:
    first_plane = np.array(
        [
            [1, 1, 0],
            [0, 0, 2],
        ],
        dtype=np.int32,
    )
    stack = np.stack((first_plane, np.zeros_like(first_plane)))
    reference = np.array(
        [
            [10, 10, 0],
            [0, 0, 20],
        ],
        dtype=np.int32,
    )

    with pytest.raises(ValueError, match="must share a common geometry"):
        SourceSpatialDomainAdapter.aligned_values((stack, reference))


def test_object_label_domain_project_slice_selects_declared_plane_domain() -> None:
    domain = ObjectLabelDomain(
        declared_object_id_domains=((1, 2), (1, 2, 3, 4)),
        scope=ObjectLabelDomainScope.PLANE,
    )

    projected = domain.project_slice(slice_index=1, slice_count=2)

    assert projected.declared_object_ids == (1, 2, 3, 4)
    assert projected.declared_object_id_domains == ()
    assert projected.scope is ObjectLabelDomainScope.PAYLOAD


def test_object_label_domain_project_slice_rejects_single_domain_broadcast() -> None:
    domain = ObjectLabelDomain(
        declared_object_id_domains=((1, 2),),
        scope=ObjectLabelDomainScope.PLANE,
    )

    with pytest.raises(ValueError, match="must match PURE_2D slice count"):
        domain.project_slice(slice_index=1, slice_count=2)


def test_object_label_domain_project_slice_rejects_grouped_plane_domains() -> None:
    domain = ObjectLabelDomain(
        declared_object_id_domains=((1,), (2,), (3,), (4,)),
        scope=ObjectLabelDomainScope.PLANE,
    )

    with pytest.raises(ValueError, match="must match PURE_2D slice count"):
        domain.project_slice(slice_index=1, slice_count=2)


def test_object_label_domain_project_slice_rejects_repeated_plane_domains() -> None:
    domain = ObjectLabelDomain(
        declared_object_id_domains=((1,), (2,)),
        scope=ObjectLabelDomainScope.PLANE,
    )

    with pytest.raises(ValueError, match="must match PURE_2D slice count"):
        domain.project_slice(slice_index=3, slice_count=4)


def test_object_label_domain_project_planes_selects_noncontiguous_domains() -> None:
    domain = ObjectLabelDomain(
        declared_object_id_domains=((1,), (2,), (3,), (4,)),
        scope=ObjectLabelDomainScope.PLANE,
    )

    projected = domain.project_planes((1, 3))

    assert projected.declared_object_ids == ()
    assert projected.declared_object_id_domains == ((2,), (4,))
    assert projected.scope is ObjectLabelDomainScope.PLANE


def test_plane_domain_strategy_rejects_mismatched_single_output_plane() -> None:
    labels = np.array([[0, 2], [3, 0]], dtype=np.int32)

    with pytest.raises(ValueError, match=r"declares 3 plane\(s\)"):
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=labels),
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            domain=ObjectLabelDomain(
                scope=ObjectLabelDomainScope.PLANE,
                declared_object_id_domains=((1,), (2,), (3,)),
            ),
        )


def test_object_label_domain_project_slice_rejects_mismatched_plane_domains() -> None:
    domain = ObjectLabelDomain(
        declared_object_id_domains=((1, 2), (1, 2, 3, 4)),
        scope=ObjectLabelDomainScope.PLANE,
    )

    with pytest.raises(ValueError, match="must match PURE_2D slice count"):
        domain.project_slice(slice_index=0, slice_count=3)


def test_compose_one_image_bundle_preserves_shared_crop_domain() -> None:
    metadata = ImagePayloadMetadata(source_dtype="float32").with_spatial_crop(
        input_shape_yx=(8, 9),
        output_shape_yx=(5, 5),
        offset_yx=(1, 2),
        physical_border_edges_yx=(False, False, False, False),
    )
    first = metadata.payload_with(np.ones((5, 5), dtype=np.float32), None)
    second = metadata.payload_with(np.full((5, 5), 2, dtype=np.float32), None)

    bundle = ImagePayloadBundleContext.from_payloads((first, second)).compose()

    assert image_payload_data(bundle).shape[-2:] == (5, 5)
    assert image_payload_metadata(bundle).spatial_origin_yx == (1, 2)
    assert image_payload_metadata(bundle).source_spatial_shape_yx == (8, 9)


def test_aligned_image_stack_exposes_slice_source_spatial_domain() -> None:
    metadata = ImagePayloadMetadata(source_dtype="float32").with_spatial_crop(
        input_shape_yx=(8, 9),
        output_shape_yx=(5, 5),
        offset_yx=(1, 2),
        physical_border_edges_yx=(False, False, False, False),
    )
    reference = metadata.payload_with(np.ones((5, 5), dtype=np.float32), None)
    stack = AlignedImageStack((reference,))
    adapter = stack.first_slice_source_spatial_adapter()

    assert adapter is not None
    assert adapter.payload_domain.origin_yx == (1, 2)
    assert adapter.payload_domain.spatial_shape_yx == (5, 5)
    assert adapter.payload_domain.source_shape_yx == (8, 9)


def test_aligned_image_stack_kwarg_resolver_selects_nominal_stack_slice() -> None:
    first = np.zeros((3, 4), dtype=np.int32)
    second = np.ones((3, 4), dtype=np.int32)
    stack = AlignedImageStack((first, second))

    resolved = AlignedImageStackKwargResolver(
        projection_axis=RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.RUNTIME_SLICE, plane_index=1, axis_size=2
        ),
        reference_payload=second,
    ).resolve(stack)

    assert resolved is second


def test_aligned_image_stack_kwarg_resolver_requires_exact_cardinality() -> None:
    stack = AlignedImageStack((np.zeros((3, 4), dtype=np.int32),))
    resolver = AlignedImageStackKwargResolver(
        projection_axis=RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.RUNTIME_SLICE, plane_index=0, axis_size=2
        ),
    )

    with pytest.raises(ValueError, match="cardinality must exactly match"):
        resolver.resolve(stack)


def test_aligned_kwarg_resolution_preserves_image_payload_metadata() -> None:
    source = ImageMetadataPayload(
        data=np.arange(16, dtype=np.float32).reshape(4, 4),
        metadata=ImagePayloadMetadata(
            source_dtype="float32",
            source_spatial_domain=SourceSpatialDomain(source_shape_yx=(4, 4)),
        ),
    )
    reference = ImagePayloadMetadata(
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(2, 0),
            source_shape_yx=(4, 4),
        ),
    ).payload_with(np.ones((2, 2), dtype=np.float32), None)

    resolved = AlignedImageStackKwargResolver(
        projection_axis=RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.RUNTIME_SLICE, plane_index=0, axis_size=1
        ),
        reference_payload=reference,
    ).resolve(source)

    assert isinstance(resolved, ImageMetadataPayload)
    np.testing.assert_array_equal(resolved.data, source.data[2:4, 0:2])
    assert resolved.metadata.source_dtype == "float32"
    assert resolved.metadata.source_spatial_domain.origin_yx == (2, 0)
    assert resolved.metadata.source_spatial_domain.source_shape_yx == (4, 4)


def test_aligned_kwarg_resolution_preserves_image_payload_mask() -> None:
    source = MaskedImagePayload(
        data=np.arange(16, dtype=np.float32).reshape(4, 4),
        mask=np.arange(16).reshape(4, 4) % 2 == 0,
        metadata=ImagePayloadMetadata(
            source_spatial_domain=SourceSpatialDomain(source_shape_yx=(4, 4)),
        ),
    )
    reference = ImagePayloadMetadata(
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(1, 1),
            source_shape_yx=(4, 4),
        ),
    ).payload_with(np.ones((2, 2), dtype=np.float32), None)

    resolved = AlignedImageStackKwargResolver(
        projection_axis=RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.RUNTIME_SLICE, plane_index=0, axis_size=1
        ),
        reference_payload=reference,
    ).resolve(source)

    assert isinstance(resolved, MaskedImagePayload)
    np.testing.assert_array_equal(resolved.data, source.data[1:3, 1:3])
    np.testing.assert_array_equal(resolved.mask, source.mask[1:3, 1:3])
    assert resolved.metadata.source_spatial_domain.origin_yx == (1, 1)


def test_source_spatial_domain_adapter_extracts_source_array_to_payload_domain() -> (
    None
):
    metadata = ImagePayloadMetadata(source_dtype="float32").with_spatial_crop(
        input_shape_yx=(8, 9),
        output_shape_yx=(5, 5),
        offset_yx=(1, 2),
        physical_border_edges_yx=(False, False, False, False),
    )
    image = metadata.payload_with(np.ones((5, 5), dtype=np.float32), None)
    labels = np.arange(72, dtype=np.int32).reshape(8, 9)
    adapter = SourceSpatialDomainAdapter.for_value(image)

    assert adapter is not None
    extracted = adapter.extract_source_array(labels, spatial_axes_yx=(0, 1))

    assert extracted.shape == (5, 5)
    np.testing.assert_array_equal(extracted, labels[1:6, 2:7])


def test_aligned_dense_object_label_arrays_never_selects_projection_by_content() -> (
    None
):
    first_plane = np.array([[1, 0]], dtype=np.int32)
    second_plane = np.array([[2, 0]], dtype=np.int32)

    with pytest.raises(ValueError, match="must share a common geometry"):
        SourceSpatialDomainAdapter.aligned_values(
            (
                np.stack((first_plane, second_plane)),
                np.array([[1, 0]], dtype=np.int32),
            )
        )


def test_object_label_parent_child_payload_rejects_undeclared_stack_projection() -> (
    None
):
    parent_plane = np.array(
        [
            [1, 1, 0],
            [0, 2, 2],
        ],
        dtype=np.int32,
    )
    child_plane = np.array(
        [
            [1, 1, 0],
            [0, 2, 2],
        ],
        dtype=np.int32,
    )
    parent_stack = np.stack((parent_plane, parent_plane))

    with pytest.raises(ValueError, match="must share a common geometry"):
        object_label_parent_child_payload(parent_stack, child_plane)


def test_object_label_parent_child_payload_preserves_plane_scoped_stack_identity() -> (
    None
):
    parent_stack = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.asarray(
                (
                    ((1, 1, 0), (0, 0, 0)),
                    ((2, 2, 0), (0, 0, 0)),
                ),
                dtype=np.int32,
            )
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    child_stack = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.asarray(
                (
                    ((1, 1, 0), (0, 0, 0)),
                    ((1, 1, 0), (0, 0, 0)),
                ),
                dtype=np.int32,
            )
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (1,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    payload = object_label_parent_child_payload(parent_stack, child_stack)

    assert payload.source_ids == (1, 2)
    assert payload.target_ids == (1, 1)
    assert payload.slice_indices == (0, 1)
    assert payload.slice_count == 2


def test_parent_child_payload_does_not_infer_slice_identity() -> None:
    payload = DirectedObjectRelationshipPayload(source_ids=(1, 2), target_ids=(3, 4))

    assert payload.slice_indices == ()
    assert payload.slice_count is None


def test_aligned_dense_object_label_arrays_reject_erased_source_domain() -> None:
    compact_parent = np.array(
        [
            [1, 1],
            [0, 2],
        ],
        dtype=np.int32,
    )
    compact_child = np.array(
        [
            [1, 1],
            [0, 2],
        ],
        dtype=np.int32,
    )
    parent_payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=compact_parent),
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(2, 3),
            source_shape_yx=(6, 7),
        ),
    )

    assert isinstance(
        SourceSpatialDomainAdapter.for_value(parent_payload),
        ObjectLabelPayloadSourceSpatialDomainAdapter,
    )
    with pytest.raises(ValueError, match="every value to declare a source domain"):
        SourceSpatialDomainAdapter.aligned_values((parent_payload, compact_child))


def test_aligned_dense_object_label_stacks_share_payload_source_domain() -> None:
    compact_primary = np.stack(
        (
            np.array([[1, 0], [0, 2]], dtype=np.int32),
            np.array([[1, 1], [0, 0]], dtype=np.int32),
        )
    )
    compact_secondary = compact_primary.copy()
    primary_payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=compact_primary),
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(1, 2),
            source_shape_yx=(5, 6),
        ),
    )
    secondary_payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=compact_secondary),
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(1, 2),
            source_shape_yx=(5, 6),
        ),
    )

    stacks, _adapters = SourceSpatialDomainAdapter.aligned_values(
        (primary_payload, secondary_payload)
    )

    assert stacks is not None
    primary_stack, secondary_stack = stacks
    assert primary_stack.shape == (2, 5, 6)
    assert secondary_stack.shape == (2, 5, 6)
    np.testing.assert_array_equal(primary_stack[:, 1:3, 2:4], compact_primary)
    np.testing.assert_array_equal(secondary_stack[:, 1:3, 2:4], compact_secondary)


def test_aligned_dense_object_label_stack_alignment_restores_secondary_domain() -> None:
    compact_primary = np.stack(
        (
            np.array([[1, 0], [0, 2]], dtype=np.int32),
            np.array([[1, 1], [0, 0]], dtype=np.int32),
        )
    )
    compact_secondary = compact_primary.copy()
    primary_payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=compact_primary),
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(1, 2),
            source_shape_yx=(5, 6),
        ),
    )
    secondary_payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=compact_secondary),
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(1, 2),
            source_shape_yx=(5, 6),
        ),
    )
    (primary_stack, secondary_stack), adapters = (
        SourceSpatialDomainAdapter.aligned_values(
            (primary_payload, secondary_payload)
        )
    )

    source_domain_output = np.zeros_like(secondary_stack)
    source_domain_output[:, 1:3, 2:4] = compact_secondary
    restored = adapters[1].extract_source_array(
        source_domain_output,
        spatial_axes_yx=adapters[1].spatial_axes_yx,
    )

    assert restored.shape == compact_secondary.shape
    np.testing.assert_array_equal(restored, compact_secondary)


def test_dense_object_label_stack_alignment_preserves_conflicting_label_planes() -> (
    None
):
    first = np.stack(
        (
            np.array([[1, 0], [0, 2]], dtype=np.int32),
            np.array([[3, 0], [0, 4]], dtype=np.int32),
        )
    )
    second = np.stack(
        (
            np.array([[10, 0], [0, 20]], dtype=np.int32),
            np.array([[30, 0], [0, 40]], dtype=np.int32),
        )
    )

    (first_stack, second_stack), _adapters = (
        SourceSpatialDomainAdapter.aligned_values((first, second))
    )

    np.testing.assert_array_equal(first_stack, first)
    np.testing.assert_array_equal(second_stack, second)


def test_dense_object_label_pair_alignment_rejects_factorized_runtime_axes() -> None:
    first = np.stack(
        tuple(np.full((4, 5), index + 1, dtype=np.int32) for index in range(6))
    )
    second = np.stack(
        (
            np.full((4, 5), 7, dtype=np.int32),
            np.full((4, 5), 8, dtype=np.int32),
        )
    )

    with pytest.raises(ValueError, match="must share a common geometry"):
        SourceSpatialDomainAdapter.aligned_values((first, second))


def test_dense_object_label_id_domain_uses_declared_count_for_empty_labels() -> None:
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.zeros((3, 3), dtype=np.int32)),
        domain=ObjectLabelDomain(
            declared_object_count=4,
        ),
    )

    assert dense_object_label_id_domain(payload) == (1, 2, 3, 4)
    assert (
        DenseObjectLabelExtentDomainDeclaration()
        .declared_domain(None, payload)
        .require_explicit_id_domain(context="Dense extent test")
        == ()
    )


def test_dense_object_label_id_domain_rejects_undeclared_dense_labels() -> None:
    labels = np.array([[1, 0, 3]], dtype=np.int32)

    with pytest.raises(ValueError, match="explicit object-ID domain"):
        dense_object_label_id_domain(labels)
    assert ObjectLabelIdDomainStrategy.for_value(labels).present_ids(labels) == (1, 3)
    assert DenseObjectLabelExtentDomainDeclaration().declared_domain(
        None, labels
    ).require_explicit_id_domain(context="Dense extent test") == (1, 2, 3)


def test_dense_object_label_measurement_row_domain_rejects_undeclared_labels() -> None:
    labels = np.array([[1, 0, 3]], dtype=np.int32)

    with pytest.raises(ValueError, match="explicit object-ID domain"):
        dense_object_label_measurement_row_domain(labels, labels)


def test_dense_object_label_measurement_row_domain_preserves_declared_count() -> None:
    labels = np.array([[1, 0, 3]], dtype=np.int32)
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain(declared_object_count=4),
    )

    assert dense_object_label_measurement_row_domain(payload, labels) == (1, 2, 3, 4)


def test_dense_object_label_measurement_row_domain_preserves_declared_ids() -> None:
    labels = np.array([[1, 0, 3]], dtype=np.int32)
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain(declared_object_ids=(1, 2, 3, 4)),
    )

    assert dense_object_label_measurement_row_domain(payload, labels) == (1, 2, 3, 4)


def test_dense_object_label_measurement_row_domain_does_not_expand_sparse_declared_ids() -> (
    None
):
    labels = np.array([[1, 0, 3], [0, 0, 5]], dtype=np.int32)
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain(declared_object_ids=(1, 3, 5)),
    )

    assert dense_object_label_measurement_row_domain(payload, labels) == (1, 3, 5)


def test_dense_object_label_measurement_row_domain_preserves_explicit_high_ids() -> (
    None
):
    labels = np.zeros((3, 3), dtype=np.int32)
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain(declared_object_ids=(10, 20, 30)),
    )

    assert dense_object_label_measurement_row_domain(payload, labels) == (10, 20, 30)


def test_object_label_identity_lineage_uses_declared_id_domain_for_3d_resize() -> None:
    parent = np.zeros((3, 4, 5), dtype=np.int32)
    child = np.zeros((2, 4, 5), dtype=np.int32)
    parent[0, 1, 1] = 1
    parent[1, 1, 2] = 2
    parent[2, 1, 3] = 3
    child[0, 1, 1] = 1
    child[1, 1, 2] = 2

    payload = object_label_identity_lineage_payload(
        declared_label_payload(parent),
        declared_label_payload(child),
    )

    assert payload.source_ids == (1, 2)
    assert payload.target_ids == (1, 2)


def test_dense_object_label_id_domain_handles_sparse_high_integer_ids() -> None:
    labels = np.array([[0, 100_000]], dtype=np.int32)

    assert ObjectLabelIdDomainStrategy.for_value(labels).present_ids(labels) == (
        100_000,
    )


def test_dense_object_label_id_domain_handles_float_labels() -> None:
    labels = np.array([[0.0, 2.0]], dtype=np.float64)

    assert ObjectLabelIdDomainStrategy.for_value(labels).present_ids(labels) == (2,)


def test_object_label_id_domain_uses_sparse_ijv_labels_without_densifying() -> None:
    rows = SparseIJVLabelRows.from_label_slice(
        np.array(
            [
                [0, 0, 4],
                [1, 2, 2],
                [4, 5, 4],
            ],
            dtype=np.int32,
        )
    )

    assert ObjectLabelIdDomainStrategy.for_value(rows).present_ids(rows) == (2, 4)
    assert ObjectLabelIdDomainStrategy.for_value(rows).max_present_id(rows) == 4
    with pytest.raises(ValueError, match="explicit object-ID domain"):
        dense_object_label_id_domain(rows)


def test_object_label_id_domain_delegates_through_sparse_payload_wrappers() -> None:
    rows = SparseIJVLabelRows.from_slices(
        (
            SparseIJVLabelRows.from_label_slice(np.array([[0, 0, 3]], dtype=np.int32)),
            SparseIJVLabelRows.from_label_slice(np.array([[0, 0, 1]], dtype=np.int32)),
        )
    )
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=rows),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
    )
    label_set = ObjectLabelSet(
        name="Worms",
        variant_data=ObjectLabelVariantData(labels=rows),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
    )

    assert ObjectLabelIdDomainStrategy.for_value(payload).present_ids(payload) == (1, 3)
    assert ObjectLabelIdDomainStrategy.for_value(label_set).present_ids(label_set) == (
        1,
        3,
    )


def test_dense_object_label_plane_id_domains_use_explicit_plane_declarations() -> None:
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.array(
                [
                    [[1, 3], [0, 0]],
                    [[0, 2], [0, 0]],
                ],
                dtype=np.int32,
            )
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1, 3), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    assert dense_object_label_plane_id_domains(labels) == (
        (1, 3),
        (2,),
    )


def test_dense_object_label_plane_id_domains_preserve_explicit_single_plane() -> None:
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.array(
                [
                    [
                        [1, 0],
                        [0, 3],
                    ],
                ],
                dtype=np.int32,
            )
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1, 3),),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    assert dense_object_label_plane_id_domains(labels) == ((1, 3),)
    assert dense_object_label_identity_domains(labels) == ((1, 3),)


def test_dense_object_label_relabeling_strategy_uses_single_semantic_remap() -> None:
    labels = np.array(
        [
            [0, 1000, 7],
            [42, 0, 1000],
        ],
        dtype=np.int32,
    )

    relabeled = DenseObjectLabelConsecutiveRelabelingStrategy.for_labels(
        labels
    ).relabel(labels, dtype=np.int32)

    np.testing.assert_array_equal(
        relabeled,
        np.array(
            [
                [0, 3, 1],
                [2, 0, 3],
            ],
            dtype=np.int32,
        ),
    )


def test_payload_object_label_identity_domain_does_not_repeat_stack_planes() -> None:
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.array(
                [
                    [[1, 2], [0, 0]],
                    [[1, 2], [0, 0]],
                ],
                dtype=np.int32,
            )
        ),
        domain=ObjectLabelDomain(declared_object_ids=(1, 2)),
    )

    assert dense_object_label_plane_id_domains(labels) == ((1, 2),)
    assert dense_object_label_identity_domains(labels) == ((1, 2),)
