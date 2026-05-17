from __future__ import annotations

import numpy as np
import pytest

from openhcs.core.aligned_image_payload import (
    AlignedImageStack,
    ObjectLabelPayloadSourceSpatialDomainAdapter,
    aligned_image_stack_kwarg,
    compose_one_image_bundle,
)
from openhcs.core.runtime_semantics import (
    ObjectLabelDomain,
    ObjectLabelDomainScope,
    ObjectLabelRepresentation,
    GroupRuntimePlaneProjection,
    RuntimePlaneProjection,
    SourceSpatialDomainAdapter,
    StackRuntimePlaneProjection,
    DenseObjectLabelConsecutiveRelabelingStrategy,
    DenseObjectLabelPairAligner,
    ObjectLabelIdDomainStrategy,
    dense_object_label_extent_id_domain,
    dense_object_label_id_domain,
    dense_object_label_identity_domains,
    dense_object_label_plane_id_domains,
    object_label_parent_child_payload,
)
from openhcs.core.runtime_values import (
    ImagePayloadMetadata,
    ObjectLabelPayload,
    ObjectLabelSet,
    SparseIJVLabelRows,
    image_payload_data,
    image_payload_metadata,
    image_payload_with_context,
)


def test_runtime_plane_projection_is_nominal_and_validated() -> None:
    stack_projection = RuntimePlaneProjection.for_group_key(None, plane_index=None)
    group_projection = RuntimePlaneProjection.for_group_key("2", plane_index=1)

    assert isinstance(stack_projection, StackRuntimePlaneProjection)
    assert stack_projection.runtime_slice_plane_index() is None
    assert isinstance(group_projection, GroupRuntimePlaneProjection)
    assert group_projection.runtime_slice_plane_index() == 1

    with pytest.raises(ValueError, match="Ungrouped runtime execution"):
        RuntimePlaneProjection.for_group_key(None, plane_index=0)
    with pytest.raises(ValueError, match="Grouped runtime execution requires"):
        RuntimePlaneProjection.for_group_key("2", plane_index=None)
    with pytest.raises(ValueError, match="cannot be negative"):
        RuntimePlaneProjection.group(-1)


def test_aligned_dense_object_label_arrays_projects_unambiguous_stack() -> None:
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

    aligned_stack, aligned_reference = DenseObjectLabelPairAligner(
        stack,
        reference,
    ).aligned()

    np.testing.assert_array_equal(aligned_stack, first_plane)
    np.testing.assert_array_equal(aligned_reference, reference)


def test_object_label_domain_project_slice_selects_declared_plane_domain() -> None:
    domain = ObjectLabelDomain(
        declared_object_id_domains=((1, 2), (1, 2, 3, 4)),
        scope=ObjectLabelDomainScope.PLANE,
    )

    projected = domain.project_slice(slice_index=1, slice_count=2)

    assert projected.declared_object_ids == (1, 2, 3, 4)
    assert projected.declared_object_id_domains == ()
    assert projected.scope is ObjectLabelDomainScope.PLANE


def test_object_label_domain_project_slice_broadcasts_single_declared_plane_domain() -> None:
    domain = ObjectLabelDomain(
        declared_object_id_domains=((1, 2),),
        scope=ObjectLabelDomainScope.PLANE,
    )

    projected = domain.project_slice(slice_index=1, slice_count=2)

    assert projected.declared_object_ids == (1, 2)
    assert projected.declared_object_id_domains == ()
    assert projected.scope is ObjectLabelDomainScope.PLANE


def test_object_label_domain_project_slice_selects_grouped_plane_domains() -> None:
    domain = ObjectLabelDomain(
        declared_object_id_domains=((1,), (2,), (3,), (4,)),
        scope=ObjectLabelDomainScope.PLANE,
    )

    projected = domain.project_slice(slice_index=1, slice_count=2)

    assert projected.declared_object_ids == ()
    assert projected.declared_object_id_domains == ((3,), (4,))
    assert projected.scope is ObjectLabelDomainScope.PLANE


def test_object_label_domain_project_slice_repeats_declared_plane_domains() -> None:
    domain = ObjectLabelDomain(
        declared_object_id_domains=((1,), (2,)),
        scope=ObjectLabelDomainScope.PLANE,
    )

    projected = domain.project_slice(slice_index=3, slice_count=4)

    assert projected.declared_object_ids == (2,)
    assert projected.declared_object_id_domains == ()
    assert projected.scope is ObjectLabelDomainScope.PLANE


def test_object_label_domain_project_planes_selects_noncontiguous_domains() -> None:
    domain = ObjectLabelDomain(
        declared_object_id_domains=((1,), (2,), (3,), (4,)),
        scope=ObjectLabelDomainScope.PLANE,
    )

    projected = domain.project_planes((1, 3))

    assert projected.declared_object_ids == ()
    assert projected.declared_object_id_domains == ((2,), (4,))
    assert projected.scope is ObjectLabelDomainScope.PLANE


def test_plane_domain_strategy_rederives_single_output_plane_from_labels() -> None:
    labels = np.array([[0, 2], [3, 0]], dtype=np.int32)

    domains = dense_object_label_plane_id_domains(
        labels,
        domain_scope=ObjectLabelDomainScope.PLANE,
        declared_object_id_domains=((1,), (2,), (3,)),
    )

    assert domains == ((2, 3),)


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
    first = image_payload_with_context(
        np.ones((5, 5), dtype=np.float32),
        metadata=metadata,
    )
    second = image_payload_with_context(
        np.full((5, 5), 2, dtype=np.float32),
        metadata=metadata,
    )

    bundle = compose_one_image_bundle((first, second))

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
    reference = image_payload_with_context(
        np.ones((5, 5), dtype=np.float32),
        metadata=metadata,
    )
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

    resolved = aligned_image_stack_kwarg(
        stack,
        slice_index=1,
        slice_count=2,
        reference_payload=second,
    )

    assert resolved is second


def test_source_spatial_domain_adapter_extracts_source_array_to_payload_domain() -> None:
    metadata = ImagePayloadMetadata(source_dtype="float32").with_spatial_crop(
        input_shape_yx=(8, 9),
        output_shape_yx=(5, 5),
        offset_yx=(1, 2),
        physical_border_edges_yx=(False, False, False, False),
    )
    image = image_payload_with_context(
        np.ones((5, 5), dtype=np.float32),
        metadata=metadata,
    )
    labels = np.arange(72, dtype=np.int32).reshape(8, 9)
    adapter = SourceSpatialDomainAdapter.for_value(image)

    assert adapter is not None
    extracted = adapter.extract_source_array(labels)

    assert extracted.shape == (5, 5)
    np.testing.assert_array_equal(extracted, labels[1:6, 2:7])


def test_aligned_dense_object_label_arrays_rejects_conflicting_stack_projection() -> None:
    first_plane = np.array([[1, 0]], dtype=np.int32)
    second_plane = np.array([[2, 0]], dtype=np.int32)

    with pytest.raises(ValueError, match="conflicting positive labels"):
        DenseObjectLabelPairAligner(
            np.stack((first_plane, second_plane)),
            np.array([[1, 0]], dtype=np.int32),
        ).aligned()


def test_object_label_parent_child_payload_aligns_parent_stack_to_child_plane() -> None:
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

    payload = object_label_parent_child_payload(parent_stack, child_plane)

    assert payload.parent_ids == (1, 2)
    assert payload.child_ids == (1, 2)


def test_object_label_parent_child_payload_preserves_plane_scoped_stack_identity() -> None:
    parent_stack = ObjectLabelPayload(
        labels=np.asarray(
            (
                ((1, 1, 0), (0, 0, 0)),
                ((2, 2, 0), (0, 0, 0)),
            ),
            dtype=np.int32,
        ),
        domain_scope=ObjectLabelDomainScope.PLANE,
    )
    child_stack = ObjectLabelPayload(
        labels=np.asarray(
            (
                ((1, 1, 0), (0, 0, 0)),
                ((1, 1, 0), (0, 0, 0)),
            ),
            dtype=np.int32,
        ),
        domain_scope=ObjectLabelDomainScope.PLANE,
    )

    payload = object_label_parent_child_payload(parent_stack, child_stack)

    assert payload.parent_ids == (1, 2)
    assert payload.child_ids == (1, 1)
    assert payload.slice_indices == (0, 1)
    assert payload.slice_count == 2


def test_aligned_dense_object_label_arrays_applies_payload_domain_to_matching_raw_labels() -> None:
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
        labels=compact_parent,
        spatial_origin_yx=(2, 3),
        source_spatial_shape_yx=(6, 7),
    )

    assert isinstance(
        SourceSpatialDomainAdapter.for_value(parent_payload),
        ObjectLabelPayloadSourceSpatialDomainAdapter,
    )
    parent, child = DenseObjectLabelPairAligner(
        parent_payload,
        compact_child,
    ).aligned()

    assert parent.shape == (6, 7)
    assert child.shape == (6, 7)
    np.testing.assert_array_equal(parent[2:4, 3:5], compact_parent)
    np.testing.assert_array_equal(child[2:4, 3:5], compact_child)


def test_aligned_dense_object_label_stacks_share_payload_source_domain() -> None:
    compact_primary = np.stack(
        (
            np.array([[1, 0], [0, 2]], dtype=np.int32),
            np.array([[1, 1], [0, 0]], dtype=np.int32),
        )
    )
    compact_secondary = compact_primary.copy()
    primary_payload = ObjectLabelPayload(
        labels=compact_primary,
        spatial_origin_yx=(1, 2),
        source_spatial_shape_yx=(5, 6),
    )

    stacks = DenseObjectLabelPairAligner(
        primary_payload,
        compact_secondary,
    ).aligned_stacks(2)

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
        labels=compact_primary,
        spatial_origin_yx=(1, 2),
        source_spatial_shape_yx=(5, 6),
    )
    alignment = DenseObjectLabelPairAligner(
        primary_payload,
        compact_secondary,
    ).aligned_stack_context(2)

    assert alignment is not None
    source_domain_output = np.zeros_like(alignment.second_stack)
    source_domain_output[:, 1:3, 2:4] = compact_secondary
    restored = alignment.restore_second_stack(source_domain_output)

    assert restored.shape == compact_secondary.shape
    np.testing.assert_array_equal(restored, compact_secondary)


def test_dense_object_label_id_domain_uses_declared_count_for_empty_labels() -> None:
    payload = ObjectLabelPayload(
        labels=np.zeros((3, 3), dtype=np.int32),
        declared_object_count=4,
    )

    assert dense_object_label_id_domain(payload) == (1, 2, 3, 4)
    assert dense_object_label_extent_id_domain(payload) == ()


def test_dense_object_label_id_domain_uses_present_dense_ids_without_declaration() -> None:
    labels = np.array([[1, 0, 3]], dtype=np.int32)

    assert dense_object_label_id_domain(labels) == (1, 3)
    assert dense_object_label_extent_id_domain(labels) == (1, 2, 3)


def test_object_label_id_domain_uses_sparse_ijv_labels_without_densifying() -> None:
    rows = SparseIJVLabelRows.from_yx_label(
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
    assert dense_object_label_id_domain(rows) == (2, 4)


def test_object_label_id_domain_delegates_through_sparse_payload_wrappers() -> None:
    rows = SparseIJVLabelRows.from_slices(
        (
            SparseIJVLabelRows.from_yx_label(np.array([[0, 0, 3]], dtype=np.int32)),
            SparseIJVLabelRows.from_yx_label(np.array([[0, 0, 1]], dtype=np.int32)),
        )
    )
    payload = ObjectLabelPayload(labels=rows)
    label_set = ObjectLabelSet(
        name="Worms",
        labels=rows,
        representation=ObjectLabelRepresentation.SPARSE_IJV,
    )

    assert ObjectLabelIdDomainStrategy.for_value(payload).present_ids(payload) == (1, 3)
    assert ObjectLabelIdDomainStrategy.for_value(label_set).present_ids(label_set) == (
        1,
        3,
    )


def test_dense_object_label_plane_id_domains_repeat_plane_scoped_stack_declaration() -> None:
    labels = ObjectLabelPayload(
        labels=np.array(
            [
                [[1, 3], [0, 0]],
                [[0, 2], [0, 0]],
            ],
            dtype=np.int32,
        ),
        declared_object_count=4,
        domain_scope=ObjectLabelDomainScope.PLANE,
    )

    assert dense_object_label_plane_id_domains(labels) == (
        (1, 2, 3, 4),
        (1, 2, 3, 4),
    )


def test_dense_object_label_plane_id_domains_preserve_single_plane_declaration() -> None:
    labels = ObjectLabelPayload(
        labels=np.array(
            [
                [1, 0],
                [0, 3],
            ],
            dtype=np.int32,
        ),
        declared_object_count=4,
        domain_scope=ObjectLabelDomainScope.PLANE,
    )

    assert dense_object_label_plane_id_domains(labels) == ((1, 2, 3, 4),)
    assert dense_object_label_identity_domains(labels) == ((1, 2, 3, 4),)


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
        labels=np.array(
            [
                [[1, 2], [0, 0]],
                [[1, 2], [0, 0]],
            ],
            dtype=np.int32,
        ),
    )

    assert dense_object_label_plane_id_domains(labels) == ((1, 2), (1, 2))
    assert dense_object_label_identity_domains(labels) == ((1, 2),)
