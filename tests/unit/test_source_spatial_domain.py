import numpy as np

from openhcs.core.aligned_image_payload import ObjectLabelPayloadSourceSpatialDomainAdapter
from openhcs.core.runtime_object_labels import ObjectLabelRepresentation, ObjectLabelVariant
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    )
from openhcs.core.runtime_object_label_building import SourceImageObjectLabelBuildRequest
from openhcs.core.runtime_sparse_labels import SparseIJVLabelRows
from openhcs.core.source_spatial_domain import (
    SourceSpatialDomain,
    SourceSpatialDomainAdapter,
    dense_array_in_source_spatial_domain,
)


def test_dense_source_domain_materializes_hwc_color_crop_over_yx_axes():
    image = np.ones((2, 3, 3), dtype=np.uint8)

    materialized = dense_array_in_source_spatial_domain(
        image,
        spatial_axes_yx=(0, 1),
        spatial_origin_yx=(1, 2),
        source_spatial_shape_yx=(5, 7),
        fill_value=0,
        value_name="Image payload",
    )

    assert materialized.shape == (5, 7, 3)
    assert np.all(materialized[1:3, 2:5] == image)
    assert np.count_nonzero(materialized[:1]) == 0


def test_dense_source_domain_materializes_nhwc_color_stack_over_yx_axes():
    image = np.ones((2, 3, 4, 3), dtype=np.uint8)

    materialized = dense_array_in_source_spatial_domain(
        image,
        spatial_axes_yx=(1, 2),
        spatial_origin_yx=(1, 2),
        source_spatial_shape_yx=(6, 8),
        fill_value=0,
        value_name="Image payload",
    )

    assert materialized.shape == (2, 6, 8, 3)
    assert np.all(materialized[:, 1:4, 2:6] == image)
    assert np.count_nonzero(materialized[:, :1]) == 0


def test_object_label_source_domain_adapter_uses_dense_sparse_ijv_domain_shape():
    source_image = np.zeros((5, 7), dtype=np.uint8)
    labels = SourceImageObjectLabelBuildRequest(
        image=source_image,
        labels=SparseIJVLabelRows(np.zeros((0, 3), dtype=np.int32)),
    ).label_set(
        name="Worms",
        representation=ObjectLabelRepresentation.SPARSE_IJV,
    )

    adapter = SourceSpatialDomainAdapter.for_value(labels)

    assert isinstance(adapter, ObjectLabelPayloadSourceSpatialDomainAdapter)
    assert adapter.spatial_shape_yx == source_image.shape
    np.testing.assert_array_equal(adapter.materialize(), np.zeros_like(source_image))


def test_sparse_object_label_projection_preserves_all_declared_variants():
    source_image = np.zeros((8, 9), dtype=np.uint8)
    final_rows = SparseIJVLabelRows.from_label_slice(
        np.asarray(((2, 3, 1), (4, 6, 2)), dtype=np.int32)
    )
    unedited_rows = SparseIJVLabelRows.from_label_slice(
        np.asarray(((2, 3, 1), (5, 6, 3)), dtype=np.int32)
    )
    labels = SourceImageObjectLabelBuildRequest(
        image=source_image,
        labels=final_rows,
        unedited_labels=unedited_rows,
    ).label_set(
        name="Objects",
        representation=ObjectLabelRepresentation.SPARSE_IJV,
    )
    target = ImagePayloadMetadata(
            source_spatial_domain=SourceSpatialDomain(
                origin_yx=(2, 3),
                source_shape_yx=source_image.shape,
            )
        ).payload_with(np.zeros((3, 4), dtype=np.uint8), None)

    projected = SourceSpatialDomainAdapter.for_value(labels).value_in_payload_domain(
        SourceSpatialDomainAdapter.for_value(target)
    )

    assert projected.representation is ObjectLabelRepresentation.DENSE_LABELS
    assert isinstance(projected.labels, np.ndarray)
    assert isinstance(
        projected.variant_data.labels_for_variant(ObjectLabelVariant.UNEDITED),
        np.ndarray,
    )
    np.testing.assert_array_equal(
        projected.labels,
        np.asarray(
            (
                (1, 0, 0, 0),
                (0, 0, 0, 0),
                (0, 0, 0, 2),
            ),
            dtype=np.int32,
        ),
    )
