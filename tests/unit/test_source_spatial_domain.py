import numpy as np

from openhcs.core.aligned_image_payload import ObjectLabelPayloadSourceSpatialDomainAdapter
from openhcs.core.runtime_semantics import ObjectLabelRepresentation
from openhcs.core.runtime_values import (
    SourceImageObjectLabelBuildRequest,
    SparseIJVLabelRows,
)
from openhcs.core.source_spatial_domain import (
    SourceSpatialDomainAdapter,
    dense_array_in_source_spatial_domain,
)


def test_dense_source_domain_materializes_hwc_color_crop_over_yx_axes():
    image = np.ones((2, 3, 3), dtype=np.uint8)

    materialized = dense_array_in_source_spatial_domain(
        image,
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
