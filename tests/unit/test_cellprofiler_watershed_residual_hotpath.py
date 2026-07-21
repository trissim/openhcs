from __future__ import annotations

import numpy as np
import pytest
import skimage.measure
import skimage.segmentation

from openhcs.core.runtime_image_values import ImageMetadataPayload, ImagePayloadMetadata
from openhcs.core.runtime_object_label_domains import ObjectLabelDomainScope
from openhcs.core.runtime_object_labels import object_label_dense_array
from openhcs.processing.backends.cellprofiler.structuring_elements import (
    StructuringElement,
)
from openhcs.processing.backends.cellprofiler.watershed import (
    CellProfiler4InitialWatershedStrategy,
    WatershedDeclumpMethod,
    WatershedMethod,
    WatershedParameters,
    WatershedSeedMethod,
    watershed_cellprofiler4,
    watershed_connected_components,
)


def test_empty_marker_mask_deletes_segmentation_and_labeling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.ones((5, 17, 19), dtype=np.float32)
    metadata = ImagePayloadMetadata.for_array(
        image,
        source_path="/plate/A01_s001_w1.tiff",
    )
    image_payload = ImageMetadataPayload(data=image, metadata=metadata)
    markers = np.zeros(image.shape, dtype=np.int32)
    markers[2, 8, 9] = 70_001
    mask = np.zeros(image.shape, dtype=bool)

    def fail_external_call(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("empty marker domains must skip external labeling")

    monkeypatch.setattr(skimage.segmentation, "watershed", fail_external_call)
    monkeypatch.setattr(skimage.measure, "label", fail_external_call)

    source_image, rows, label_value = watershed_cellprofiler4(
        image_payload,
        topology_inputs=(markers, mask),
        watershed_method=WatershedMethod.MARKERS,
        use_advanced_settings=False,
    )

    labels = object_label_dense_array(label_value)
    assert source_image is image_payload
    assert labels.shape == image.shape
    assert labels.dtype == np.int32
    assert not np.any(labels)
    assert label_value.domain.scope is ObjectLabelDomainScope.PAYLOAD
    assert label_value.domain.declared_object_count == 0
    assert label_value.domain.declared_object_ids == ()
    assert label_value.source_provenance == metadata.source_provenance
    assert label_value.source_spatial_domain.source_shape_yx == image.shape[-2:]
    assert label_value.parent_image_source_voxel_spacing == metadata.source_voxel_spacing
    assert rows.row_mappings() == (
        {"slice_index": 0, "object_count": 0, "mean_area": 0.0},
    )


@pytest.mark.parametrize("connectivity", (1, 2, 3))
def test_nonempty_sparse_high_markers_preserve_connectivity_reference(
    connectivity: int,
) -> None:
    image = np.zeros((7, 31, 33), dtype=bool)
    image[:, 2:-2, 2:-2] = True
    mask = image.copy()
    mask[:, 14:17, 15:18] = False
    markers = np.zeros(image.shape, dtype=np.int32)
    markers[1, 5, 5] = 70_001
    markers[1, 25, 5] = 70_001
    markers[5, 25, 27] = 3

    initial_labels = skimage.segmentation.watershed(
        image=image,
        markers=markers,
        mask=mask,
        connectivity=connectivity,
        compactness=0.0,
        watershed_line=False,
    )
    expected_labels = skimage.measure.label(initial_labels).astype(
        np.int32,
        copy=False,
    )
    parameters = WatershedParameters.from_settings(
        image_ndim=image.ndim,
        watershed_method=WatershedMethod.MARKERS,
        declump_method=WatershedDeclumpMethod.SHAPE,
        seed_method=WatershedSeedMethod.LOCAL,
        use_advanced_settings=True,
        max_seeds=-1,
        downsample=1,
        min_distance=1,
        min_intensity=0.0,
        footprint=8,
        connectivity=connectivity,
        compactness=0.0,
        exclude_border=False,
        watershed_line=False,
        gaussian_sigma=0.0,
        structuring_element=StructuringElement.DISK,
        structuring_element_size=1,
    )

    actual_initial_labels, mask_source = (
        CellProfiler4InitialWatershedStrategy.for_enum_member(
            WatershedMethod.MARKERS
        ).labels(
            image,
            None,
            markers,
            mask,
            parameters,
        )
    )
    labels = watershed_connected_components(actual_initial_labels)

    np.testing.assert_array_equal(actual_initial_labels, initial_labels)
    assert mask_source is image
    np.testing.assert_array_equal(labels, expected_labels)
    assert labels.dtype == np.int32

    if connectivity != 1:
        return

    _, rows, label_value = watershed_cellprofiler4(
        image,
        topology_inputs=(markers, mask),
        watershed_method=WatershedMethod.MARKERS,
        use_advanced_settings=False,
    )

    public_labels = object_label_dense_array(label_value)
    object_count = int(expected_labels.max(initial=0))
    np.testing.assert_array_equal(public_labels, expected_labels)
    assert public_labels.dtype == np.int32
    assert label_value.domain.scope is ObjectLabelDomainScope.PAYLOAD
    assert label_value.domain.explicit_id_domain() == tuple(
        range(1, object_count + 1)
    )
    assert rows.row_mappings() == (
        {
            "slice_index": 0,
            "object_count": object_count,
            "mean_area": np.count_nonzero(expected_labels) / object_count,
        },
    )
