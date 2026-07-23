"""Exactness gates for stacked CellProfiler object-morphology hot paths."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from skimage.measure import label as relabel

from openhcs.core.runtime_object_label_domains import ObjectLabelDomain
from openhcs.core.runtime_object_labels import (
    ObjectLabelSet,
    ObjectLabelVariantData,
    object_label_dense_array,
)
from openhcs.core.source_image_provenance import SourceImageProvenance
from openhcs.core.source_metadata import SourceVoxelSpacing
from openhcs.core.source_spatial_domain import SourceSpatialDomain
from openhcs.processing.backends.cellprofiler.morphology import (
    MidpointPreservationPolicy,
    MorphologyBackendStrategy,
    StructuringElement,
    build_structuring_element,
    erode_image,
    erode_objects,
    resize_objects_3d,
)


def _raw_function(function: Callable[..., object]) -> Callable[..., object]:
    while hasattr(function, "__wrapped__"):
        function = function.__wrapped__
    return function


def _stacked_labels() -> tuple[np.ndarray, ObjectLabelSet]:
    labels = np.zeros((60, 32, 36), dtype=np.int32)
    labels[3:25, 3:15, 4:16] = 2
    labels[29:56, 17:29, 19:33] = 7
    labels[12:16, 24:28, 5:9] = 11
    provenance = SourceImageProvenance(
        source_path="/plate/A01_z001.tif",
        source_component_metadata={"well": "A01", "channel": "1"},
    )
    return labels, ObjectLabelSet(
        name="Nuclei",
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain(declared_object_ids=(2, 7, 11)),
        source_provenance=provenance,
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(0, 0),
            source_shape_yx=labels.shape[-2:],
        ),
        parent_image_source_voxel_spacing=SourceVoxelSpacing((2.0, 1.0, 1.0)),
    )


def test_erode_objects_preserves_exact_stack_ids_lineage_and_provenance() -> None:
    labels, payload = _stacked_labels()
    footprint = build_structuring_element(StructuringElement.BALL, 2)
    expected = MorphologyBackendStrategy.for_memory_type().erode_labeled_objects(
        labels, footprint
    )
    missing_ids = np.setxor1d(labels, expected)
    expected = MidpointPreservationPolicy.for_footprint(
        footprint
    ).preserve_missing_labels(labels, expected, missing_ids)

    _image, rows, observed, relationship = _raw_function(erode_objects)(
        np.zeros(labels.shape, dtype=np.float32),
        payload,
        structuring_element=StructuringElement.BALL,
        size=2,
    )

    np.testing.assert_array_equal(object_label_dense_array(observed), expected)
    assert observed.name == payload.name
    assert observed.source_provenance == payload.source_provenance
    assert observed.source_spatial_domain == payload.source_spatial_domain
    assert observed.domain == ObjectLabelDomain(declared_object_ids=(2, 7, 11))
    assert rows.columns["input_object_count"] == (3,)
    assert rows.columns["output_object_count"] == (3,)
    assert rows.columns["objects_removed"] == (0,)
    assert relationship.source_ids == (2, 7, 11)
    assert relationship.target_ids == (2, 7, 11)


def test_erode_objects_relabel_branch_keeps_overlap_lineage_and_dense_domain() -> None:
    labels, payload = _stacked_labels()
    footprint = build_structuring_element(StructuringElement.BALL, 2)
    eroded = MorphologyBackendStrategy.for_memory_type().erode_labeled_objects(
        labels, footprint
    )
    expected = relabel(eroded > 0).astype(np.int32)

    _image, rows, observed, relationship = _raw_function(erode_objects)(
        np.zeros(labels.shape, dtype=np.float32),
        payload,
        structuring_element=StructuringElement.BALL,
        size=2,
        preserve_midpoints=False,
        relabel_objects=True,
    )

    np.testing.assert_array_equal(object_label_dense_array(observed), expected)
    expected_ids = tuple(range(1, int(expected.max()) + 1))
    assert observed.domain == ObjectLabelDomain(declared_object_ids=expected_ids)
    assert rows.columns["output_object_count"] == (len(expected_ids),)
    assert relationship.source_ids == (2, 7)
    assert relationship.target_ids == expected_ids


def test_resize_objects_3d_reuses_exact_ids_and_updates_spatial_provenance() -> None:
    labels, payload = _stacked_labels()
    expected = np.repeat(np.repeat(labels, 2, axis=1), 2, axis=2)

    _image, rows, observed, relationship = _raw_function(resize_objects_3d)(
        np.zeros(labels.shape, dtype=np.float32),
        payload,
        factor_x=2.0,
        factor_y=2.0,
        factor_z=1.0,
    )

    np.testing.assert_array_equal(object_label_dense_array(observed), expected)
    assert observed.name == payload.name
    assert observed.source_provenance == payload.source_provenance
    assert observed.source_spatial_domain.origin_yx == (0, 0)
    assert observed.source_spatial_domain.source_shape_yx == expected.shape[-2:]
    assert observed.source_spatial_domain.fill_value == 0
    assert observed.source_spatial_domain.value_name == "Object-label"
    assert observed.parent_image_source_voxel_spacing == SourceVoxelSpacing()
    assert observed.domain == ObjectLabelDomain(declared_object_ids=(2, 7, 11))
    assert rows.columns["original_depth"] == (60,)
    assert rows.columns["new_depth"] == (60,)
    assert rows.columns["new_height"] == (64,)
    assert rows.columns["new_width"] == (72,)
    assert rows.columns["object_count"] == (3,)
    assert relationship.source_ids == (2, 7, 11)
    assert relationship.target_ids == (2, 7, 11)


def test_erode_image_returns_exact_nonempty_stack_without_same_dtype_copy(
    monkeypatch,
) -> None:
    image = np.zeros((60, 32, 36), dtype=np.float32)
    image[5:54, 4:29, 6:31] = np.float32(0.75)

    def preserve_pixels(
        pixels: np.ndarray,
        _structuring_element: StructuringElement | str,
        _size: int,
        _operation: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> np.ndarray:
        return pixels

    import openhcs.processing.backends.cellprofiler.morphology as morphology

    monkeypatch.setattr(morphology, "_morph_image_pixels", preserve_pixels)
    observed = _raw_function(erode_image)(
        image,
        structuring_element=StructuringElement.BALL,
        size=1,
        slice_by_slice=False,
    )

    assert observed is image
    np.testing.assert_array_equal(observed, image)
