import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure

from benchmark.cellprofiler_library.functions.expandorshrinkobjects import (
    expand_or_shrink_objects,
)
from openhcs.core.config import DtypeConfig
from openhcs.core.runtime_semantics import ObjectLabelDomain
from openhcs.core.runtime_values import ObjectLabelPayload


def test_expand_or_shrink_objects_accepts_generated_mode_literals():
    image = np.zeros((5, 5), dtype=float)
    labels = np.zeros((5, 5), dtype=np.int32)
    labels[1:4, 1:4] = 1

    _, result = expand_or_shrink_objects(
        image,
        labels,
        mode="shrink_defined_pixels",
        iterations=1,
        fill_holes=False,
        dtype_config=DtypeConfig(),
    )

    assert isinstance(result, ObjectLabelPayload)
    assert result.labels.astype(bool).sum() == 1
    assert result.labels[2, 2] == 1


def test_expand_or_shrink_objects_expands_by_euclidean_distance():
    image = np.zeros((9, 9), dtype=float)
    labels = np.zeros((9, 9), dtype=np.int32)
    labels[4, 4] = 1

    _, result = expand_or_shrink_objects(
        image,
        labels,
        mode="expand_defined_pixels",
        iterations=3,
        dtype_config=DtypeConfig(),
    )

    yy, xx = np.indices(labels.shape)
    expected = ((yy - 4) ** 2 + (xx - 4) ** 2 <= 9).astype(np.int32)
    assert result.labels.dtype == np.int32
    assert np.array_equal(result.labels, expected)


def test_expand_or_shrink_objects_expands_stacked_labels_planewise():
    image = np.zeros((2, 9, 9), dtype=float)
    labels = np.zeros((2, 9, 9), dtype=np.int32)
    labels[0, 4, 4] = 1
    labels[1, 2, 6] = 2

    _, result = expand_or_shrink_objects(
        image,
        labels,
        mode="expand_defined_pixels",
        iterations=2,
        dtype_config=DtypeConfig(),
    )

    yy, xx = np.indices(labels.shape[-2:])
    expected = np.zeros_like(labels, dtype=np.int32)
    expected[0] = ((yy - 4) ** 2 + (xx - 4) ** 2 <= 4).astype(np.int32)
    expected[1] = (((yy - 2) ** 2 + (xx - 6) ** 2 <= 4) * 2).astype(np.int32)
    assert result.labels.dtype == np.int32
    np.testing.assert_array_equal(result.labels, expected)


def test_expand_or_shrink_objects_declares_output_label_extent():
    image = np.zeros((9, 9), dtype=float)
    labels = np.zeros((9, 9), dtype=np.int32)
    labels[4, 4] = 4
    payload = ObjectLabelPayload(labels=labels, domain=ObjectLabelDomain(declared_object_count=9))

    _, result = expand_or_shrink_objects(
        image,
        payload,
        mode="expand_defined_pixels",
        iterations=1,
        dtype_config=DtypeConfig(),
    )

    assert result.domain.declared_object_count == 4
    assert result.domain.declared_object_ids == ()


def test_expand_or_shrink_objects_shrinks_labels_like_per_object_erosion():
    image = np.zeros((8, 9), dtype=float)
    labels = np.zeros((8, 9), dtype=np.int32)
    labels[1:6, 1:6] = 1
    labels[2:7, 6:8] = 2
    labels[0:2, 0:2] = 3

    _, result = expand_or_shrink_objects(
        image,
        labels,
        mode="shrink_defined_pixels",
        iterations=2,
        fill_holes=False,
        dtype_config=DtypeConfig(),
    )

    expected = np.zeros_like(labels)
    struct = generate_binary_structure(2, 1)
    for label_id in (1, 2, 3):
        eroded = binary_erosion(labels == label_id, structure=struct, iterations=2)
        expected[eroded] = label_id
    assert result.labels.dtype == np.int32
    assert np.array_equal(result.labels, expected)
