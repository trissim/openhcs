from __future__ import annotations

import numpy as np
import pytest
import skimage.measure
import skimage.segmentation
import skimage.transform

from openhcs.core.runtime_object_labels import object_label_dense_array
from openhcs.processing.backends.cellprofiler.watershed import (
    WatershedMethod,
    watershed_cellprofiler4,
    watershed_resize_labels,
)


def test_marker_watershed_nonempty_mask_matches_connected_component_reference() -> None:
    image = np.zeros((5, 17, 19), dtype=bool)
    image[:, 2:15, 2:17] = True
    mask = image.copy()
    mask[:, 7:10, 8:11] = False
    markers = np.zeros(image.shape, dtype=np.int32)
    markers[1, 5, 5] = 7
    markers[3, 12, 14] = 2

    initial_labels = skimage.segmentation.watershed(
        image=image,
        markers=markers,
        mask=mask,
        connectivity=1,
        compactness=0.0,
        watershed_line=False,
    )
    expected_labels = skimage.measure.label(initial_labels).astype(
        np.int32,
        copy=False,
    )

    _, stats, label_value = watershed_cellprofiler4(
        image,
        topology_inputs=(markers, mask),
        watershed_method=WatershedMethod.MARKERS,
        use_advanced_settings=False,
    )

    labels = object_label_dense_array(label_value)
    assert np.count_nonzero(markers) == 2
    assert np.count_nonzero(mask) > 0
    np.testing.assert_array_equal(labels, expected_labels)
    (stats_row,) = stats.row_mappings()
    object_count = int(expected_labels.max(initial=0))
    assert stats_row["object_count"] == object_count
    assert stats_row["mean_area"] == pytest.approx(
        np.count_nonzero(expected_labels) / object_count,
    )


def test_integer_label_resize_reuses_exact_resize_result(monkeypatch) -> None:
    resized = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)

    def fake_resize(
        labels,
        output_shape,
        *,
        mode,
        order,
        preserve_range,
    ):
        assert labels.shape == (1, 2, 2)
        assert output_shape == resized.shape
        assert mode == "edge"
        assert order == 0
        assert preserve_range is True
        return resized

    monkeypatch.setattr(skimage.transform, "resize", fake_resize)

    result = watershed_resize_labels(
        np.ones((1, 2, 2), dtype=np.uint16),
        resized.shape,
    )

    assert result is resized


def test_float_label_resize_rounds_in_place_before_uint16_cast(monkeypatch) -> None:
    resized = np.array([[[0.49, 0.5], [1.5, 2.51]]], dtype=np.float32)
    expected = np.rint(resized).astype(np.uint16)

    monkeypatch.setattr(
        skimage.transform,
        "resize",
        lambda *args, **kwargs: resized,
    )

    result = watershed_resize_labels(
        np.ones((1, 1, 1), dtype=np.float32),
        resized.shape,
    )

    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(resized, np.rint(resized))
