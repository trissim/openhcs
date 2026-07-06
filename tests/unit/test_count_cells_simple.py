from inspect import unwrap

import numpy as np
from skimage.draw import disk

from openhcs.processing.backends.analysis.count_cells_simple import (
    ThresholdMethod,
    count_cells_simple,
)


def _count_cells_simple_impl():
    return unwrap(count_cells_simple)


def test_count_cells_simple_filters_eccentricity_after_size_filter():
    image = np.zeros((1, 64, 64), dtype=float)
    rr, cc = disk((24, 24), 5, shape=image.shape[1:])
    image[0, rr, cc] = 1.0
    image[0, 42:45, 10:40] = 1.0

    _, unfiltered_results, unfiltered_masks = _count_cells_simple_impl()(
        image,
        threshold_method=ThresholdMethod.MANUAL,
        threshold=0.5,
        min_size=20,
        max_size=200,
        max_eccentricity=1.0,
    )
    _, filtered_results, filtered_masks = _count_cells_simple_impl()(
        image,
        threshold_method=ThresholdMethod.MANUAL,
        threshold=0.5,
        min_size=20,
        max_size=200,
        max_eccentricity=0.9,
    )

    assert unfiltered_results == [{"slice_index": 0, "cell_count": 2}]
    assert set(np.unique(unfiltered_masks[0])) == {0, 1, 2}
    assert filtered_results == [{"slice_index": 0, "cell_count": 1}]
    assert set(np.unique(filtered_masks[0])) == {0, 1}


def test_count_cells_simple_watersheds_large_objects_before_size_filter():
    image = np.zeros((1, 64, 64), dtype=float)
    rr, cc = disk((32, 26), 8, shape=image.shape[1:])
    image[0, rr, cc] = 1.0
    rr, cc = disk((32, 38), 8, shape=image.shape[1:])
    image[0, rr, cc] = 1.0

    _, unsplit_results, unsplit_masks = _count_cells_simple_impl()(
        image,
        threshold_method=ThresholdMethod.MANUAL,
        threshold=0.5,
        min_size=50,
        max_size=220,
        watershed_large_objects=False,
    )
    _, split_results, split_masks = _count_cells_simple_impl()(
        image,
        threshold_method=ThresholdMethod.MANUAL,
        threshold=0.5,
        min_size=50,
        max_size=220,
        watershed_large_objects=True,
        watershed_min_distance=5,
    )

    assert unsplit_results == [{"slice_index": 0, "cell_count": 0}]
    assert set(np.unique(unsplit_masks[0])) == {0}
    assert split_results == [{"slice_index": 0, "cell_count": 2}]
    assert set(np.unique(split_masks[0])) == {0, 1, 2}
