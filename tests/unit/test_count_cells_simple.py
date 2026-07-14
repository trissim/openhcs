import importlib
from dataclasses import replace
from inspect import signature, unwrap

import numpy as np
from skimage.draw import disk

from openhcs.processing.backends.analysis.count_cells_simple import (
    Foreground,
    SimpleCellSegmentationConfig,
    SimpleColocalizationMethod,
    ThresholdMethod,
    count_cells_simple,
    count_cells_simple_dual_channel,
)

count_cells_simple_module = importlib.import_module(
    "openhcs.processing.backends.analysis.count_cells_simple"
)


def _count_cells_simple_impl():
    return unwrap(count_cells_simple)


def _count_cells_simple_dual_channel_impl():
    return unwrap(count_cells_simple_dual_channel)


def _settings(**overrides):
    return replace(SimpleCellSegmentationConfig(), **overrides)


def test_dual_channel_signature_exposes_two_ordered_segmentation_configs():
    parameter_names = list(
        signature(_count_cells_simple_dual_channel_impl()).parameters
    )

    assert parameter_names == [
        "image",
        "channel_1_index",
        "channel_1_settings",
        "channel_2_index",
        "channel_2_settings",
        "colocalization_method",
        "min_overlap_fraction",
        "max_colocalization_distance",
        "return_channel_masks",
    ]
    assert count_cells_simple_dual_channel.__special_outputs__ == (
        "dual_channel_counts",
        "colocalization_masks",
    )


def test_count_cells_simple_area_filter_fast_path_does_not_use_regionprops(monkeypatch):
    image = np.zeros((1, 32, 32), dtype=float)
    rr, cc = disk((10, 10), 4, shape=image.shape[1:])
    image[0, rr, cc] = 1.0
    rr, cc = disk((22, 22), 4, shape=image.shape[1:])
    image[0, rr, cc] = 1.0
    image[0, 0, 0] = 1.0

    def fail_regionprops(_labels):
        raise AssertionError("regionprops should not be needed without shape filtering")

    monkeypatch.setattr(count_cells_simple_module, "regionprops", fail_regionprops)

    _, results, masks = _count_cells_simple_impl()(
        image,
        segmentation_settings=_settings(
            threshold_method=ThresholdMethod.MANUAL,
            threshold=0.5,
            min_size=20,
            max_size=200,
            max_eccentricity=1.0,
        ),
    )

    assert results == [{"slice_index": 0, "cell_count": 2}]
    assert set(np.unique(masks[0])) == {0, 1, 2}


def test_count_cells_simple_filters_eccentricity_after_size_filter():
    image = np.zeros((1, 64, 64), dtype=float)
    rr, cc = disk((24, 24), 5, shape=image.shape[1:])
    image[0, rr, cc] = 1.0
    image[0, 42:45, 10:40] = 1.0

    _, unfiltered_results, unfiltered_masks = _count_cells_simple_impl()(
        image,
        segmentation_settings=_settings(
            threshold_method=ThresholdMethod.MANUAL,
            threshold=0.5,
            min_size=20,
            max_size=200,
            max_eccentricity=1.0,
        ),
    )
    _, filtered_results, filtered_masks = _count_cells_simple_impl()(
        image,
        segmentation_settings=_settings(
            threshold_method=ThresholdMethod.MANUAL,
            threshold=0.5,
            min_size=20,
            max_size=200,
            max_eccentricity=0.9,
        ),
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
        segmentation_settings=_settings(
            threshold_method=ThresholdMethod.MANUAL,
            threshold=0.5,
            min_size=50,
            max_size=220,
            watershed_large_objects=False,
        ),
    )
    _, split_results, split_masks = _count_cells_simple_impl()(
        image,
        segmentation_settings=_settings(
            threshold_method=ThresholdMethod.MANUAL,
            threshold=0.5,
            min_size=50,
            max_size=220,
            watershed_large_objects=True,
            watershed_min_distance=5,
        ),
    )
    _, capped_results, capped_masks = _count_cells_simple_impl()(
        image,
        segmentation_settings=_settings(
            threshold_method=ThresholdMethod.MANUAL,
            threshold=0.5,
            min_size=50,
            max_size=220,
            watershed_large_objects=True,
            watershed_max_size=300,
            watershed_min_distance=5,
        ),
    )

    assert unsplit_results == [{"slice_index": 0, "cell_count": 0}]
    assert set(np.unique(unsplit_masks[0])) == {0}
    assert split_results == [{"slice_index": 0, "cell_count": 2}]
    assert set(np.unique(split_masks[0])) == {0, 1, 2}
    assert capped_results == [{"slice_index": 0, "cell_count": 0}]
    assert set(np.unique(capped_masks[0])) == {0}


def test_count_cells_simple_watershed_min_size_separates_split_trigger_from_filter():
    image = np.zeros((1, 80, 80), dtype=float)
    rr, cc = disk((32, 32), 12, shape=image.shape[1:])
    image[0, rr, cc] = 1.0
    rr, cc = disk((48, 44), 12, shape=image.shape[1:])
    image[0, rr, cc] = 1.0

    _, unsplit_results, unsplit_masks = _count_cells_simple_impl()(
        image,
        segmentation_settings=_settings(
            threshold_method=ThresholdMethod.MANUAL,
            threshold=0.5,
            min_size=20,
            max_size=900,
            watershed_large_objects=True,
            watershed_min_distance=1,
            watershed_footprint_size=5,
        ),
    )
    _, split_results, split_masks = _count_cells_simple_impl()(
        image,
        segmentation_settings=_settings(
            threshold_method=ThresholdMethod.MANUAL,
            threshold=0.5,
            min_size=20,
            max_size=900,
            watershed_large_objects=True,
            watershed_min_size=100,
            watershed_min_distance=1,
            watershed_footprint_size=5,
        ),
    )

    assert unsplit_results == [{"slice_index": 0, "cell_count": 1}]
    assert set(np.unique(unsplit_masks[0])) == {0, 1}
    assert split_results == [{"slice_index": 0, "cell_count": 2}]
    assert set(np.unique(split_masks[0])) == {0, 1, 2}


def test_count_cells_simple_dual_channel_reports_overlap_colocalization():
    image = np.zeros((2, 64, 64), dtype=float)
    rr, cc = disk((20, 20), 5, shape=image.shape[1:])
    image[0, rr, cc] = 1.0
    image[1, rr, cc] = 1.0

    rr, cc = disk((45, 45), 5, shape=image.shape[1:])
    image[0, rr, cc] = 1.0
    rr, cc = disk((20, 45), 5, shape=image.shape[1:])
    image[1, rr, cc] = 1.0

    output, results, masks = _count_cells_simple_dual_channel_impl()(
        image,
        channel_1_settings=_settings(
            threshold_method=ThresholdMethod.MANUAL,
            threshold=0.5,
            min_size=20,
            max_size=200,
        ),
        channel_2_settings=_settings(
            threshold_method=ThresholdMethod.MANUAL,
            threshold=0.5,
            min_size=20,
            max_size=200,
        ),
        colocalization_method=SimpleColocalizationMethod.OVERLAP,
        min_overlap_fraction=0.5,
        return_channel_masks=True,
    )

    assert output is image
    assert results == [
        {
            "channel_1_index": 0,
            "channel_2_index": 1,
            "channel_1_count": 2,
            "channel_2_count": 2,
            "colocalized_count": 1,
            "channel_1_only_count": 1,
            "channel_2_only_count": 1,
            "channel_1_colocalized_percent": 50.0,
            "channel_2_colocalized_percent": 50.0,
            "colocalization_method": "overlap",
            "mean_colocalization_distance": 0.0,
            "mean_overlap_fraction": 1.0,
        }
    ]
    assert len(masks) == 3
    assert set(np.unique(masks[0])) == {0, 1, 2}
    assert set(np.unique(masks[1])) == {0, 1, 2}
    assert set(np.unique(masks[2])) == {0, 1}


def test_count_cells_simple_dual_channel_uses_independent_channel_settings():
    image = np.empty((2, 64, 64), dtype=float)
    image[0] = 0.0
    image[1] = 1.0

    shared_rr, shared_cc = disk((20, 20), 5, shape=image.shape[1:])
    image[0, shared_rr, shared_cc] = 1.0
    image[1, shared_rr, shared_cc] = 0.0

    channel_1_rr, channel_1_cc = disk((45, 45), 5, shape=image.shape[1:])
    image[0, channel_1_rr, channel_1_cc] = 1.0
    channel_2_rr, channel_2_cc = disk((20, 45), 5, shape=image.shape[1:])
    image[1, channel_2_rr, channel_2_cc] = 0.0

    _, results, masks = _count_cells_simple_dual_channel_impl()(
        image,
        channel_1_settings=_settings(
            threshold_method=ThresholdMethod.MANUAL,
            threshold=0.5,
            foreground=Foreground.BRIGHT,
            min_size=20,
            max_size=200,
        ),
        channel_2_settings=_settings(
            threshold_method=ThresholdMethod.MANUAL,
            threshold=0.5,
            foreground=Foreground.DARK,
            min_size=20,
            max_size=200,
        ),
        colocalization_method=SimpleColocalizationMethod.OVERLAP,
        min_overlap_fraction=0.5,
        return_channel_masks=True,
    )

    assert results[0]["channel_1_count"] == 2
    assert results[0]["channel_2_count"] == 2
    assert results[0]["colocalized_count"] == 1
    assert [set(np.unique(mask)) for mask in masks] == [
        {0, 1, 2},
        {0, 1, 2},
        {0, 1},
    ]
