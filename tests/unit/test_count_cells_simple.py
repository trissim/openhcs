import importlib
from dataclasses import fields, replace
from inspect import signature, unwrap

import numpy as np
from skimage.draw import disk

from openhcs.processing.backends.analysis.count_cells_simple import (
    Foreground,
    MetaXpressW2Settings,
    MetaXpressWavelengthSettings,
    SimpleCellSegmentationConfig,
    StainedArea,
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


def test_dual_channel_signature_exposes_only_metaxpress_scoring_controls():
    parameters = signature(_count_cells_simple_dual_channel_impl()).parameters
    special_inputs = count_cells_simple_dual_channel.__special_inputs__
    exposed_names = [name for name in parameters if name not in special_inputs]

    assert exposed_names == [
        "image",
        "w1",
        "w2",
        "minimum_stained_area",
    ]
    assert special_inputs == {"pixel_size": True}
    assert parameters["pixel_size"].annotation._ui_hidden is True
    assert [field.name for field in fields(MetaXpressWavelengthSettings)] == [
        "channel_index",
        "approx_min_width",
        "approx_max_width",
        "intensity_above_local_background",
    ]
    assert [field.name for field in fields(MetaXpressW2Settings)] == [
        "channel_index",
        "approx_min_width",
        "approx_max_width",
        "intensity_above_local_background",
        "stained_area",
    ]
    assert [choice.value for choice in StainedArea] == [
        "nucleus",
        "nucleus and cytoplasm",
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


def _metaxpress_settings(**overrides):
    return replace(MetaXpressWavelengthSettings(), **overrides)


def _metaxpress_w2_settings(**overrides):
    return replace(MetaXpressW2Settings(), **overrides)


def test_dual_channel_scores_w2_positive_cells_by_minimum_stained_area():
    image = np.full((2, 64, 64), 100.0)
    for center in ((20, 20), (45, 45)):
        rr, cc = disk(center, 5, shape=image.shape[1:])
        image[0, rr, cc] = 1000.0

    rr, cc = disk((20, 20), 4, shape=image.shape[1:])
    image[1, rr, cc] = 700.0

    output, results, masks = _count_cells_simple_dual_channel_impl()(
        image,
        w1=_metaxpress_settings(
            channel_index=0,
            approx_min_width=6.0,
            approx_max_width=14.0,
            intensity_above_local_background=300.0,
        ),
        w2=_metaxpress_w2_settings(
            channel_index=1,
            approx_min_width=4.0,
            approx_max_width=14.0,
            intensity_above_local_background=200.0,
            stained_area=StainedArea.NUCLEUS,
        ),
        minimum_stained_area=20.0,
        pixel_size=1.0,
    )

    assert output is image
    assert results == [
        {
            "w1_channel_index": 0,
            "w2_channel_index": 1,
            "total_cell_count": 2,
            "w2_positive_cell_count": 1,
            "w2_negative_cell_count": 1,
            "w2_positive_percent": 50.0,
            "w2_stained_area": "nucleus",
            "minimum_stained_area": 20.0,
            "all_w2_mean_stained_area": 22.5,
            "positive_w2_mean_stained_area": 45.0,
        }
    ]
    assert [(mask.source_index, mask.role) for mask in masks.masks] == [
        (0, "w1_nuclei"),
        (1, "w2_stain"),
    ]
    assert [set(np.unique(mask.mask)) for mask in masks.masks] == [
        {0, 1, 2},
        {0, 1},
    ]
    assert masks.masks[0].label_metadata == {
        1: {"w2_positive": True, "w2_stained_area_um2": 45.0},
        2: {"w2_positive": False, "w2_stained_area_um2": 0.0},
    }

    _, stricter_results, _ = _count_cells_simple_dual_channel_impl()(
        image,
        w1=_metaxpress_settings(
            channel_index=0,
            approx_min_width=6.0,
            approx_max_width=14.0,
            intensity_above_local_background=300.0,
        ),
        w2=_metaxpress_w2_settings(
            channel_index=1,
            approx_min_width=4.0,
            approx_max_width=14.0,
            intensity_above_local_background=200.0,
            stained_area=StainedArea.NUCLEUS,
        ),
        minimum_stained_area=46.0,
        pixel_size=1.0,
    )
    assert stricter_results[0]["w2_positive_cell_count"] == 0


def test_w2_nucleus_and_cytoplasm_scores_stain_outside_the_nucleus():
    image = np.full((2, 64, 64), 100.0)
    rr, cc = disk((32, 32), 4, shape=image.shape[1:])
    image[0, rr, cc] = 1000.0

    outer_rr, outer_cc = disk((32, 32), 8, shape=image.shape[1:])
    image[1, outer_rr, outer_cc] = 700.0
    image[1, rr, cc] = 100.0

    w1 = _metaxpress_settings(
        channel_index=0,
        approx_min_width=5.0,
        approx_max_width=10.0,
        intensity_above_local_background=300.0,
    )
    w2 = _metaxpress_w2_settings(
        channel_index=1,
        approx_min_width=6.0,
        approx_max_width=24.0,
        intensity_above_local_background=200.0,
        stained_area=StainedArea.NUCLEUS,
    )

    _, nucleus_results, _ = _count_cells_simple_dual_channel_impl()(
        image,
        w1=w1,
        w2=w2,
        minimum_stained_area=20.0,
        pixel_size=1.0,
    )
    _, whole_cell_results, masks = _count_cells_simple_dual_channel_impl()(
        image,
        w1=w1,
        w2=replace(w2, stained_area=StainedArea.NUCLEUS_AND_CYTOPLASM),
        minimum_stained_area=20.0,
        pixel_size=1.0,
    )

    assert nucleus_results[0]["w2_positive_cell_count"] == 0
    assert whole_cell_results[0]["w2_positive_cell_count"] == 1
    assert whole_cell_results[0]["w2_stained_area"] == "nucleus and cytoplasm"
    assert set(np.unique(masks.masks[1].mask)) == {0, 1}


def test_width_settings_derive_watershed_for_touching_w1_nuclei():
    image = np.full((2, 64, 64), 100.0)
    for center in ((32, 27), (32, 37)):
        rr, cc = disk(center, 6, shape=image.shape[1:])
        image[0, rr, cc] = 1000.0

    _, results, masks = _count_cells_simple_dual_channel_impl()(
        image,
        w1=_metaxpress_settings(
            channel_index=0,
            approx_min_width=6.0,
            approx_max_width=12.0,
            intensity_above_local_background=300.0,
        ),
        w2=_metaxpress_w2_settings(
            channel_index=1,
            approx_min_width=4.0,
            approx_max_width=12.0,
            intensity_above_local_background=2000.0,
        ),
        minimum_stained_area=10.0,
        pixel_size=1.0,
    )

    assert results[0]["total_cell_count"] == 2
    assert results[0]["w2_positive_cell_count"] == 0
    assert set(np.unique(masks.masks[0].mask)) == {0, 1, 2}
