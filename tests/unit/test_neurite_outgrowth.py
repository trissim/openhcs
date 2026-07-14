from dataclasses import fields
from inspect import signature, unwrap

import numpy as np
import pytest
from skimage.draw import disk, line

from openhcs.processing.backends.analysis.neurite_outgrowth import (
    MetaXpressCellBodySettings,
    MetaXpressNuclearSettings,
    MetaXpressOutgrowthSettings,
    NeuriteIllumination,
    neurite_outgrowth_metaxpress,
)


def _implementation():
    return unwrap(neurite_outgrowth_metaxpress)


def _draw_fluorescent_neuron(*, crossing=False, branched=False):
    image = np.zeros((1, 128, 128), dtype=np.uint16)
    rows, columns = disk((64, 20), 9, shape=image.shape[1:])
    image[0, rows, columns] = 1000
    rows, columns = line(64, 28, 64, 115)
    image[0, rows, columns] = 700
    if crossing:
        rows, columns = line(20, 75, 110, 75)
        image[0, rows, columns] = 700
    if branched:
        rows, columns = line(64, 75, 35, 105)
        image[0, rows, columns] = 700
        rows, columns = line(64, 75, 93, 105)
        image[0, rows, columns] = 700
    return image


def _cell_body_settings():
    return MetaXpressCellBodySettings(
        approximate_max_width=30.0,
        minimum_area=100.0,
        intensity_above_local_background=100.0,
    )


def _outgrowth_settings(significant_threshold=20.0):
    return MetaXpressOutgrowthSettings(
        maximum_width=3.0,
        intensity_above_local_background=100.0,
        minimum_cell_growth_to_log_as_significant=significant_threshold,
    )


def test_signature_exposes_documented_metaxpress_controls_only():
    parameters = signature(_implementation()).parameters
    exposed = [
        name
        for name in parameters
        if name not in neurite_outgrowth_metaxpress.__special_inputs__
    ]

    assert exposed == [
        "image",
        "neurite_channel_index",
        "illumination",
        "cell_body",
        "outgrowth",
        "use_nuclear_stain",
        "nuclear_stain",
    ]
    assert neurite_outgrowth_metaxpress.__special_inputs__ == {"pixel_size": True}
    assert parameters["pixel_size"].annotation._ui_hidden is True
    assert [field.name for field in fields(MetaXpressCellBodySettings)] == [
        "approximate_max_width",
        "minimum_area",
        "intensity_above_local_background",
    ]
    assert [field.name for field in fields(MetaXpressOutgrowthSettings)] == [
        "maximum_width",
        "intensity_above_local_background",
        "minimum_cell_growth_to_log_as_significant",
    ]
    assert [field.name for field in fields(MetaXpressNuclearSettings)] == [
        "channel_index",
        "approx_min_width",
        "approx_max_width",
        "intensity_above_local_background",
    ]
    assert [choice.value for choice in NeuriteIllumination] == [
        "fluorescence",
        "transmission",
    ]
    assert neurite_outgrowth_metaxpress.__special_outputs__ == (
        "neurite_outgrowth_summary",
        "neurite_outgrowth_cells",
        "neurite_outgrowth_masks",
    )


def test_branch_metrics_and_significant_threshold_is_scoring_only():
    image = _draw_fluorescent_neuron(branched=True)
    low_threshold = _implementation()(
        image,
        cell_body=_cell_body_settings(),
        outgrowth=_outgrowth_settings(20.0),
        pixel_size=1.0,
    )
    high_threshold = _implementation()(
        image,
        cell_body=_cell_body_settings(),
        outgrowth=_outgrowth_settings(1000.0),
        pixel_size=1.0,
    )

    low_summary = low_threshold[1][0]
    low_cell = low_threshold[2][0]
    high_summary = high_threshold[1][0]
    high_cell = high_threshold[2][0]

    assert low_summary["number_of_cells"] == 1
    assert low_summary["total_processes"] == 1
    assert low_summary["total_branches"] == 1
    assert low_cell["mean_process_length_um"] == pytest.approx(
        low_cell["total_outgrowth_um"]
    )
    assert 0.0 < low_cell["straightness"] <= 1.0
    assert low_cell["significant_growth"] is True
    assert high_cell["significant_growth"] is False
    assert high_summary["cells_significant_growth"] == 0
    assert high_cell["total_outgrowth_um"] == pytest.approx(
        low_cell["total_outgrowth_um"]
    )
    assert np.array_equal(
        low_threshold[3].masks[1].mask,
        high_threshold[3].masks[1].mask,
    )


def test_straight_crossover_is_not_a_branch_or_foreign_outgrowth():
    image = _draw_fluorescent_neuron(crossing=True)
    _, summary_rows, cell_rows, masks = _implementation()(
        image,
        cell_body=_cell_body_settings(),
        outgrowth=_outgrowth_settings(),
        pixel_size=1.0,
    )

    summary = summary_rows[0]
    cell = cell_rows[0]
    neurite_mask = masks.masks[1].mask
    assert summary["resolved_crossovers"] == 1
    assert summary["total_branches"] == 0
    assert summary["total_processes"] == 1
    assert cell["straightness"] == pytest.approx(1.0)
    assert 80.0 <= cell["total_outgrowth_um"] <= 90.0
    assert np.count_nonzero(neurite_mask[:, 75]) == 1
    assert np.count_nonzero(neurite_mask[64, :]) > 80


def test_optional_nucleus_is_segmented_and_aligned_to_its_channel():
    neurite_image = _draw_fluorescent_neuron()[0]
    nucleus_image = np.zeros_like(neurite_image)
    rows, columns = disk((64, 20), 5, shape=nucleus_image.shape)
    nucleus_image[rows, columns] = 1200
    image = np.stack((neurite_image, nucleus_image))

    _, summary_rows, _, masks = _implementation()(
        image,
        cell_body=_cell_body_settings(),
        outgrowth=_outgrowth_settings(),
        use_nuclear_stain=True,
        nuclear_stain=MetaXpressNuclearSettings(
            channel_index=1,
            approx_min_width=6.0,
            approx_max_width=14.0,
            intensity_above_local_background=200.0,
        ),
        pixel_size=1.0,
    )

    assert summary_rows[0]["number_of_cells"] == 1
    assert summary_rows[0]["nuclear_channel_index"] == 1
    assert [(mask.role, mask.source_index) for mask in masks.masks] == [
        ("cell_bodies", 0),
        ("neurite_outgrowth", 0),
        ("nuclei", 1),
    ]
    assert masks.masks[2].mask.max() == 1


def test_transmission_mode_detects_dark_cell_and_neurite():
    fluorescence = _draw_fluorescent_neuron()[0]
    transmission = np.full(fluorescence.shape, 2000, dtype=np.uint16)
    transmission[fluorescence > 0] = 500

    _, summary_rows, cell_rows, _ = _implementation()(
        transmission[None, ...],
        illumination=NeuriteIllumination.TRANSMISSION,
        cell_body=_cell_body_settings(),
        outgrowth=_outgrowth_settings(),
        pixel_size=1.0,
    )

    assert summary_rows[0]["number_of_cells"] == 1
    assert summary_rows[0]["total_processes"] == 1
    assert cell_rows[0]["total_outgrowth_um"] > 80.0


def test_rejects_a_plain_2d_image_because_channels_must_be_explicit():
    with pytest.raises(ValueError, match="2D channel stack"):
        _implementation()(np.zeros((32, 32)), pixel_size=1.0)
