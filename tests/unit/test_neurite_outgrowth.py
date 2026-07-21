from dataclasses import fields
from inspect import signature, unwrap

import numpy as np
import pytest
from skimage.draw import disk, line

from openhcs.core.artifacts import (
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.callable_contract import CallableContract
from openhcs.processing.backends.analysis.neurite_outgrowth import (
    MetaXpressCellBodySettings,
    MetaXpressNuclearSettings,
    MetaXpressOutgrowthSettings,
    NeuriteIllumination,
    neurite_outgrowth_metaxpress,
)


def _implementation():
    return unwrap(neurite_outgrowth_metaxpress)


def _rows(rows):
    return rows.row_mappings()


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
    contract = CallableContract.from_callable(neurite_outgrowth_metaxpress)
    runtime_artifact_parameter_names = contract.artifact_inputs.names()
    exposed = [
        name
        for name in parameters
        if name not in runtime_artifact_parameter_names
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
    assert runtime_artifact_parameter_names == ("pixel_size",)
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
    assert contract.artifact_outputs.names() == (
        "neurite_outgrowth_summary",
        "neurite_outgrowth_cells",
        "cell_bodies",
        "neurite_outgrowth",
        "nuclei",
    )
    summary_spec, cell_spec, body_spec, neurite_spec, nuclei_spec = (
        contract.artifact_outputs
    )
    assert summary_spec.artifact_type is MeasurementsArtifactType
    assert cell_spec.artifact_type is MeasurementsArtifactType
    assert cell_spec.relations[0].measurement_subject().name == body_spec.name
    assert all(
        spec.artifact_type is ObjectLabelsArtifactType
        for spec in (body_spec, neurite_spec, nuclei_spec)
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

    low_summary = _rows(low_threshold[1])[0]
    low_cell = _rows(low_threshold[2])[0]
    high_summary = _rows(high_threshold[1])[0]
    high_cell = _rows(high_threshold[2])[0]

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
        low_threshold[4],
        high_threshold[4],
    )


def test_straight_crossover_is_not_a_branch_or_foreign_outgrowth():
    image = _draw_fluorescent_neuron(crossing=True)
    _, summary_rows, cell_rows, _, neurite_labels, _ = _implementation()(
        image,
        cell_body=_cell_body_settings(),
        outgrowth=_outgrowth_settings(),
        pixel_size=1.0,
    )

    summary = _rows(summary_rows)[0]
    cell = _rows(cell_rows)[0]
    neurite_mask = neurite_labels[0]
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

    _, summary_rows, _, cell_bodies, neurite_labels, nuclei = _implementation()(
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

    assert _rows(summary_rows)[0]["number_of_cells"] == 1
    assert _rows(summary_rows)[0]["nuclear_channel_index"] == 1
    assert cell_bodies.shape == neurite_labels.shape == nuclei.shape == image.shape
    assert np.count_nonzero(cell_bodies[1]) == 0
    assert np.count_nonzero(neurite_labels[1]) == 0
    assert np.count_nonzero(nuclei[0]) == 0
    assert nuclei[1].max() == 1


def test_transmission_mode_detects_dark_cell_and_neurite():
    fluorescence = _draw_fluorescent_neuron()[0]
    transmission = np.full(fluorescence.shape, 2000, dtype=np.uint16)
    transmission[fluorescence > 0] = 500

    _, summary_rows, cell_rows, _, _, _ = _implementation()(
        transmission[None, ...],
        illumination=NeuriteIllumination.TRANSMISSION,
        cell_body=_cell_body_settings(),
        outgrowth=_outgrowth_settings(),
        pixel_size=1.0,
    )

    assert _rows(summary_rows)[0]["number_of_cells"] == 1
    assert _rows(summary_rows)[0]["total_processes"] == 1
    assert _rows(cell_rows)[0]["total_outgrowth_um"] > 80.0


def test_rejects_a_plain_2d_image_because_channels_must_be_explicit():
    with pytest.raises(ValueError, match="2D channel stack"):
        _implementation()(np.zeros((32, 32)), pixel_size=1.0)
