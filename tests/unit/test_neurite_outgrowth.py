from dataclasses import fields
from inspect import signature

import numpy as np
import pytest
from skimage.draw import disk, line

from openhcs.core.artifacts import (
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.callable_contract import CallableContract
from openhcs.core.measurement_row_materialization import (
    DataclassMeasurementColumnarRows,
)
from openhcs.processing.backends.cellprofiler.skeleton import (
    ObjectSkeletonMeasurement,
)
from openhcs.processing.backends.analysis.neurite_outgrowth import (
    MetaXpressCellBodySettings,
    MetaXpressNuclearSettings,
    MetaXpressOutgrowthSettings,
    NeuriteIllumination,
    neurite_outgrowth_metaxpress,
)


def _implementation():
    return CallableContract.from_callable(
        neurite_outgrowth_metaxpress
    ).resolve_raw_runtime_callable()


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


def _cell_body_settings(channel_index=None):
    return MetaXpressCellBodySettings(
        approximate_max_width=30.0,
        minimum_area=100.0,
        intensity_above_local_background=100.0,
        channel_index=channel_index,
    )


def _outgrowth_settings(significant_threshold=20.0):
    return MetaXpressOutgrowthSettings(
        maximum_width=3.0,
        intensity_above_local_background=100.0,
        minimum_cell_growth_to_log_as_significant=significant_threshold,
    )


def _with_separate_body_channel(neurite_stack):
    body_image = np.zeros_like(neurite_stack[0])
    rows, columns = disk((64, 20), 9, shape=body_image.shape)
    body_image[rows, columns] = 1000
    return np.stack((body_image, neurite_stack[0]))


def test_signature_exposes_documented_metaxpress_controls_only():
    parameters = signature(_implementation()).parameters
    contract = CallableContract.from_callable(neurite_outgrowth_metaxpress)
    runtime_artifact_parameter_names = contract.artifact_inputs.names()
    exposed = [
        name for name in parameters if name not in runtime_artifact_parameter_names
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
        "channel_index",
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
        "neurons",
        "nuclei",
    )
    summary_spec, cell_spec, body_spec, neurite_spec, neurons_spec, nuclei_spec = (
        contract.artifact_outputs
    )
    assert summary_spec.artifact_type is MeasurementsArtifactType
    assert cell_spec.artifact_type is MeasurementsArtifactType
    assert cell_spec.relations[0].measurement_subject().name == body_spec.name
    assert all(
        spec.artifact_type is ObjectLabelsArtifactType
        for spec in (body_spec, neurite_spec, neurons_spec, nuclei_spec)
    )


def test_cp_metrics_and_significant_threshold_is_scoring_only():
    image = _with_separate_body_channel(_draw_fluorescent_neuron(branched=True))
    low_threshold = _implementation()(
        image,
        neurite_channel_index=1,
        cell_body=_cell_body_settings(channel_index=0),
        outgrowth=_outgrowth_settings(20.0),
        pixel_size=1.0,
    )
    high_threshold = _implementation()(
        image,
        neurite_channel_index=1,
        cell_body=_cell_body_settings(channel_index=0),
        outgrowth=_outgrowth_settings(1000.0),
        pixel_size=1.0,
    )

    low_summary = _rows(low_threshold[1])[0]
    low_cell = _rows(low_threshold[2])[0]
    high_summary = _rows(high_threshold[1])[0]
    high_cell = _rows(high_threshold[2])[0]

    assert low_summary["number_of_cells"] == 1
    assert low_summary["total_processes"] > 0
    assert low_summary["total_processes"] == high_summary["total_processes"]
    assert low_summary["total_branches"] == high_summary["total_branches"]
    assert low_cell["mean_process_length_um"] == pytest.approx(
        low_cell["total_outgrowth_um"] / low_cell["processes"]
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


def test_crossing_structure_uses_cp_branch_metrics_and_owned_trace():
    image = _with_separate_body_channel(_draw_fluorescent_neuron(crossing=True))
    _, summary_rows, cell_rows, _, neurite_labels, _, _ = _implementation()(
        image,
        neurite_channel_index=1,
        cell_body=_cell_body_settings(channel_index=0),
        outgrowth=_outgrowth_settings(),
        pixel_size=1.0,
    )

    summary = _rows(summary_rows)[0]
    cell = _rows(cell_rows)[0]
    neurite_mask = neurite_labels[1]
    assert summary["resolved_crossovers"] == 0
    assert summary["total_branches"] == cell["branches"] > 0
    assert summary["total_processes"] == cell["processes"] > 0
    assert 0.0 < cell["straightness"] <= 1.0
    assert cell["total_outgrowth_um"] > 150.0
    assert np.count_nonzero(neurite_mask[:, 75]) > 80
    assert np.count_nonzero(neurite_mask[64, :]) > 80


def test_disconnected_neurites_are_absent_from_owned_output_and_measurements():
    baseline_image = _with_separate_body_channel(_draw_fluorescent_neuron())
    image = baseline_image.copy()
    rows, columns = line(20, 50, 20, 110)
    image[1, rows, columns] = 700

    result = _implementation()(
        image,
        neurite_channel_index=1,
        cell_body=_cell_body_settings(channel_index=0),
        outgrowth=_outgrowth_settings(),
        pixel_size=1.0,
    )
    baseline = _implementation()(
        baseline_image,
        neurite_channel_index=1,
        cell_body=_cell_body_settings(channel_index=0),
        outgrowth=_outgrowth_settings(),
        pixel_size=1.0,
    )

    assert not result[4][1, 20, 80]
    assert result[4][1, 64, 80]
    assert _rows(result[1])[0]["total_outgrowth_um"] == pytest.approx(
        _rows(baseline[1])[0]["total_outgrowth_um"]
    )


def test_explicit_body_nuclear_and_neurite_channels_are_aligned():
    neurite_image = _draw_fluorescent_neuron()[0]
    body_image = np.zeros_like(neurite_image)
    rows, columns = disk((64, 20), 9, shape=body_image.shape)
    body_image[rows, columns] = 1000
    nucleus_image = np.zeros_like(neurite_image)
    rows, columns = disk((64, 20), 5, shape=nucleus_image.shape)
    nucleus_image[rows, columns] = 1200
    image = np.stack((nucleus_image, body_image, neurite_image))

    (
        _,
        summary_rows,
        cell_rows,
        cell_bodies,
        neurite_labels,
        neurons,
        nuclei,
    ) = _implementation()(
        image,
        neurite_channel_index=2,
        cell_body=_cell_body_settings(channel_index=1),
        outgrowth=_outgrowth_settings(),
        use_nuclear_stain=True,
        nuclear_stain=MetaXpressNuclearSettings(
            channel_index=0,
            approx_min_width=6.0,
            approx_max_width=14.0,
            intensity_above_local_background=200.0,
        ),
        pixel_size=1.0,
    )

    summary = _rows(summary_rows)[0]
    assert summary["number_of_cells"] == 1
    assert summary["neurite_channel_index"] == 2
    assert summary["cell_body_channel_index"] == 1
    assert summary["nuclear_channel_index"] == 0
    assert {row["slice_index"] for row in _rows(cell_rows)} == {1}
    assert cell_bodies.shape == neurite_labels.shape == nuclei.shape == image.shape
    assert cell_bodies[1].max() == 1
    assert np.count_nonzero(cell_bodies[[0, 2]]) == 0
    assert neurite_labels[2].max() == 1
    assert np.count_nonzero(neurite_labels[[0, 1]]) == 0
    assert neurons[2].max() == 1
    assert neurons[2, 64, 20] == 1
    assert neurons[2, 64, 80] == 1
    assert np.count_nonzero(neurons[[0, 1]]) == 0
    assert nuclei[0].max() == 1
    assert np.count_nonzero(nuclei[[1, 2]]) == 0


def test_overwide_nuclear_guided_foreground_is_not_a_cell_body():
    image = np.zeros((2, 128, 128), dtype=np.uint16)
    rows, columns = disk((64, 20), 9, shape=image.shape[1:])
    image[0, rows, columns] = 1000
    rows, columns = line(64, 28, 64, 115)
    image[0, rows, columns] = 700
    image[0, 20:65, 70:115] = 1000

    rows, columns = disk((64, 20), 5, shape=image.shape[1:])
    image[1, rows, columns] = 1200
    rows, columns = disk((42, 92), 5, shape=image.shape[1:])
    image[1, rows, columns] = 1200

    _, summary_rows, _, cell_bodies, _, _, nuclei = _implementation()(
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

    assert nuclei[1].max() == 2
    assert _rows(summary_rows)[0]["number_of_cells"] == 1
    assert cell_bodies[0].max() == 1
    assert cell_bodies[0, 64, 20] == 1
    assert cell_bodies[0, 42, 92] == 0


def test_transmission_mode_detects_dark_cell_and_neurite():
    fluorescence = _draw_fluorescent_neuron()[0]
    transmission = np.full(fluorescence.shape, 2000, dtype=np.uint16)
    transmission[fluorescence > 0] = 500

    _, summary_rows, cell_rows, _, _, _, _ = _implementation()(
        transmission[None, ...],
        illumination=NeuriteIllumination.TRANSMISSION,
        cell_body=_cell_body_settings(),
        outgrowth=_outgrowth_settings(),
        pixel_size=1.0,
    )

    assert _rows(summary_rows)[0]["number_of_cells"] == 1
    assert _rows(summary_rows)[0]["total_processes"] > 0
    assert _rows(cell_rows)[0]["total_outgrowth_um"] > 60.0


def test_cell_rows_map_authoritative_cp_seed_measurements(monkeypatch):
    image = _with_separate_body_channel(_draw_fluorescent_neuron())

    def fixed_cp_measurements(skeleton, *, seed_labels, **kwargs):
        del seed_labels, kwargs
        return (
            skeleton,
            DataclassMeasurementColumnarRows(
                (
                    ObjectSkeletonMeasurement(
                        slice_index=0,
                        object_label=1,
                        number_trunks=4,
                        number_non_trunk_branches=2,
                        number_branch_ends=5,
                        total_skeleton_length=123.0,
                    ),
                ),
                row_type=ObjectSkeletonMeasurement,
            ),
        )

    monkeypatch.setattr(
        "openhcs.processing.backends.analysis.neurite_outgrowth.measure_object_skeleton",
        fixed_cp_measurements,
    )
    _, summary_rows, cell_rows, _, _, _, _ = _implementation()(
        image,
        neurite_channel_index=1,
        cell_body=MetaXpressCellBodySettings(
            approximate_max_width=40.0,
            minimum_area=40.0,
            intensity_above_local_background=100.0,
            channel_index=0,
        ),
        outgrowth=_outgrowth_settings(),
        pixel_size=1.0,
    )

    summary = _rows(summary_rows)[0]
    cell = _rows(cell_rows)[0]
    assert cell["total_outgrowth_um"] == 123.0
    assert cell["processes"] == 4
    assert cell["branches"] == 2
    assert summary["total_outgrowth_um"] == 123.0
    assert summary["total_processes"] == 4
    assert summary["total_branches"] == 2


def test_explicit_same_body_channel_remains_valid_and_bounds_are_checked():
    image = _draw_fluorescent_neuron()
    result = _implementation()(
        image,
        neurite_channel_index=0,
        cell_body=_cell_body_settings(channel_index=0),
        outgrowth=_outgrowth_settings(),
        pixel_size=1.0,
    )

    assert _rows(result[1])[0]["cell_body_channel_index"] == 0
    with pytest.raises(ValueError, match="cell_body.channel_index"):
        _implementation()(
            image,
            cell_body=_cell_body_settings(channel_index=1),
            pixel_size=1.0,
        )


def test_body_and_nuclear_channel_may_be_shared():
    neurite_image = _draw_fluorescent_neuron()[0]
    body_and_nucleus = np.zeros_like(neurite_image)
    rows, columns = disk((64, 20), 9, shape=body_and_nucleus.shape)
    body_and_nucleus[rows, columns] = 1200
    image = np.stack((body_and_nucleus, neurite_image))

    result = _implementation()(
        image,
        neurite_channel_index=1,
        cell_body=_cell_body_settings(channel_index=0),
        outgrowth=_outgrowth_settings(),
        use_nuclear_stain=True,
        nuclear_stain=MetaXpressNuclearSettings(
            channel_index=0,
            approx_min_width=6.0,
            approx_max_width=20.0,
            intensity_above_local_background=200.0,
        ),
        pixel_size=1.0,
    )

    assert _rows(result[1])[0]["number_of_cells"] == 1
    assert result[3][0].max() == 1
    assert result[6][0].max() == 1


def test_rejects_a_plain_2d_image_because_channels_must_be_explicit():
    with pytest.raises(ValueError, match="2D channel stack"):
        _implementation()(np.zeros((32, 32)), pixel_size=1.0)
