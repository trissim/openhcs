from dataclasses import fields
from inspect import signature

import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage.draw import disk, line

from openhcs.core.artifacts import (
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
    SpatialGraphArtifactType,
)
from openhcs.core.callable_contract import CallableContract
from openhcs.core.measurement_row_materialization import (
    DataclassMeasurementColumnarRows,
)
from openhcs.core.runtime_object_labels import object_label_dense_array
from openhcs.core.runtime_spatial_graph import SpatialGraph
from openhcs.processing.backends.cellprofiler.primary_objects import (
    identify_primary_objects,
)
from openhcs.processing.backends.cellprofiler.secondary import (
    identify_secondary_objects,
)
from openhcs.processing.backends.cellprofiler.skeleton import (
    ObjectSkeletonMeasurement,
)
from openhcs.processing.backends.analysis.neurite_outgrowth import (
    CELLPROFILER_NEURITE_ENGINE_PROFILE,
    MetaXpressCellBodySettings,
    MetaXpressNuclearSettings,
    MetaXpressOutgrowthSettings,
    NeuriteIllumination,
    _TopologyResult,
    _analyze_topology,
    _build_neurite_morphology_graph,
    _prune_soma_detached_skeleton,
    _repair_signal_supported_skeleton,
    neurite_outgrowth_metaxpress,
)
from openhcs.processing.materialization import SpatialGraphROIOptions, SWCOptions


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
        "neurite_morphology",
    )
    (
        summary_spec,
        cell_spec,
        body_spec,
        neurite_spec,
        neurons_spec,
        nuclei_spec,
        morphology_spec,
    ) = contract.artifact_outputs
    assert summary_spec.artifact_type is MeasurementsArtifactType
    assert cell_spec.artifact_type is MeasurementsArtifactType
    assert cell_spec.relations[0].measurement_subject().name == neurons_spec.name
    assert cell_spec.relations[0].measurement_subject().id_field == "cell"
    assert all(
        spec.artifact_type is ObjectLabelsArtifactType
        for spec in (body_spec, neurite_spec, neurons_spec, nuclei_spec)
    )
    assert morphology_spec.artifact_type is SpatialGraphArtifactType
    morphology_subject = morphology_spec.relations[0].object_subject_binding()
    assert morphology_subject.source == neurons_spec.ref()
    assert morphology_subject.id_field == "label"
    assert tuple(
        type(output) for output in morphology_spec.materialization.outputs
    ) == (
        SWCOptions,
        SpatialGraphROIOptions,
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
    _, summary_rows, cell_rows, _, neurite_labels, _, _, _ = _implementation()(
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


def test_crossing_resolution_retains_two_logical_endpoint_groups():
    skeleton = np.zeros((65, 65), dtype=bool)
    skeleton[32, 5:60] = True
    skeleton[5:60, 32] = True
    cell_bodies = np.zeros(skeleton.shape, dtype=np.int32)
    cell_bodies[30:35, 3:9] = 1

    topology = _analyze_topology(
        skeleton,
        cell_bodies,
        pixel_size_um=1.0,
        outgrowth_width_px=3.0,
    )

    crossing_groups = []
    for coordinates, endpoint_groups in zip(
        topology.path_coordinates,
        topology.path_endpoint_groups,
    ):
        for endpoint_index, coordinate in enumerate((coordinates[0], coordinates[-1])):
            if tuple(coordinate) == (32, 32):
                crossing_groups.append(endpoint_groups[endpoint_index])

    assert len(crossing_groups) == 4
    assert len(set(crossing_groups)) == 2
    assert sorted(crossing_groups.count(group) for group in set(crossing_groups)) == [
        2,
        2,
    ]
    assert np.all(topology.path_branch_types == 1)


def test_neurite_morphology_is_soma_rooted_feature_bearing_forest():
    image = _with_separate_body_channel(_draw_fluorescent_neuron(branched=True))

    result = _implementation()(
        image,
        neurite_channel_index=1,
        cell_body=_cell_body_settings(channel_index=0),
        outgrowth=_outgrowth_settings(),
        pixel_size=1.0,
    )
    cell_bodies = result[3][0]
    neurite_labels = result[4][1]
    morphology = result[-1]

    assert isinstance(morphology, SpatialGraph)
    morphology.require_directed_forest()
    assert morphology.name == "neurite_morphology"
    assert morphology.coordinate_spacing == (1.0, 1.0)
    assert len(morphology.roots()) == 1
    assert len(morphology.edges) == len(morphology.nodes) - len(morphology.roots())
    assert morphology.roots()[0].feature_mapping() == {
        "label": 1,
        "neuron_label": 1,
        "node_role": "soma_attachment_root",
    }
    root_index = tuple(int(value) for value in morphology.roots()[0].coordinates)
    assert cell_bodies[root_index] == 0
    assert ndi.binary_dilation(cell_bodies == 1)[root_index]
    assert not np.any((neurite_labels > 0) & (cell_bodies > 0))
    expected_edge_features = {
        "label",
        "neuron_label",
        "branch_distance_um",
        "euclidean_distance_um",
        "tortuosity",
        "distance_from_soma_um",
        "branch_type",
    }
    for edge in morphology.edges:
        features = edge.feature_mapping()
        assert set(features) == expected_edge_features
        assert features["label"] == features["neuron_label"] == 1
        assert features["branch_distance_um"] > 0
        assert features["euclidean_distance_um"] > 0
        assert features["tortuosity"] >= 1.0
        assert features["distance_from_soma_um"] >= 0
        coordinates = np.rint(edge.coordinates).astype(int)
        assert not np.any(cell_bodies[tuple(coordinates.T)] > 0)


def test_neurite_morphology_provenance_selects_the_neurite_plane():
    image = _with_separate_body_channel(_draw_fluorescent_neuron(branched=True))

    morphology = _implementation()(
        image,
        neurite_channel_index=1,
        cell_body=_cell_body_settings(channel_index=0),
        outgrowth=_outgrowth_settings(),
        pixel_size=1.0,
    )[-1]

    assert morphology.source_plane_index == 1


def test_neurite_morphology_breaks_cycle_without_dropping_path_geometry():
    path_coordinates = (
        np.array([[8, 8], [8, 20]], dtype=int),
        np.array([[8, 20], [20, 20]], dtype=int),
        np.array([[20, 20], [8, 8]], dtype=int),
    )
    topology = _TopologyResult(
        path_owners=np.ones(3, dtype=np.int32),
        path_distances=np.array([0.0, 12.0, 12.0]),
        path_lengths=np.array([12.0, 12.0, np.hypot(12.0, 12.0)]),
        path_euclidean_lengths=np.array([12.0, 12.0, np.hypot(12.0, 12.0)]),
        path_coordinates=path_coordinates,
        path_endpoint_groups=((1, 2), (2, 3), (3, 1)),
        path_branch_types=np.full(3, 2, dtype=np.int32),
        endpoint_group_coordinates={
            1: (8.0, 8.0),
            2: (8.0, 20.0),
            3: (20.0, 20.0),
        },
        transitions={0: (1, 2), 1: (0, 2), 2: (0, 1)},
        root_paths_by_cell={1: (0, 2)},
        branch_owner={},
        crossing_nodes=frozenset(),
    )
    cell_bodies = np.zeros((32, 32), dtype=np.int32)
    cell_bodies[6:11, 6:11] = 1

    morphology = _build_neurite_morphology_graph(
        topology,
        cell_bodies,
        pixel_size_um=1.0,
        outgrowth_width_px=3.0,
    )

    morphology.require_directed_forest()
    assert len(morphology.nodes) == 4
    assert len(morphology.edges) == 3
    assert len(morphology.roots()) == 1
    assert (
        sum(
            node.feature_mapping().get("node_role") == "cycle_break"
            for node in morphology.nodes
        )
        == 1
    )
    assert {
        tuple(int(value) for value in coordinate)
        for edge in morphology.edges
        for coordinate in edge.coordinates
    } >= {
        tuple(int(value) for value in coordinate)
        for path in path_coordinates
        for coordinate in path
    }


def test_neurite_morphology_does_not_fabricate_links_between_components():
    topology = _TopologyResult(
        path_owners=np.ones(2, dtype=np.int32),
        path_distances=np.zeros(2, dtype=float),
        path_lengths=np.array([8.0, 10.0]),
        path_euclidean_lengths=np.array([8.0, 10.0]),
        path_coordinates=(
            np.array([[12, 12], [12, 20]], dtype=int),
            np.array([[14, 12], [24, 12]], dtype=int),
        ),
        path_endpoint_groups=((1, 2), (3, 4)),
        path_branch_types=np.zeros(2, dtype=np.int32),
        endpoint_group_coordinates={
            1: (12.0, 12.0),
            2: (12.0, 20.0),
            3: (14.0, 12.0),
            4: (24.0, 12.0),
        },
        transitions={0: (), 1: ()},
        root_paths_by_cell={1: (0, 1)},
        branch_owner={},
        crossing_nodes=frozenset(),
    )
    cell_bodies = np.zeros((32, 32), dtype=np.int32)
    cell_bodies[10:17, 9:14] = 1

    morphology = _build_neurite_morphology_graph(
        topology,
        cell_bodies,
        pixel_size_um=1.0,
        outgrowth_width_px=3.0,
    )

    assert len(morphology.roots()) == 2
    assert len(morphology.edges) == 2
    assert {edge.source.node_id for edge in morphology.edges} == {
        root.node_id for root in morphology.roots()
    }
    for edge in morphology.edges:
        segment_lengths = np.linalg.norm(np.diff(edge.coordinates, axis=0), axis=1)
        assert edge.feature_mapping()["branch_distance_um"] == pytest.approx(
            float(segment_lengths.sum())
        )
        assert tuple(edge.coordinates[0]) in {(12.0, 12.0), (14.0, 12.0)}


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
        morphology,
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
    assert isinstance(morphology, SpatialGraph)


def test_compact_final_neurons_equal_modular_cp_seed_propagation():
    image = _with_separate_body_channel(_draw_fluorescent_neuron(branched=True))
    engine = CELLPROFILER_NEURITE_ENGINE_PROFILE

    result = _implementation()(
        image,
        neurite_channel_index=1,
        cell_body=_cell_body_settings(channel_index=0),
        outgrowth=_outgrowth_settings(),
        pixel_size=1.0,
    )

    *_, detected_body_payload = CallableContract.from_callable(
        identify_primary_objects
    ).resolve_raw_runtime_callable()(
        image[0],
        **engine.compact_body_detection_kwargs(adaptive_window_size=64),
    )
    accepted_body_payload = detected_body_payload.with_replacement_labels(result[3][0])
    *_, expected_neuron_payload = CallableContract.from_callable(
        identify_secondary_objects
    ).resolve_raw_runtime_callable()(
        image[1],
        primary_labels=accepted_body_payload,
        **engine.secondary_kwargs(),
    )

    np.testing.assert_array_equal(
        result[5][1],
        object_label_dense_array(expected_neuron_payload),
    )


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

    _, summary_rows, _, cell_bodies, _, _, nuclei, _ = _implementation()(
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

    _, summary_rows, cell_rows, _, _, _, _, _ = _implementation()(
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
    _, summary_rows, cell_rows, _, _, _, _, _ = _implementation()(
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


def test_nuclear_body_mode_rejects_non_neuronal_dapi_seed_and_unowned_signal():
    image = np.zeros((2, 160, 160), dtype=np.uint16)
    for center in ((80, 25), (35, 105)):
        rows, columns = disk(center, 9, shape=image.shape[1:])
        image[0, rows, columns] = 1200

    rows, columns = disk((80, 25), 9, shape=image.shape[1:])
    image[1, rows, columns] = 1200
    rows, columns = line(80, 33, 80, 130)
    image[1, rows, columns] = 700
    rows, columns = line(35, 123, 35, 150)
    image[1, rows, columns] = 900

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

    summary = _rows(result[1])[0]
    assert summary["number_of_cells"] == 1
    assert result[3][0, 80, 25] == 1
    assert result[3][0, 35, 105] == 0
    assert result[5][1, 80, 100] == 1
    assert {row["cell"] for row in _rows(result[2])} == {1}
    for row in _rows(result[2]):
        assert np.any(result[4][1] == row["cell"])
        assert row["total_outgrowth_um"] > 0
    graph_pixels = np.zeros(image.shape[1:], dtype=bool)
    for edge in result[-1].edges:
        coordinates = np.rint(edge.coordinates).astype(int)
        for start, end in zip(coordinates[:-1], coordinates[1:]):
            rows, columns = line(start[0], start[1], end[0], end[1])
            graph_pixels[rows, columns] = True
    assert np.all(
        ~result[4][1].astype(bool)
        | ndi.binary_dilation(graph_pixels, structure=np.ones((3, 3), dtype=bool))
    )


def test_nuclear_seeds_fill_bounded_signal_bodies_and_keep_zero_growth_cell():
    image = np.zeros((2, 160, 160), dtype=np.uint16)
    for center in ((80, 25), (35, 105)):
        rows, columns = disk(center, 5, shape=image.shape[1:])
        image[0, rows, columns] = 1200
        rows, columns = disk(center, 9, shape=image.shape[1:])
        image[1, rows, columns] = 1200
    rows, columns = line(80, 33, 80, 130)
    image[1, rows, columns] = 700

    result = _implementation()(
        image,
        neurite_channel_index=1,
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

    summary = _rows(result[1])[0]
    cell_rows = _rows(result[2])
    cell_bodies = result[3]
    neurites = result[4]
    nuclei = result[6]
    assert summary["number_of_cells"] == 2
    assert np.count_nonzero(cell_bodies[0]) == 0
    assert cell_bodies[1].max() == 2
    assert nuclei[0].max() == 2
    assert np.count_nonzero(nuclei[1]) == 0
    assert sorted(row["total_outgrowth_um"] for row in cell_rows) == [0.0, 42.0]
    zero_growth_cell = next(
        row["cell"] for row in cell_rows if row["total_outgrowth_um"] == 0.0
    )
    assert not np.any(neurites[1] == zero_growth_cell)
    assert zero_growth_cell not in {
        edge.feature_mapping()["neuron_label"] for edge in result[-1].edges
    }
    assert not np.any((cell_bodies[1] > 0) & (neurites[1] > 0))
    assert result[5][1, 35, 140] == 0


def test_signal_supported_repair_follows_curved_trace_instead_of_chord():
    labels = np.zeros((64, 64), dtype=np.int32)
    labels[32, 8:15] = 1
    labels[32, 45:52] = 1
    cell_bodies = np.zeros_like(labels)
    cell_bodies[29:36, 5:12] = 1
    response = np.zeros(labels.shape, dtype=float)
    first_rows, first_columns = line(32, 14, 18, 28)
    second_rows, second_columns = line(18, 28, 32, 45)
    response[first_rows, first_columns] = 150.0
    response[second_rows, second_columns] = 150.0
    response[labels == 1] = 150.0
    owner_regions = np.zeros_like(labels)
    owner_regions[(response >= 100.0) | (cell_bodies == 1)] = 1

    repaired = _repair_signal_supported_skeleton(
        labels,
        response,
        owner_regions,
        cell_bodies,
        minimum_response=100.0,
    )

    assert ndi.label(repaired == 1, structure=np.ones((3, 3), dtype=bool))[1] == 1
    assert repaired[18, 28] == 1
    assert not np.any(repaired[32, 16:43])


def test_signal_supported_repair_rejects_unsupported_and_foreign_owner_routes():
    labels = np.zeros((48, 48), dtype=np.int32)
    labels[20, 5:10] = 1
    labels[20, 38:43] = 1
    labels[8:40, 24] = 2
    cell_bodies = np.zeros_like(labels)
    cell_bodies[17:24, 2:8] = 1
    cell_bodies[5:11, 21:28] = 2
    response = np.zeros(labels.shape, dtype=float)
    response[20, 5:43] = 150.0
    response[labels > 0] = 150.0
    owner_regions = np.zeros_like(labels)
    owner_regions[20, 5:24] = 1
    owner_regions[20, 25:43] = 1
    owner_regions[8:40, 24] = 2
    owner_regions[cell_bodies > 0] = cell_bodies[cell_bodies > 0]

    repaired = _repair_signal_supported_skeleton(
        labels,
        response,
        owner_regions,
        cell_bodies,
        minimum_response=100.0,
    )

    assert np.all(repaired[20, 5:10] == 1)
    assert not np.any(repaired[20, 38:43] == 1)
    assert np.all(repaired[8:40, 24] == 2)
    assert not np.any(repaired[:, 24] == 1)


def test_public_neurite_skeleton_prunes_components_detached_from_the_soma():
    labels = np.zeros((32, 32), dtype=np.int32)
    labels[16, 9:16] = 1
    labels[5, 20:26] = 1
    cell_bodies = np.zeros_like(labels)
    rows, columns = disk((16, 5), 4, shape=labels.shape)
    cell_bodies[rows, columns] = 1

    pruned = _prune_soma_detached_skeleton(
        labels,
        cell_bodies,
        attachment_distance=2.5,
    )

    assert np.all(pruned[16, 9:16] == 1)
    assert not np.any(pruned[5, 20:26])


def test_rejects_a_plain_2d_image_because_channels_must_be_explicit():
    with pytest.raises(ValueError, match="2D channel stack"):
        _implementation()(np.zeros((32, 32)), pixel_size=1.0)
