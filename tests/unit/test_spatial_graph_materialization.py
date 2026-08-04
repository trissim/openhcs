import numpy as np
import pytest

import openhcs  # noqa: F401
from polystore.filemanager import FileManager
from polystore.memory import MemoryStorageBackend
from polystore.napari_stream import NapariStreamingBackend
from polystore.roi import PolylineShape
from polystore.roi_converters import NapariROIConverter

from openhcs.core.artifacts import (
    ArtifactOutputPlan,
    ArtifactSpec,
    ArtifactType,
    ObjectArtifactMemberSubjectRelation,
    ObjectArtifactSubjectBinding,
    ObjectLabelsArtifactType,
    SpatialGraphArtifactType,
)
from openhcs.core.runtime_image_values import ImagePayloadMetadata
from openhcs.core.runtime_plane_projection import RuntimePlaneAxis
from openhcs.core.runtime_spatial_graph import (
    SpatialGraph,
    SpatialGraphEdge,
    SpatialGraphNode,
)
from openhcs.core.steps.function_runtime import (
    FunctionOutputContextStrategy,
    SpatialGraphFunctionOutputContextStrategy,
)
from openhcs.core.source_image_provenance import SourceImageProvenancePlanes
from openhcs.processing.materialization import (
    BackendSaver,
    MaterializationSpec,
    Output,
    SpatialGraphROIOptions,
    SWCOptions,
    WriteMode,
    materialization_outputs,
    registered_materialization_option_types,
)


def _edge(
    edge_id: int,
    source: SpatialGraphNode,
    target: SpatialGraphNode,
    coordinates,
    **features,
) -> SpatialGraphEdge:
    return SpatialGraphEdge.from_features(
        edge_id=edge_id,
        source=source,
        target=target,
        coordinates=np.asarray(coordinates, dtype=float),
        features=features,
    )


def _branched_graph() -> SpatialGraph:
    root = SpatialGraphNode(1, (0.0, 0.0), radius=2.0)
    junction = SpatialGraphNode(2, (0.0, 2.0))
    upper = SpatialGraphNode(3, (2.0, 3.0))
    lower = SpatialGraphNode(4, (-2.0, 3.0))
    return SpatialGraph(
        name="neurite_graph",
        nodes=(root, junction, upper, lower),
        edges=(
            _edge(
                1,
                root,
                junction,
                ((0.0, 0.0), (0.0, 1.0), (0.0, 2.0)),
                branch_distance_um=2.0,
                tortuosity=1.0,
            ),
            _edge(
                2,
                junction,
                upper,
                ((0.0, 2.0), (1.0, 2.5), (2.0, 3.0)),
                branch_distance_um=2.8,
                neuron_label=7,
            ),
            _edge(
                3,
                junction,
                lower,
                ((0.0, 2.0), (-1.0, 2.5), (-2.0, 3.0)),
                branch_distance_um=2.8,
                neuron_label=7,
            ),
        ),
        coordinate_spacing=(2.0, 3.0),
    )


def _outputs(graph: SpatialGraph):
    filemanager = FileManager({"memory": MemoryStorageBackend()})
    return materialization_outputs(
        MaterializationSpec(SWCOptions(), SpatialGraphROIOptions()),
        data=graph,
        path="/tmp/A01_neurite_graph_step3.roi.zip",
        filemanager=filemanager,
    )


@pytest.mark.unit
def test_spatial_graph_is_registered_and_owns_direct_immutable_references() -> None:
    graph = _branched_graph()

    assert ArtifactType.coerce("spatial_graph") is SpatialGraphArtifactType
    assert SpatialGraphArtifactType.accepts_runtime_payload(graph)
    assert graph.edges[0].source is graph.nodes[0]
    assert graph.edges[0].target is graph.nodes[1]
    assert graph.roots() == (graph.nodes[0],)
    with pytest.raises(ValueError, match="read-only"):
        graph.edges[0].coordinates[0, 0] = 5.0


@pytest.mark.unit
def test_spatial_graph_output_context_preserves_exact_source_identity() -> None:
    graph = _branched_graph()
    source = ImagePayloadMetadata(
        source_path="/tmp/A01_s002_w1_z001_t001.tif",
        source_component_metadata={
            "well": "A01",
            "site": 2,
            "channel": 1,
            "z_index": 1,
            "timepoint": 1,
        },
    ).payload_with(np.zeros((8, 8), dtype=np.float32))
    output_plan = ArtifactOutputPlan(
        name=graph.name,
        path="/memory/neurite_graph.pkl",
        artifact_type=SpatialGraphArtifactType,
    )

    strategy = FunctionOutputContextStrategy.for_output_plan(output_plan)
    contextualized = strategy.contextualize(source, graph, output_plan, None)

    assert isinstance(strategy, SpatialGraphFunctionOutputContextStrategy)
    assert isinstance(contextualized, SpatialGraph)
    assert contextualized is not graph
    assert contextualized.source_path == "/tmp/A01_s002_w1_z001_t001.tif"
    assert dict(contextualized.source_component_metadata or {}) == {
        "well": "A01",
        "site": 2,
        "channel": 1,
        "z_index": 1,
        "timepoint": 1,
    }
    assert not graph.source_provenance.has_values


@pytest.mark.unit
def test_spatial_graph_output_context_projects_declared_source_plane() -> None:
    graph = _branched_graph().replace_fields(source_plane_index=1)
    source = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/tmp/hoechst.tif", "/tmp/smi312.tif"),
            component_metadata=({"channel": "1"}, {"channel": "4"}),
        ),
        source_image_names=("Hoechst", "SMI312"),
    ).payload_with(np.zeros((2, 8, 8), dtype=np.float32))
    output_plan = ArtifactOutputPlan(
        name=graph.name,
        path="/memory/neurite_graph.pkl",
        artifact_type=SpatialGraphArtifactType,
    )

    contextualized = FunctionOutputContextStrategy.for_output_plan(
        output_plan
    ).contextualize(source, graph, output_plan, None)

    assert contextualized.source_path == "/tmp/smi312.tif"
    assert contextualized.source_component_metadata["channel"] == "4"
    assert contextualized.source_image_names == ("SMI312",)


@pytest.mark.unit
def test_spatial_graph_normalizes_mutable_node_and_edge_sequences() -> None:
    root = SpatialGraphNode(1, (0.0, 0.0))
    target = SpatialGraphNode(2, (0.0, 1.0))
    edge = _edge(1, root, target, ((0.0, 0.0), (0.0, 1.0)))
    nodes = [root, target]
    edges = [edge]

    graph = SpatialGraph("graph", nodes, edges)  # type: ignore[arg-type]
    nodes.clear()
    edges.clear()

    assert graph.nodes == (root, target)
    assert graph.edges == (edge,)


@pytest.mark.unit
def test_spatial_graph_rejects_nonmember_endpoint_references() -> None:
    root = SpatialGraphNode(1, (0.0, 0.0))
    member_target = SpatialGraphNode(2, (0.0, 1.0))
    equal_nonmember_target = SpatialGraphNode(2, (0.0, 1.0))
    edge = _edge(
        1,
        root,
        equal_nonmember_target,
        ((0.0, 0.0), (0.0, 1.0)),
    )

    with pytest.raises(ValueError, match="directly reference nodes"):
        SpatialGraph("graph", (root, member_target), (edge,))


@pytest.mark.unit
def test_directed_forest_validation_rejects_cycles_and_multiple_parents() -> None:
    first = SpatialGraphNode(1, (0.0, 0.0))
    second = SpatialGraphNode(2, (0.0, 1.0))
    third = SpatialGraphNode(3, (1.0, 1.0))
    cycle = SpatialGraph(
        "cycle",
        (first, second, third),
        (
            _edge(1, first, second, ((0.0, 0.0), (0.0, 1.0))),
            _edge(2, second, third, ((0.0, 1.0), (1.0, 1.0))),
            _edge(3, third, first, ((1.0, 1.0), (0.0, 0.0))),
        ),
    )
    multiple_parents = SpatialGraph(
        "multiple_parents",
        (first, second, third),
        (
            _edge(1, first, third, ((0.0, 0.0), (1.0, 1.0))),
            _edge(2, second, third, ((0.0, 1.0), (1.0, 1.0))),
        ),
    )

    with pytest.raises(ValueError, match="cycle detected"):
        cycle.require_directed_forest()
    with pytest.raises(ValueError, match="multiple incoming edges"):
        multiple_parents.require_directed_forest()
    with pytest.raises(ValueError, match="cycle detected"):
        _outputs(cycle)


@pytest.mark.unit
def test_swc_writer_emits_deterministic_root_to_leaf_physical_samples() -> None:
    swc_output, _roi_output = _outputs(_branched_graph())

    assert swc_output.path == "/tmp/A01_neurite_graph_step3.swc"
    assert swc_output.content.splitlines() == [
        "# OpenHCS spatial graph: neurite_graph",
        "# id type x y z radius parent",
        "1 1 0 0 0 2 -1",
        "2 2 3 0 0 1 1",
        "3 2 6 0 0 1 2",
        "4 2 7.5 2 0 1 3",
        "5 2 9 4 0 1 4",
        "6 2 7.5 -2 0 1 3",
        "7 2 9 -4 0 1 6",
    ]


@pytest.mark.unit
def test_swc_reader_restores_physical_forest_and_standard_features(tmp_path) -> None:
    swc_output, _roi_output = _outputs(_branched_graph())
    swc_path = tmp_path / "saved-neurites.swc"
    swc_path.write_text(swc_output.content, encoding="utf-8")

    restored = SpatialGraph.from_swc(swc_path)

    assert restored.name == "neurite_graph"
    assert restored.source_path == str(swc_path)
    assert restored.coordinate_spacing == (1.0, 1.0, 1.0)
    assert tuple(node.node_id for node in restored.nodes) == tuple(range(1, 8))
    assert tuple(edge.target_node_id for edge in restored.edges) == tuple(range(2, 8))
    assert restored.nodes[0].coordinates == (0.0, 0.0, 0.0)
    assert restored.nodes[1].coordinates == (0.0, 0.0, 3.0)
    assert restored.nodes[0].feature_mapping() == {
        "swc_type": 1,
        "swc_parent_id": -1,
    }
    assert restored.nodes[1].feature_mapping() == {
        "swc_type": 2,
        "swc_parent_id": 1,
    }
    assert restored.nodes[0].radius == 2.0
    restored.require_directed_forest()


@pytest.mark.unit
def test_swc_reader_writer_round_trip_preserves_structure_types(tmp_path) -> None:
    source_path = tmp_path / "typed-neurites.swc"
    source_path.write_text(
        "\n".join(
            (
                "10 1 1 2 3 4 -1",
                "30 3 2 4 6 2 10",
                "70 4 3 6 9 1 30",
                "",
            )
        ),
        encoding="utf-8",
    )
    restored = SpatialGraph.from_swc(source_path)

    swc_output = materialization_outputs(
        MaterializationSpec(SWCOptions()),
        data=restored,
        path="/tmp/typed-neurites.roi.zip",
        filemanager=FileManager({"memory": MemoryStorageBackend()}),
    )[0]

    assert swc_output.content.splitlines()[2:] == [
        "1 1 1 2 3 4 -1",
        "2 3 2 4 6 2 1",
        "3 4 3 6 9 1 2",
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("rows", "message"),
    (
        ("1 1 0 0 0 1 -1\n2 2 1 0 0 1 99\n", "missing parent 99"),
        ("1 1 0 0 0 1 2\n2 2 1 0 0 1 1\n", "cycle detected"),
    ),
)
def test_swc_reader_rejects_invalid_parent_topology(tmp_path, rows, message) -> None:
    swc_path = tmp_path / "invalid.swc"
    swc_path.write_text(rows, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        SpatialGraph.from_swc(swc_path)


@pytest.mark.unit
def test_graph_roi_projection_preserves_paths_and_graph_features() -> None:
    _swc_output, roi_output = _outputs(_branched_graph())

    assert roi_output.path == "/tmp/A01_neurite_graph_step3.graph.roi.zip"
    assert len(roi_output.content) == 3
    first_roi = roi_output.content[0]
    assert isinstance(first_roi.shapes[0], PolylineShape)
    np.testing.assert_array_equal(
        first_roi.shapes[0].coordinates,
        np.array(((0.0, 0.0), (0.0, 1.0), (0.0, 2.0))),
    )
    assert first_roi.metadata == {
        "label": 1,
        "area": 0.0,
        "perimeter": 2.0,
        "centroid": (0.0, 1.0),
        "graph": "neurite_graph",
        "edge_id": 1,
        "source_node_id": 1,
        "target_node_id": 2,
        "branch_distance_um": 2.0,
        "tortuosity": 1.0,
    }

    napari_shapes = NapariROIConverter.rois_to_shapes(roi_output.content)
    assert napari_shapes[0]["type"] == "path"
    assert napari_shapes[0]["metadata"]["branch_distance_um"] == 2.0


@pytest.mark.unit
def test_graph_roi_projects_declared_object_subject_without_losing_edge_identity() -> None:
    neurons = ArtifactSpec.output("neurons", ObjectLabelsArtifactType)
    base_graph = _branched_graph()
    graph = SpatialGraph(
        name=base_graph.name,
        nodes=base_graph.nodes,
        edges=tuple(
            SpatialGraphEdge.from_features(
                edge_id=edge.edge_id,
                source=edge.source,
                target=edge.target,
                coordinates=edge.coordinates,
                features={**edge.feature_mapping(), "neuron_label": 7},
            )
            for edge in base_graph.edges
        ),
        coordinate_spacing=base_graph.coordinate_spacing,
    )
    output_plan = ArtifactOutputPlan(
        name="neurite_graph",
        path="/memory/neurite_graph.pkl",
        artifact_type=SpatialGraphArtifactType,
        relations=(
            ObjectArtifactMemberSubjectRelation(
                source=neurons.ref(),
                member_id_field="neuron_label",
            ),
        ),
        producer_step_index=4,
        producer_step_scope_id="neurite-step",
    )
    outputs = materialization_outputs(
        MaterializationSpec(SpatialGraphROIOptions()),
        data=graph,
        path="/tmp/A01_neurite_graph_step4.roi.zip",
        filemanager=FileManager({"memory": MemoryStorageBackend()}),
        output_plan=output_plan,
    )

    rois = outputs[0].content
    assert [roi.metadata["edge_id"] for roi in rois] == [1, 2, 3]
    assert [
        roi.metadata[ObjectArtifactSubjectBinding.SUBJECT_ID_FEATURE]
        for roi in rois
    ] == [7, 7, 7]
    subject_tokens = {
        roi.metadata[ObjectArtifactSubjectBinding.SUBJECT_FEATURE]
        for roi in rois
    }
    assert len(subject_tokens) == 1
    assert '"object_labels","neurons","neurite-step",4' in subject_tokens.pop()


@pytest.mark.unit
def test_graph_materialization_candidate_paths_and_streaming_support_are_generic() -> (
    None
):
    spec = MaterializationSpec(SWCOptions(), SpatialGraphROIOptions())

    assert spec.candidate_paths("/tmp/A01_neurites.roi.zip") == (
        "/tmp/A01_neurites.swc",
        "/tmp/A01_neurites.graph.roi.zip",
    )
    assert SWCOptions in registered_materialization_option_types()
    assert SpatialGraphROIOptions in registered_materialization_option_types()
    streaming_backend = NapariStreamingBackend()
    assert streaming_backend.supports_file_path("/tmp/A01_neurites.graph.roi.zip")
    assert not streaming_backend.supports_file_path("/tmp/A01_neurites.swc")


@pytest.mark.unit
def test_empty_graph_roi_persists_without_inventing_a_viewer_element() -> None:
    output = Output(
        path="/tmp/A01_neurites.graph.roi.zip",
        content=[],
    )
    memory_backend = MemoryStorageBackend()
    streaming_backend = NapariStreamingBackend()

    assert memory_backend.accepts_payload(output.content, output.path)
    assert not streaming_backend.accepts_payload(output.content, output.path)

    filemanager = FileManager(
        {
            "memory": memory_backend,
            "napari_stream": streaming_backend,
        }
    )
    BackendSaver(
        ["memory", "napari_stream"],
        filemanager,
        {},
        write_mode=WriteMode.OVERWRITE,
    ).save_all((output,))

    assert filemanager.exists(output.path, "memory")


@pytest.mark.unit
def test_graph_roi_projection_rejects_three_dimensional_imagej_paths() -> None:
    root = SpatialGraphNode(1, (0.0, 0.0, 0.0))
    target = SpatialGraphNode(2, (1.0, 2.0, 3.0))
    graph = SpatialGraph(
        "graph_3d",
        (root, target),
        (_edge(1, root, target, ((0.0, 0.0, 0.0), (1.0, 2.0, 3.0))),),
        coordinate_spacing=(1.0, 1.0, 1.0),
    )

    with pytest.raises(ValueError, match="ImageJ polyline ROI archives are 2D"):
        _outputs(graph)
