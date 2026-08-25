import json
from pathlib import Path

import numpy as np
from polystore.base import ImageSamplingResult
from polystore.bioformats_storage import BioFormatsPlaneRef
from polystore.virtual_workspace import SourcePixelRef

from openhcs.constants import AllComponents, GroupBy, Microscope, VariableComponents
from openhcs.constants.input_source import InputSource
from openhcs.core.artifacts import ImageArtifactType
from openhcs.core.callable_contract import CallableContract
from openhcs.core.function_patterns import get_core_callable
from openhcs.core.source_bindings import ComponentSelector
from openhcs.core.source_projection import (
    OpenHCSPlaneAddress,
    SourceCandidate,
    SourceDatasetIdentity,
    SourcePlaneDataset,
    SourcePlaneStoreIdentity,
)
from openhcs.processing.backends.analysis.neurite_outgrowth import (
    count_neuronal_cell_bodies_metaxpress,
)
from openhcs.processing.backends.analysis.skan_axon_analysis import (
    AnalysisDimension,
    ThresholdMethod,
    skan_axon_skeletonize_and_analyze,
)
from openhcs.processing.backends.processors.numpy_processor import tophat
from openhcs.processing.presets.demo_contribution import PipelineDemoContribution
from openhcs.processing.presets.pipelines import czi_brain_axon_cellbody as demo_module
from openhcs.processing.presets.pipelines.czi_brain_axon_cellbody import (
    CziBrainAxonCellBodyInputs,
    CziBrainCropSelection,
    build_czi_brain_axon_cellbody_demo,
    czi_brain_axon_cellbody_demo_contribution,
)


def _inputs(plate_path: Path, output_root: Path) -> CziBrainAxonCellBodyInputs:
    return CziBrainAxonCellBodyInputs(
        plate_path=plate_path,
        output_root=output_root,
        viewer_port=6008,
    )


def _candidate(
    root: Path,
    identity: SourceDatasetIdentity,
    *,
    site: str,
    site_label: str,
    channel: str,
    channel_label: str,
) -> SourceCandidate:
    source_path = root
    series_index = int(site) - 1
    plane_index = int(channel) - 1
    return SourceCandidate(
        source_ref=SourcePixelRef(
            backend="bioformats",
            backend_address=BioFormatsPlaneRef(
                source_path=source_path,
                series_index=series_index,
                plane_index=plane_index,
            ).to_backend_address(),
        ),
        relative_path=source_path.name,
        metadata={},
        component_labels={
            AllComponents.SITE.value: site_label,
            AllComponents.CHANNEL.value: channel_label,
        },
        declared_address=OpenHCSPlaneAddress.from_values(
            well=".",
            site=site,
            channel=channel,
            z_index="1",
            timepoint="1",
        ),
        dataset_identity=identity,
        store_identity=SourcePlaneStoreIdentity(
            container_paths=(source_path,),
            sample_group_id=f"sample:{site}",
            image_id=f"image:{site}",
            series_id=f"series:{site}",
            plane_id=f"plane:{site}:{channel}",
        ),
    )


def _declared_dataset(root: Path) -> SourcePlaneDataset:
    identity = SourceDatasetIdentity.for_root(root)
    candidates = (
        _candidate(
            root,
            identity,
            site="2",
            site_label="ScanRegion1",
            channel="4",
            channel_label="Cy5 (Red Laser)",
        ),
        _candidate(
            root,
            identity,
            site="1",
            site_label="ScanRegion0",
            channel="1",
            channel_label="DAPI (UV Laser)",
        ),
        _candidate(
            root,
            identity,
            site="1",
            site_label="ScanRegion0",
            channel="3",
            channel_label="Rhodamine (Green Laser)",
        ),
        _candidate(
            root,
            identity,
            site="1",
            site_label="ScanRegion0",
            channel="4",
            channel_label="Cy5 (Red Laser)",
        ),
    )
    return SourcePlaneDataset(
        root=root,
        identity=identity,
        candidates=candidates,
        pixel_size=0.325,
    )


def test_czi_demo_declares_bounded_three_channel_analysis_and_artifacts(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path / "plate", tmp_path / "output")
    pipeline_config, steps = build_czi_brain_axon_cellbody_demo(inputs)

    assert pipeline_config.microscope is Microscope.BIOFORMATS
    assert pipeline_config.well_filter_config.well_filter == "A01"
    assert pipeline_config.path_planning_config.well_filter == 0
    assert pipeline_config.materialize_runtime_artifacts is True
    assert pipeline_config.materialization_results_path == (
        inputs.output_root.resolve() / "results"
    )
    source_bindings = pipeline_config.source_bindings_config.bindings
    assert [binding.alias for binding in source_bindings] == [
        "Axon",
        "NeuronalSoma",
        "Nuclei",
    ]
    assert [binding.selector.components for binding in source_bindings] == [
        (ComponentSelector(AllComponents.CHANNEL, "1"),),
        (ComponentSelector(AllComponents.CHANNEL, "2"),),
        (ComponentSelector(AllComponents.CHANNEL, "3"),),
    ]
    assert [step.name for step in steps] == [
        "CZI Tissue Background Correction",
        "CZI Neuronal Cell-Body Count",
        "CZI Axon Background Correction",
        "CZI Full-Field Axon Network",
    ]
    assert get_core_callable(steps[0].func) is tophat
    assert get_core_callable(steps[1].func) is count_neuronal_cell_bodies_metaxpress
    assert get_core_callable(steps[2].func) is tophat
    assert get_core_callable(steps[3].func) is skan_axon_skeletonize_and_analyze

    assert steps[0].processing_config.variable_components == [
        VariableComponents.CHANNEL
    ]
    assert steps[0].processing_config.input_source is InputSource.PIPELINE_START
    assert steps[1].processing_config.variable_components == [
        VariableComponents.CHANNEL
    ]
    assert steps[1].processing_config.input_source is InputSource.PREVIOUS_STEP
    assert steps[2].processing_config.group_by is GroupBy.NONE
    assert steps[2].processing_config.variable_components == [
        VariableComponents.CHANNEL
    ]
    assert steps[2].processing_config.input_source is InputSource.PIPELINE_START
    assert steps[2].source_bindings.enabled is True
    assert [binding.alias for binding in steps[2].source_bindings.bindings] == ["Axon"]
    assert steps[3].processing_config.group_by is GroupBy.NONE
    assert steps[3].processing_config.variable_components == [
        VariableComponents.CHANNEL
    ]
    assert steps[3].processing_config.input_source is InputSource.PREVIOUS_STEP

    count_kwargs = steps[1].func[1]
    assert count_kwargs["cell_body"].channel_index == 1
    assert count_kwargs["nuclear_stain"].channel_index == 2
    assert steps[2].func[1] == {
        "selem_radius": 50,
        "downsample_factor": 2,
        "downsample_anti_aliasing": True,
        "upsample_order": 0,
    }
    axon_kwargs = steps[3].func[1]
    assert axon_kwargs == {
        "voxel_spacing": (1.0, 0.325, 0.325),
        "threshold_method": ThresholdMethod.MANUAL,
        "threshold_value": 6000.0,
        "min_object_size": 16,
        "min_branch_length": 3.0,
        "return_skeleton_visualizations": True,
        "analysis_dimension": AnalysisDimension.TWO_D,
    }

    for step in (steps[1], steps[3]):
        assert step.napari_streaming_config.enabled is True
        assert step.napari_streaming_config.persistent is True
        assert step.napari_streaming_config.port == inputs.viewer_port
    assert steps[2].napari_streaming_config.enabled is False
    assert CallableContract.from_callable(
        count_neuronal_cell_bodies_metaxpress
    ).artifact_outputs.names() == (
        "neuronal_cell_body_summary",
        "neuronal_cell_body_measurements",
        "neuronal_cell_bodies",
        "nuclei",
    )
    assert CallableContract.from_callable(
        skan_axon_skeletonize_and_analyze
    ).artifact_outputs.names() == (
        "axon_summary",
        "axon_branches",
        "skeleton_visualizations",
        "skeleton_masks",
    )


def test_czi_sampling_filemanager_uses_handler_owned_context_backend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_path = tmp_path / "source.czi"
    source_path.write_bytes(b"metadata fixture only")
    observed = {}

    class FakeFileManager:
        def __init__(self, registry):
            observed["registry"] = registry

    class FakeBioFormatsHandler:
        def __init__(self, filemanager):
            observed["handler_filemanager"] = filemanager

        def register_workspace_backends(self, plate_root, filemanager):
            observed["plate_root"] = plate_root
            observed["registered_filemanager"] = filemanager

        def initialize_workspace(self, plate_root, filemanager):
            observed["initialized_plate_root"] = plate_root
            self.register_workspace_backends(plate_root, filemanager)

    monkeypatch.setattr(demo_module, "ensure_storage_registry", lambda: None)
    monkeypatch.setattr(demo_module, "storage_registry", {"disk": object()})
    monkeypatch.setattr(demo_module, "FileManager", FakeFileManager)
    monkeypatch.setattr(demo_module, "BioFormatsHandler", FakeBioFormatsHandler)

    workspace_root = tmp_path / "source_workspace"
    filemanager = demo_module._source_filemanager(source_path, workspace_root)

    assert observed == {
        "registry": {"disk": demo_module.storage_registry["disk"]},
        "handler_filemanager": filemanager,
        "initialized_plate_root": workspace_root,
        "plate_root": workspace_root,
        "registered_filemanager": filemanager,
    }
    source_link = workspace_root / source_path.name
    assert source_link.samefile(source_path)
    assert not source_link.is_symlink()


def test_czi_contributor_samples_only_declared_crop_planes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_path = tmp_path / "source.czi"
    source_path.write_bytes(b"metadata fixture only")
    dataset = _declared_dataset(source_path)
    monkeypatch.setattr(
        demo_module.BioFormatsMetadataHandler,
        "source_dataset",
        lambda _self, _path: dataset,
    )

    sample_calls = []
    values = {
        "Cy5 (Red Laser)": 4,
        "Rhodamine (Green Laser)": 3,
        "DAPI (UV Laser)": 1,
    }

    class FakeFileManager:
        def sample(self, file_path, backend, request):
            sample_calls.append((file_path, backend, request))
            source_ref = BioFormatsPlaneRef.from_backend_address(file_path)
            value = values[
                {
                    0: "DAPI (UV Laser)",
                    2: "Rhodamine (Green Laser)",
                    3: "Cy5 (Red Laser)",
                }[source_ref.plane_index]
            ]
            pixels = np.full(request.shape_yx, value, dtype=np.uint16)
            return ImageSamplingResult(
                data=pixels,
                statistics_data=pixels,
                source_shape=(33_383, 42_526),
                resolution_shape=(33_383, 42_526),
                sample_origin_yx=request.origin_yx,
                selected_resolution_index=0,
            )

    filemanager_roots = []
    fake_filemanager = FakeFileManager()

    def fake_source_filemanager(source_root, workspace_root):
        filemanager_roots.append((source_root, workspace_root))
        return fake_filemanager

    monkeypatch.setattr(demo_module, "_source_filemanager", fake_source_filemanager)
    selection = CziBrainCropSelection(
        site_label="ScanRegion0",
        origin_yx=(19_000, 30_000),
        shape_yx=(768, 768),
    )
    contribution = czi_brain_axon_cellbody_demo_contribution(
        session_root=tmp_path / "session",
        source_path=source_path,
        selection=selection,
    )

    assert isinstance(contribution, PipelineDemoContribution)
    assert contribution.demo_id == "czi_brain_axon_cellbody"
    assert contribution.title == "CZI brain axons and neuronal cell bodies"
    assert contribution.plate_path.name == contribution.title
    assert contribution.presentation_identity.output_key == ("skeleton_visualizations")
    assert contribution.presentation_identity.artifact_kind == "image"
    assert contribution.presentation_identity.step_name == (
        "CZI Full-Field Axon Network"
    )
    assert contribution.supporting_presentation_identities[0].output_key == (
        "neuronal_cell_bodies"
    )
    assert contribution.supporting_presentation_identities[0].artifact_kind == (
        "object_labels"
    )
    assert contribution.prepare is not None
    contribution.prepare()

    assert filemanager_roots == [
        (
            source_path.resolve(),
            tmp_path / "session" / "source_workspaces" / "czi_brain_axon_cellbody",
        )
    ]
    assert [
        BioFormatsPlaneRef.from_backend_address(call[0]).plane_index
        for call in sample_calls
    ] == [
        3,
        2,
        0,
    ]
    assert {call[1] for call in sample_calls} == {"bioformats"}
    assert all(call[2].origin_yx == selection.origin_yx for call in sample_calls)
    assert all(call[2].shape_yx == selection.shape_yx for call in sample_calls)
    assert all(call[2].resolution_index == 0 for call in sample_calls)

    stack = np.load(contribution.plate_path / "stack.npy")
    assert stack.shape == (1, 1, 3, 768, 768)
    assert stack.dtype == np.uint16
    assert [int(stack[0, 0, index, 0, 0]) for index in range(3)] == [4, 3, 1]
    manifest = json.loads(
        (contribution.plate_path / "bioformats_spw.json").read_text(encoding="utf-8")
    )
    image = manifest["images"][0]
    assert image["channel_names"] == list(selection.ordered_channel_labels)
    assert image["pixel_size"] == 0.325
    assert image["pixels"] == {
        "size_c": 3,
        "size_z": 1,
        "size_t": 1,
        "planes": [
            {"c": 1, "z": 1, "t": 1, "index": 0},
            {"c": 2, "z": 1, "t": 1, "index": 1},
            {"c": 3, "z": 1, "t": 1, "index": 2},
        ],
    }
    assert "source_provenance" not in manifest

    monkeypatch.undo()
    prepared_dataset = demo_module.BioFormatsMetadataHandler().source_dataset(
        contribution.plate_path
    )
    assert prepared_dataset.pixel_size == 0.325
    assert [
        candidate.component_labels[AllComponents.CHANNEL.value]
        for candidate in prepared_dataset.candidates
    ] == list(selection.ordered_channel_labels)
    assert [
        candidate.declared_address.value_for(AllComponents.CHANNEL)
        for candidate in prepared_dataset.candidates
        if candidate.declared_address is not None
    ] == ["1", "2", "3"]
    assert [
        candidate.source_ref.source_axis_indices
        for candidate in prepared_dataset.candidates
    ] == [
        (0, 0, 0),
        (0, 0, 1),
        (0, 0, 2),
    ]

    from multiprocessing import SimpleQueue

    from objectstate import ObjectStateRegistry
    from objectstate.lazy_factory import ensure_global_config_context

    from openhcs.core.config import GlobalPipelineConfig
    from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
    from openhcs.core.progress import set_progress_queue

    ObjectStateRegistry.clear()
    set_progress_queue(SimpleQueue())
    try:
        ensure_global_config_context(
            GlobalPipelineConfig,
            GlobalPipelineConfig(num_workers=1),
        )
        compilation = (
            PipelineOrchestrator(
                contribution.plate_path,
                pipeline_config=contribution.pipeline_config,
            )
            .initialize()
            .compile_pipelines(
                pipeline_definition=list(contribution.pipeline_steps),
                well_filter=["A01"],
                is_zmq_execution=True,
            )
        )
    finally:
        set_progress_queue(None)

    context = compilation["execution_bundle"].runtime_contexts["A01"]
    full_axon_plan = context.step_plans[3]
    compiled_pattern = full_axon_plan.compiled_function_pattern
    assert compiled_pattern is not None
    invocation = next(compiled_pattern.iter_invocations())
    (main_flow_source,) = invocation.contract.artifact_inputs
    visualization_spec = (
        invocation.contract.artifact_outputs.require_by_name_and_artifact_type(
            "skeleton_visualizations",
            ImageArtifactType,
        )
    )
    visualization_plan = next(
        output
        for output in full_axon_plan.artifact_outputs.values()
        if output.name == visualization_spec.name
    )
    assert visualization_spec.source_context_sources() == (main_flow_source.ref(),)
    assert visualization_plan.source_context_source() == main_flow_source.ref()


def test_czi_contributor_rejects_missing_declared_role_plane(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_path = tmp_path / "source.czi"
    source_path.write_bytes(b"metadata fixture only")
    dataset = _declared_dataset(source_path)
    dataset = SourcePlaneDataset(
        root=dataset.root,
        identity=dataset.identity,
        candidates=tuple(
            candidate
            for candidate in dataset.candidates
            if candidate.component_labels.get(AllComponents.CHANNEL.value)
            != "Rhodamine (Green Laser)"
        ),
        pixel_size=dataset.pixel_size,
    )
    monkeypatch.setattr(
        demo_module.BioFormatsMetadataHandler,
        "source_dataset",
        lambda _self, _path: dataset,
    )

    contribution = czi_brain_axon_cellbody_demo_contribution(
        session_root=tmp_path / "session",
        source_path=source_path,
    )

    try:
        contribution.prepare()
    except ValueError as exc:
        assert "Rhodamine (Green Laser)" in str(exc)
        assert "ScanRegion0" in str(exc)
    else:
        raise AssertionError("missing declared soma channel must fail")
