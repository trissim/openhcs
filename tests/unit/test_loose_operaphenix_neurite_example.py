from multiprocessing import SimpleQueue
from pathlib import Path

import numpy as np
from objectstate import ObjectStateRegistry
import tifffile

from openhcs.agent.dto.knowledge import (
    KnowledgeBaseDocumentRequest,
    KnowledgeBaseSearchRequest,
)
from openhcs.agent.services.knowledge_base_service import KnowledgeBaseService
from objectstate.lazy_factory import ensure_global_config_context
from openhcs.constants.constants import AllComponents, Microscope
from openhcs.core.config import GlobalPipelineConfig
from openhcs.core.function_patterns import get_core_callable
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.progress import set_progress_queue
from openhcs.processing.backends.cellprofiler.secondary import (
    identify_secondary_objects,
)
from openhcs.processing.backends.cellprofiler.skeleton import (
    measure_object_skeleton_with_branchpoint_image,
)
from openhcs.processing.backends.cellprofiler.spreadsheet_export import (
    export_to_spreadsheet,
)
from openhcs.processing.presets.demo_contribution import PipelineDemoContribution
from openhcs.processing.presets.pipelines import (
    loose_operaphenix_neurite_outgrowth as example_module,
)
from openhcs.processing.presets.pipelines.loose_operaphenix_neurite_outgrowth import (
    NEURITE_BRANCHPOINT_IMAGE_NAME,
    LooseOperaPhenixNeuriteInputs,
    SemanticImageSource,
    build_loose_operaphenix_neurite_pipeline,
    loose_operaphenix_neurite_demo_contribution,
)


def _inputs(plate_path: Path, output_root: Path) -> LooseOperaPhenixNeuriteInputs:
    return LooseOperaPhenixNeuriteInputs(
        plate_path=plate_path,
        output_root=output_root,
        well="B03",
        site="7",
        z_index="1",
        timepoint="1",
        viewer_port=5999,
        hoechst=SemanticImageSource(
            alias="Hoechst",
            filename="b03f07p01-ch1sk1fk1fl1.tiff",
            channel="1",
        ),
        map2=SemanticImageSource(
            alias="MAP2",
            filename="b03f07p01-ch2sk1fk1fl1.tiff",
            channel="2",
        ),
        smi312=SemanticImageSource(
            alias="SMI312",
            filename="b03f07p01-ch4sk1fk1fl1.tiff",
            channel="4",
        ),
    )


def test_neurite_example_declares_bounded_sources_and_final_user_result(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path / "plate", tmp_path / "output")
    pipeline_config, steps = build_loose_operaphenix_neurite_pipeline(inputs)

    assert pipeline_config.microscope is Microscope.SOURCE_BINDINGS
    assert pipeline_config.well_filter_config.well_filter == "B03"
    assert pipeline_config.path_planning_config.well_filter == 0
    assert pipeline_config.materialize_runtime_artifacts is True
    assert pipeline_config.materialization_results_path == (
        inputs.output_root.resolve() / "results"
    )

    bindings = pipeline_config.source_bindings_config.bindings
    assert [binding.alias for binding in bindings] == ["Hoechst", "MAP2", "SMI312"]
    assert [binding.selector.filters[0].value for binding in bindings] == [
        inputs.hoechst.filename,
        inputs.map2.filename,
        inputs.smi312.filename,
    ]
    assert [
        next(
            selector.value
            for selector in binding.component_identity
            if selector.component is AllComponents.CHANNEL
        )
        for binding in bindings
    ] == ["1", "2", "4"]

    assert [step.name for step in steps] == [
        "NeuronBodies",
        "SMI312SourceSignal",
        "EnhancedNeurites",
        "NeuriteForeground",
        "NeuriteSkeleton",
        "PerNeuronNeuriteTopology",
        "UnifiedNeurons",
        "NeuriteSpreadsheetExport",
    ]
    assert get_core_callable(steps[5].func) is (
        measure_object_skeleton_with_branchpoint_image
    )
    assert get_core_callable(steps[6].func) is identify_secondary_objects
    assert get_core_callable(steps[7].func) is export_to_spreadsheet

    streamed_steps = [
        step for step in steps if step.napari_streaming_config.enabled is True
    ]
    assert [step.name for step in streamed_steps][-1] == "UnifiedNeurons"
    assert all(
        step.napari_streaming_config.well_filter is None for step in streamed_steps
    )
    assert {step.step_materialization_config.sub_dir for step in steps} >= {
        "qc_smi312_signal",
        "qc_neurite_mask",
        "qc_neurite_skeleton",
    }
    assert [
        step.name for step in steps if step.step_materialization_config.enabled is True
    ] == ["SMI312SourceSignal", "NeuriteForeground", "NeuriteSkeleton"]
    assert "/tmp/" not in Path(example_module.__file__).read_text(encoding="utf-8")


def test_neurite_example_supports_dapi_smi312_without_map2(tmp_path: Path) -> None:
    three_channel = _inputs(tmp_path / "plate", tmp_path / "output")
    inputs = LooseOperaPhenixNeuriteInputs(
        plate_path=three_channel.plate_path,
        output_root=three_channel.output_root,
        well=three_channel.well,
        site=three_channel.site,
        z_index=three_channel.z_index,
        timepoint=three_channel.timepoint,
        viewer_port=three_channel.viewer_port,
        hoechst=three_channel.hoechst,
        map2=None,
        smi312=three_channel.smi312,
    )

    pipeline_config, steps = build_loose_operaphenix_neurite_pipeline(inputs)

    assert inputs.cell_body_source is inputs.smi312
    assert inputs.channel_stack == (inputs.hoechst, inputs.smi312)
    assert [
        binding.alias for binding in pipeline_config.source_bindings_config.bindings
    ] == ["Hoechst", "SMI312"]
    assert steps[0].source_bindings.bindings[0].alias == "SMI312"
    assert steps[1].source_bindings.bindings[0].alias == "SMI312"
    assert steps[6].source_bindings.bindings[0].alias == "SMI312"


def test_neurite_example_compiles_with_declared_loose_file_identities(
    tmp_path: Path,
) -> None:
    plate_path = tmp_path / "plate"
    plate_path.mkdir()
    inputs = _inputs(plate_path, tmp_path / "output")
    for source in (inputs.hoechst, inputs.map2, inputs.smi312):
        tifffile.imwrite(
            plate_path / source.filename,
            np.ones((32, 32), dtype=np.uint16),
        )

    pipeline_config, steps = build_loose_operaphenix_neurite_pipeline(inputs)
    expected_names = [step.name for step in steps]
    ObjectStateRegistry.clear()
    set_progress_queue(SimpleQueue())
    try:
        ensure_global_config_context(
            GlobalPipelineConfig,
            GlobalPipelineConfig(num_workers=1),
        )
        orchestrator = PipelineOrchestrator(
            plate_path,
            pipeline_config=pipeline_config,
        ).initialize()
        compilation = orchestrator.compile_pipelines(
            pipeline_definition=steps,
            well_filter=[inputs.well],
            is_zmq_execution=True,
        )
    finally:
        set_progress_queue(None)

    context = compilation["execution_bundle"].runtime_contexts[inputs.well]
    assert [plan.step_name for plan in context.step_plans.values()] == expected_names
    assert (
        context.step_plans[max(context.step_plans)].compiled_function_pattern
        is not None
    )


def test_neurite_example_contributes_modular_pipeline_to_master(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source"
    source_path.mkdir()
    source_filenames = (
        "r04c09f11p01-ch1sk1fk1fl1.tiff",
        "r04c09f11p01-ch2sk1fk1fl1.tiff",
        "r04c09f11p01-ch4sk1fk1fl1.tiff",
    )
    for filename in source_filenames:
        tifffile.imwrite(
            source_path / filename,
            np.ones((16, 16), dtype=np.uint16),
        )

    contribution = loose_operaphenix_neurite_demo_contribution(
        session_root=tmp_path / "session",
        source_path=source_path,
    )
    assert isinstance(contribution, PipelineDemoContribution)
    contribution.prepare()

    assert contribution.demo_id == ("loose_operaphenix_cellprofiler_neurite_outgrowth")
    assert contribution.plate_path.name == (
        "Opera Phenix modular CellProfiler neurite outgrowth"
    )
    assert [step.name for step in contribution.pipeline_steps][-3:] == [
        "PerNeuronNeuriteTopology",
        "UnifiedNeurons",
        "NeuriteSpreadsheetExport",
    ]
    assert contribution.presentation_identity.output_key == (
        "IdentifySecondaryObjects_7_object_labels_1"
    )
    assert contribution.presentation_identity.artifact_kind == "object_labels"
    assert contribution.presentation_identity.step_name == "UnifiedNeurons"
    assert contribution.supporting_presentation_identities[0].output_key == (
        NEURITE_BRANCHPOINT_IMAGE_NAME
    )
    assert {
        path.name for path in contribution.plate_path.iterdir() if path.is_file()
    } == set(source_filenames)


def test_neurite_example_is_discoverable_through_existing_corpus_authority() -> None:
    service = KnowledgeBaseService()
    index = service.get_document(
        KnowledgeBaseDocumentRequest.from_fields(
            document_id="openhcs_example_corpus_map",
            section_id="native-example-source-index",
            max_chars=20_000,
        )
    )
    search = service.search(
        KnowledgeBaseSearchRequest(
            query="MAP2 SMI312 UnifiedNeurons neurite outgrowth Python example",
            limit=5,
        )
    )

    source_path = (
        "openhcs/processing/presets/pipelines/loose_operaphenix_neurite_outgrowth.py"
    )
    assert source_path in index.content
    assert any(
        hit.document.document_id == "openhcs_example_corpus_map"
        and hit.section is not None
        and hit.section.section_id
        == "openhcs-processing-presets-pipelines-loose-operaphenix-neurite-outgrowth-py"
        for hit in search.hits
    )
