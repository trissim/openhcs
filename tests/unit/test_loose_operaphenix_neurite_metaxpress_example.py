from multiprocessing import SimpleQueue
from pathlib import Path

import numpy as np
from objectstate import ObjectStateRegistry
import tifffile

from openhcs.agent.dto.knowledge import KnowledgeBaseDocumentRequest
from openhcs.agent.services.knowledge_base_service import KnowledgeBaseService
from openhcs.config_framework.lazy_factory import ensure_global_config_context
from openhcs.constants import GroupBy, Microscope, VariableComponents
from openhcs.constants.input_source import InputSource
from openhcs.core.config import GlobalPipelineConfig
from openhcs.core.function_patterns import get_core_callable
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.progress import set_progress_queue
from openhcs.processing.backends.analysis.neurite_outgrowth import (
    MetaXpressCellBodySettings,
    MetaXpressNuclearSettings,
    MetaXpressOutgrowthSettings,
    neurite_outgrowth_metaxpress,
)
from openhcs.processing.presets.pipelines import (
    loose_operaphenix_neurite_outgrowth_metaxpress as example_module,
)
from openhcs.processing.presets.pipelines.loose_operaphenix_neurite_outgrowth import (
    LooseOperaPhenixNeuriteInputs,
    SemanticImageSource,
)
from openhcs.processing.presets.pipelines.loose_operaphenix_neurite_outgrowth_metaxpress import (
    build_loose_operaphenix_neurite_metaxpress_pipeline,
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


def test_compact_example_uses_owned_channel_order_and_one_function_step(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path / "plate", tmp_path / "output")
    pipeline_config, steps = build_loose_operaphenix_neurite_metaxpress_pipeline(inputs)

    assert inputs.channel_stack == (inputs.hoechst, inputs.map2, inputs.smi312)
    assert [inputs.channel_index(source) for source in inputs.channel_stack] == [
        0,
        1,
        2,
    ]
    assert pipeline_config.microscope is Microscope.SOURCE_BINDINGS
    assert pipeline_config.well_filter_config.well_filter == "B03"
    assert pipeline_config.path_planning_config.well_filter == 0
    assert [
        binding.alias for binding in pipeline_config.source_bindings_config.bindings
    ] == [
        "Hoechst",
        "MAP2",
        "SMI312",
    ]

    assert len(steps) == 1
    step = steps[0]
    assert step.name == "CompactMetaXpressNeuriteOutgrowth"
    assert get_core_callable(step.func) is neurite_outgrowth_metaxpress
    assert step.processing_config.variable_components == [VariableComponents.CHANNEL]
    assert step.processing_config.group_by is GroupBy.NONE
    assert step.processing_config.input_source is InputSource.PIPELINE_START
    assert step.napari_streaming_config.enabled is True
    assert step.napari_streaming_config.well_filter is None
    assert step.napari_streaming_config.port == 5999

    kwargs = step.func[1]
    assert kwargs["neurite_channel_index"] == 2
    assert kwargs["cell_body"] == MetaXpressCellBodySettings(
        approximate_max_width=30.0,
        minimum_area=50.0,
        intensity_above_local_background=1000.0,
        channel_index=1,
    )
    assert kwargs["outgrowth"] == MetaXpressOutgrowthSettings(
        maximum_width=4.0,
        intensity_above_local_background=50.0,
        minimum_cell_growth_to_log_as_significant=10.0,
    )
    assert kwargs["use_nuclear_stain"] is True
    assert kwargs["nuclear_stain"] == MetaXpressNuclearSettings(
        channel_index=0,
        approx_min_width=5.0,
        approx_max_width=30.0,
        intensity_above_local_background=5000.0,
    )
    assert "/tmp/" not in Path(example_module.__file__).read_text(encoding="utf-8")

    source_index = KnowledgeBaseService().get_document(
        KnowledgeBaseDocumentRequest.from_fields(
            document_id="openhcs_example_corpus_map",
            section_id="native-example-source-index",
            max_chars=20_000,
        )
    )
    assert (
        "openhcs/processing/presets/pipelines/"
        "loose_operaphenix_neurite_outgrowth_metaxpress.py"
    ) in source_index.content


def test_compact_example_compiles_the_full_ordered_channel_stack(
    tmp_path: Path,
) -> None:
    plate_path = tmp_path / "plate"
    plate_path.mkdir()
    inputs = _inputs(plate_path, tmp_path / "output")
    for source in inputs.channel_stack:
        tifffile.imwrite(
            plate_path / source.filename,
            np.ones((32, 32), dtype=np.uint16),
        )

    pipeline_config, steps = build_loose_operaphenix_neurite_metaxpress_pipeline(inputs)
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
    plan = context.step_plans[0]
    assert plan.step_name == "CompactMetaXpressNeuriteOutgrowth"
    assert tuple(plan.variable_components) == (VariableComponents.CHANNEL,)
    assert plan.compiled_function_pattern is not None
