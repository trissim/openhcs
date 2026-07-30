"""End-to-end compiler coverage for unnamed native main-flow provenance."""

from multiprocessing import SimpleQueue

import numpy as np
from objectstate import ObjectStateRegistry
import tifffile

from objectstate.lazy_factory import ensure_global_config_context
from openhcs.constants import AllComponents, GroupBy, Microscope, VariableComponents
from openhcs.constants.input_source import InputSource
from openhcs.core.callable_contract import FunctionStepExecutionScope
from openhcs.core.config import GlobalPipelineConfig, LazyProcessingConfig, PipelineConfig
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.pipeline.path_planner import PathPlannerGroupScope
from openhcs.core.progress import set_progress_queue
from openhcs.core.step_dependencies import StepInputDependency
from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.backends.cellprofiler.thresholding import threshold
from openhcs.processing.backends.processors.numpy_processor import percentile_normalize


def test_native_grouped_main_flow_compiles_into_cellprofiler_consumer(tmp_path):
    image_dir = tmp_path / "TimePoint_1"
    image_dir.mkdir()
    (tmp_path / "plate.HTD").write_text(
        '\n'.join(
            ('"XSites", 1', '"YSites", 1', '"PixelSizeUM", 1.0')
        ),
        encoding="utf-8",
    )
    for channel in (1, 2, 4):
        tifffile.imwrite(
            image_dir / f"A01_s001_w{channel}_z001_t001.tif",
            np.full((8, 8), channel, dtype=np.uint16),
        )

    global_config = GlobalPipelineConfig(
        microscope=Microscope.IMAGEXPRESS,
        num_workers=1,
    )
    pipeline_start_processing = LazyProcessingConfig(
        variable_components=[VariableComponents.SITE],
        group_by=GroupBy.CHANNEL,
        input_source=InputSource.PIPELINE_START,
    )
    previous_step_processing = LazyProcessingConfig(
        variable_components=[VariableComponents.SITE],
        group_by=GroupBy.CHANNEL,
        input_source=InputSource.PREVIOUS_STEP,
    )
    steps = [
        FunctionStep(
            func=percentile_normalize,
            name="percentile_normalize",
            processing_config=pipeline_start_processing,
        ),
        FunctionStep(
            func=(threshold, {"name_the_output_image": "Thresholded"}),
            name="Threshold",
            processing_config=previous_step_processing,
        ),
    ]

    ObjectStateRegistry.clear()
    set_progress_queue(SimpleQueue())
    try:
        ensure_global_config_context(GlobalPipelineConfig, global_config)
        orchestrator = PipelineOrchestrator(
            tmp_path,
            pipeline_config=PipelineConfig(),
        ).initialize()
        compilation = orchestrator.compile_pipelines(
            pipeline_definition=steps,
            well_filter=["A01"],
            is_zmq_execution=True,
        )
    finally:
        set_progress_queue(None)

    context = compilation["execution_bundle"].runtime_contexts["A01"]
    native_plan = context.step_plans[0]
    threshold_plan = context.step_plans[1]
    assert threshold_plan.main_input_dependency == StepInputDependency.step_output(
        source_step_index=0,
        source_step_scope_id=native_plan.step_scope_id,
    )
    assert threshold_plan.execution_group_scope == PathPlannerGroupScope.from_raw(
        ("1", "2", "4"),
        component=AllComponents.CHANNEL,
    )
    assert threshold_plan.compiled_function_pattern is not None
    invocation = next(
        threshold_plan.compiled_function_pattern.iter_invocations()
    )
    (edge,) = invocation.artifact_input_edges
    assert edge.consumes_main_flow
    assert edge.storage_plan is None
    assert invocation.contract.execution_scope is FunctionStepExecutionScope.AXIS
