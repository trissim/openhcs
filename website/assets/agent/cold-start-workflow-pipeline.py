from pathlib import Path

from openhcs.constants.constants import GroupBy, VariableComponents
from openhcs.constants.input_source import InputSource
from openhcs.core.config import (
    GlobalPipelineConfig,
    LazyAnalysisConsolidationConfig,
    LazyNapariStreamingConfig,
    LazyPathPlanningConfig,
    LazyProcessingConfig,
    LazyStepMaterializationConfig,
    NapariDimensionMode,
    PipelineConfig,
)
from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.backends.analysis.count_cells_simple import (
    count_cells_simple_dual_channel,
)
from zmqruntime.config import TransportMode

root = Path(
    "/home/ts/code/projects/openhcs/mcp_outputs/"
    "website-agent-demo/candidate-20260804-08"
)
plate_path = root / "plate"
output_root = root / "outputs"

plate_paths = [plate_path]
global_config = GlobalPipelineConfig()

per_plate_configs = {
    plate_path: PipelineConfig(
        materialization_results_path=output_root / "measurements",
        materialize_runtime_artifacts=True,
        auto_add_output_plate_to_plate_manager=True,
        processing_config=LazyProcessingConfig(
            variable_components=[VariableComponents.CHANNEL],
            group_by=GroupBy.NONE,
            input_source=InputSource.PIPELINE_START,
        ),
        analysis_consolidation_config=LazyAnalysisConsolidationConfig(
            enabled=True,
            metaxpress_style=True,
        ),
        path_planning_config=LazyPathPlanningConfig(
            global_output_folder=output_root,
            sub_dir="images",
        ),
        step_materialization_config=LazyStepMaterializationConfig(
            global_output_folder=output_root,
            sub_dir="review_images",
            enabled=True,
        ),
        napari_streaming_config=LazyNapariStreamingConfig(
            site_mode=NapariDimensionMode.STACK,
            channel_mode=NapariDimensionMode.STACK,
            z_index_mode=NapariDimensionMode.STACK,
            timepoint_mode=NapariDimensionMode.STACK,
            well_mode=NapariDimensionMode.STACK,
            enabled=True,
            persistent=True,
            host="localhost",
            transport_mode=TransportMode.IPC,
            port=5598,
        ),
    )
}

pipeline_data = {
    plate_path: [
        FunctionStep(
            func=count_cells_simple_dual_channel,
            name="Segment nuclei and quantify W2-positive cells",
            description=(
                "Channel 1 nuclei define total cells; channel 2 stained area "
                "scores marker-positive cells and emits typed measurements "
                "plus W1/W2 label ROIs."
            ),
            processing_config=LazyProcessingConfig(
                variable_components=[VariableComponents.CHANNEL],
                group_by=GroupBy.NONE,
                input_source=InputSource.PIPELINE_START,
            ),
            step_materialization_config=LazyStepMaterializationConfig(
                global_output_folder=output_root,
                sub_dir="review_images",
                enabled=True,
            ),
        )
    ]
}
