# OpenHCS pipeline

from openhcs.constants.constants import (
    AllComponents,
    VariableComponents,
)
from openhcs.core.config import (
    LazyNapariStreamingConfig,
    LazyPathPlanningConfig,
    LazyProcessingConfig,
    LazyStepMaterializationConfig,
    PipelineConfig,
)
from openhcs.core.source_bindings import (
    ComponentSelector,
    LazySourceBindingsConfig,
    NamedSourceBinding,
    SourceFilterClause,
    SourceFilterMatchType,
    SourceFilterSubject,
    SourceSelector,
)
from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.backends.analysis.neurite_outgrowth import (
    MetaXpressCellBodySettings,
    MetaXpressNuclearSettings,
    MetaXpressOutgrowthSettings,
    neurite_outgrowth_metaxpress,
)
from openhcs.processing.backends.processors.numpy_processor import percentile_normalize
from pathlib import Path

path_root = Path('/home/ts/code/projects/openhcs/mcp_outputs/website-agent-demo/candidate-20260804-13/outputs')

pipeline_config = PipelineConfig(
    materialization_results_path=path_root / 'results',
    materialize_runtime_artifacts=True,
    source_bindings_config=LazySourceBindingsConfig(
        bindings=(
            NamedSourceBinding(
                alias='W1_neuron_neurite',
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            subject=SourceFilterSubject.FILE,
                            match_type=SourceFilterMatchType.EQUALS,
                            value='1_w1.tif'
                        ),
                    )
                ),
                component_identity=(
                    ComponentSelector(
                        component=AllComponents.WELL,
                        value='1'
                    ),
                    ComponentSelector(
                        component=AllComponents.SITE,
                        value='1'
                    ),
                    ComponentSelector(
                        component=AllComponents.CHANNEL,
                        value='1'
                    ),
                    ComponentSelector(
                        component=AllComponents.Z_INDEX,
                        value='1'
                    ),
                    ComponentSelector(
                        component=AllComponents.TIMEPOINT,
                        value='1'
                    )
                )
            ),
            NamedSourceBinding(
                alias='W2_soma_nuclear',
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            subject=SourceFilterSubject.FILE,
                            match_type=SourceFilterMatchType.EQUALS,
                            value='1_w2.tif'
                        ),
                    )
                ),
                component_identity=(
                    ComponentSelector(
                        component=AllComponents.WELL,
                        value='1'
                    ),
                    ComponentSelector(
                        component=AllComponents.SITE,
                        value='1'
                    ),
                    ComponentSelector(
                        component=AllComponents.CHANNEL,
                        value='2'
                    ),
                    ComponentSelector(
                        component=AllComponents.Z_INDEX,
                        value='1'
                    ),
                    ComponentSelector(
                        component=AllComponents.TIMEPOINT,
                        value='1'
                    )
                )
            )
        )
    ),
    path_planning_config=LazyPathPlanningConfig(
        well_filter=0,
        global_output_folder=path_root
    )
)

pipeline_steps = [
    FunctionStep(
        func=(percentile_normalize, {
                'target_max': 255.0,
                'high_percentile': 99.8
            }),
        name='Enhanced neuronal signal',
        processing_config=LazyProcessingConfig(
            variable_components=[
                VariableComponents.CHANNEL
            ]
        ),
        step_materialization_config=LazyStepMaterializationConfig(
            global_output_folder=path_root,
            sub_dir='review_images',
            enabled=True
        ),
        napari_streaming_config=LazyNapariStreamingConfig(
            colormap='magma',
            enabled=True,
            persistent=True,
            port=5613
        )
    ),
    FunctionStep(
        func=(neurite_outgrowth_metaxpress, {
                'use_nuclear_stain': True,
                'nuclear_stain': MetaXpressNuclearSettings(
                    approx_max_width=32.0,
                    intensity_above_local_background=18.0
                ),
                'cell_body': MetaXpressCellBodySettings(
                    approximate_max_width=36.0,
                    minimum_area=45.0,
                    intensity_above_local_background=20.0,
                    channel_index=0
                ),
                'outgrowth': MetaXpressOutgrowthSettings(
                    maximum_width=6.0,
                    intensity_above_local_background=8.0,
                    minimum_cell_growth_to_log_as_significant=8.0
                )
            }),
        name='Per-neuron morphology and topology',
        processing_config=LazyProcessingConfig(
            variable_components=[
                VariableComponents.CHANNEL
            ]
        ),
        napari_streaming_config=LazyNapariStreamingConfig(
            colormap='magma',
            enabled=True,
            persistent=True,
            port=5613
        )
    )
]
