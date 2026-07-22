"""Compact MetaXpress-style analysis for a loose Opera Phenix image export.

This is the one-step counterpart to the modular CellProfiler example in
``loose_operaphenix_neurite_outgrowth.py``. Both use the same exact loose-file
identity boundary. The compact callable internally composes the accurate
CellProfiler-compatible segmentation and skeleton leaves while preserving the
smaller MetaXpress-style public API and typed outputs.

Use a complete Opera Phenix plate with its ``Index.xml`` through the native
``Microscope.OPERAPHENIX`` handler. Source bindings are only needed here because
the selected TIFFs were copied away from that plate metadata.
"""

from pathlib import Path

from openhcs.constants.constants import GroupBy, VariableComponents
from openhcs.constants.input_source import InputSource
from openhcs.core.config import (
    LazyNapariStreamingConfig,
    LazyProcessingConfig,
    NapariColormap,
    PipelineConfig,
)
from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.backends.analysis.neurite_outgrowth import (
    MetaXpressCellBodySettings,
    MetaXpressNuclearSettings,
    MetaXpressOutgrowthSettings,
    NeuriteIllumination,
    neurite_outgrowth_metaxpress,
)

from .loose_operaphenix_neurite_outgrowth import (
    LooseOperaPhenixNeuriteInputs,
    SemanticImageSource,
    build_loose_operaphenix_neurite_config,
)


def build_loose_operaphenix_neurite_metaxpress_pipeline(
    inputs: LooseOperaPhenixNeuriteInputs,
) -> tuple[PipelineConfig, list[FunctionStep]]:
    """Build the compact MAP2-body/SMI312-neurite/Hoechst-nuclei workflow."""

    pipeline_config = build_loose_operaphenix_neurite_config(inputs)
    step = FunctionStep(
        name="CompactMetaXpressNeuriteOutgrowth",
        func=(
            neurite_outgrowth_metaxpress,
            {
                "neurite_channel_index": inputs.channel_index(inputs.smi312),
                "illumination": NeuriteIllumination.FLUORESCENCE,
                "cell_body": MetaXpressCellBodySettings(
                    approximate_max_width=30.0,
                    minimum_area=50.0,
                    intensity_above_local_background=1000.0,
                    channel_index=inputs.channel_index(inputs.map2),
                ),
                "outgrowth": MetaXpressOutgrowthSettings(
                    maximum_width=4.0,
                    intensity_above_local_background=50.0,
                    minimum_cell_growth_to_log_as_significant=10.0,
                ),
                "use_nuclear_stain": True,
                "nuclear_stain": MetaXpressNuclearSettings(
                    channel_index=inputs.channel_index(inputs.hoechst),
                    approx_min_width=5.0,
                    approx_max_width=30.0,
                    intensity_above_local_background=5000.0,
                ),
            },
        ),
        processing_config=LazyProcessingConfig(
            variable_components=[VariableComponents.CHANNEL],
            group_by=GroupBy.NONE,
            input_source=InputSource.PIPELINE_START,
        ),
        napari_streaming_config=LazyNapariStreamingConfig(
            enabled=True,
            persistent=True,
            port=inputs.viewer_port,
            colormap=NapariColormap.MAGMA,
        ),
    )
    return pipeline_config, [step]


example_inputs = LooseOperaPhenixNeuriteInputs(
    plate_path=Path("path/to/loose_operaphenix_export"),
    output_root=Path("openhcs_neurite_metaxpress_output"),
    well="R04C09",
    site="11",
    z_index="1",
    timepoint="1",
    viewer_port=5888,
    hoechst=SemanticImageSource(
        alias="Hoechst",
        filename="r04c09f11p01-ch1sk1fk1fl1.tiff",
        channel="1",
    ),
    map2=SemanticImageSource(
        alias="MAP2",
        filename="r04c09f11p01-ch2sk1fk1fl1.tiff",
        channel="2",
    ),
    smi312=SemanticImageSource(
        alias="SMI312",
        filename="r04c09f11p01-ch4sk1fk1fl1.tiff",
        channel="4",
    ),
)

plate_path = example_inputs.plate_path.expanduser().resolve()
pipeline_config, pipeline_steps = build_loose_operaphenix_neurite_metaxpress_pipeline(
    example_inputs
)
