"""CellProfiler neurite outgrowth for a loose Opera Phenix image export.

This example is for selected Opera Phenix TIFFs that were copied without the
plate's ``Index.xml``. A complete Opera Phenix plate should use
``Microscope.OPERAPHENIX`` instead of reconstructing its identities here.

Edit ``example_inputs`` for the local plate, exact filenames, axis identities,
output directory, and viewer port. Set ``map2=None`` for a two-channel workflow
where SMI312 delineates neuronal bodies and neurites; the compact MetaXpress
preset additionally uses Hoechst as nuclear seeds. Provide MAP2 to use its
neuronal-body signal instead. The top-level well filter bounds loading to one
well. Step checkpoint and viewer filters are intentionally unset so they inherit
that same scope; path-planning filter zero suppresses the ordinary final image
copy while typed measurements and object labels remain materialized.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openhcs.constants.constants import AllComponents, Microscope
from openhcs.constants.input_source import InputSource
from openhcs.core.config import (
    LazyNapariStreamingConfig,
    NapariColormap,
    LazyPathPlanningConfig,
    LazyProcessingConfig,
    LazyStepMaterializationConfig,
    LazyWellFilterConfig,
    PipelineConfig,
)
from openhcs.core.source_bindings import (
    ComponentSelector,
    LazyStepSourceBindingsConfig,
    NamedSourceBinding,
    SourceBindingsConfig,
    SourceFilterClause,
    SourceFilterMatchType,
    SourceFilterSubject,
    SourceSelector,
)
from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.backends.analysis.neurite_outgrowth import (
    CELLPROFILER_NEURITE_ENGINE_PROFILE,
)
from openhcs.processing.backends.cellprofiler.feature_enhancement import (
    enhance_or_suppress_features,
)
from openhcs.processing.backends.cellprofiler.intensity import measure_image_intensity
from openhcs.processing.backends.cellprofiler.medial_axis import medialaxis
from openhcs.processing.backends.cellprofiler.primary_objects import (
    identify_primary_objects,
)
from openhcs.processing.backends.cellprofiler.secondary import (
    identify_secondary_objects,
)
from openhcs.processing.backends.cellprofiler.skeleton import (
    measure_object_skeleton_with_branchpoint_image,
)
from openhcs.processing.backends.cellprofiler.spreadsheet_export import (
    export_to_spreadsheet,
)
from openhcs.processing.backends.cellprofiler.thresholding import (
    threshold,
)


@dataclass(frozen=True, slots=True)
class SemanticImageSource:
    """One semantic channel reconstructed from an exact loose-export file."""

    alias: str
    filename: str
    channel: str


@dataclass(frozen=True, slots=True)
class LooseOperaPhenixNeuriteInputs:
    """Portable input and output boundary for this example pipeline."""

    plate_path: Path
    output_root: Path
    well: str
    site: str
    z_index: str
    timepoint: str
    viewer_port: int
    hoechst: SemanticImageSource
    map2: SemanticImageSource | None
    smi312: SemanticImageSource

    @property
    def cell_body_source(self) -> SemanticImageSource:
        """Use MAP2 bodies when supplied, otherwise use the SMI312 cell signal."""

        return self.smi312 if self.map2 is None else self.map2

    @property
    def channel_stack(self) -> tuple[SemanticImageSource, ...]:
        """Authoritative semantic order of the assembled channel stack."""

        return tuple(
            source
            for source in (self.hoechst, self.map2, self.smi312)
            if source is not None
        )

    def channel_index(self, source: SemanticImageSource) -> int:
        """Resolve a semantic source against the owned assembled-stack order."""

        return self.channel_stack.index(source)


def _exact_image_binding(
    source: SemanticImageSource,
    inputs: LooseOperaPhenixNeuriteInputs,
) -> NamedSourceBinding:
    return NamedSourceBinding(
        alias=source.alias,
        selector=SourceSelector(
            filters=(
                SourceFilterClause(
                    subject=SourceFilterSubject.FILE,
                    match_type=SourceFilterMatchType.EQUALS,
                    value=source.filename,
                ),
            ),
        ),
        component_identity=(
            ComponentSelector(AllComponents.WELL, inputs.well),
            ComponentSelector(AllComponents.SITE, inputs.site),
            ComponentSelector(AllComponents.CHANNEL, source.channel),
            ComponentSelector(AllComponents.Z_INDEX, inputs.z_index),
            ComponentSelector(AllComponents.TIMEPOINT, inputs.timepoint),
        ),
    )


def _named_source(alias: str) -> LazyStepSourceBindingsConfig:
    return LazyStepSourceBindingsConfig(
        enabled=True,
        bindings=(NamedSourceBinding(alias=alias),),
    )


def _qc_stream(
    inputs: LooseOperaPhenixNeuriteInputs,
    colormap: NapariColormap,
) -> LazyNapariStreamingConfig:
    return LazyNapariStreamingConfig(
        enabled=True,
        persistent=True,
        port=inputs.viewer_port,
        colormap=colormap,
    )


def _qc_checkpoint(
    output_root: Path,
    sub_dir: str,
) -> LazyStepMaterializationConfig:
    return LazyStepMaterializationConfig(
        enabled=True,
        global_output_folder=output_root,
        sub_dir=sub_dir,
    )


def build_loose_operaphenix_neurite_config(
    inputs: LooseOperaPhenixNeuriteInputs,
) -> PipelineConfig:
    """Build the shared loose-export source and materialization boundary."""

    output_root = inputs.output_root.expanduser().resolve()
    source_bindings = tuple(
        _exact_image_binding(source, inputs) for source in inputs.channel_stack
    )
    return PipelineConfig(
        microscope=Microscope.SOURCE_BINDINGS,
        well_filter_config=LazyWellFilterConfig(well_filter=inputs.well),
        path_planning_config=LazyPathPlanningConfig(
            well_filter=0,
            global_output_folder=output_root,
        ),
        materialization_results_path=output_root / "results",
        materialize_runtime_artifacts=True,
        source_bindings_config=SourceBindingsConfig(bindings=source_bindings),
    )


def build_loose_operaphenix_neurite_pipeline(
    inputs: LooseOperaPhenixNeuriteInputs,
) -> tuple[PipelineConfig, list[FunctionStep]]:
    """Build an SMI312-neurite workflow with MAP2 or SMI312 cell bodies."""

    output_root = inputs.output_root.expanduser().resolve()
    pipeline_config = build_loose_operaphenix_neurite_config(inputs)

    engine = CELLPROFILER_NEURITE_ENGINE_PROFILE
    pipeline_steps = [
        FunctionStep(
            name="NeuronBodies",
            func=(
                identify_primary_objects,
                engine.body_detection_kwargs(),
            ),
            processing_config=LazyProcessingConfig(
                input_source=InputSource.PIPELINE_START,
            ),
            source_bindings=_named_source(inputs.cell_body_source.alias),
            napari_streaming_config=_qc_stream(inputs, NapariColormap.VIRIDIS),
        ),
        FunctionStep(
            name="SMI312SourceSignal",
            func=(
                measure_image_intensity,
                {
                    "calculate_percentiles": True,
                    "percentiles": "10,50,90",
                },
            ),
            processing_config=LazyProcessingConfig(
                input_source=InputSource.PIPELINE_START,
            ),
            source_bindings=_named_source(inputs.smi312.alias),
            step_materialization_config=_qc_checkpoint(
                output_root,
                "qc_smi312_signal",
            ),
            napari_streaming_config=_qc_stream(inputs, NapariColormap.MAGMA),
        ),
        FunctionStep(
            name="EnhancedNeurites",
            func=(
                enhance_or_suppress_features,
                engine.enhancement_kwargs(),
            ),
        ),
        FunctionStep(
            name="NeuriteForeground",
            func=(
                threshold,
                engine.threshold_kwargs(),
            ),
            step_materialization_config=_qc_checkpoint(
                output_root,
                "qc_neurite_mask",
            ),
            napari_streaming_config=_qc_stream(inputs, NapariColormap.GRAY),
        ),
        FunctionStep(
            name="NeuriteSkeleton",
            func=medialaxis,
            step_materialization_config=_qc_checkpoint(
                output_root,
                "qc_neurite_skeleton",
            ),
            napari_streaming_config=_qc_stream(inputs, NapariColormap.GRAY),
        ),
        FunctionStep(
            name="PerNeuronNeuriteTopology",
            func=(
                measure_object_skeleton_with_branchpoint_image,
                {
                    "fill_small_holes": True,
                    "maximum_hole_size": 10,
                    "branchpoint_image_name": "NeuriteBranchpoints",
                },
            ),
            napari_streaming_config=_qc_stream(inputs, NapariColormap.GRAY),
        ),
        FunctionStep(
            name="UnifiedNeurons",
            func=(
                identify_secondary_objects,
                engine.secondary_kwargs(),
            ),
            processing_config=LazyProcessingConfig(
                input_source=InputSource.PIPELINE_START,
            ),
            source_bindings=_named_source(inputs.smi312.alias),
            napari_streaming_config=_qc_stream(inputs, NapariColormap.MAGMA),
        ),
        FunctionStep(
            name="NeuriteSpreadsheetExport",
            func=(
                export_to_spreadsheet,
                {
                    "add_image_metadata": True,
                    "add_image_file_names": True,
                    "output_directory": "neurite_tables",
                    "export_all_measurement_types": True,
                    "add_filename_prefix": True,
                    "filename_prefix": f"{inputs.well}_Neurite_",
                },
            ),
        ),
    ]
    return pipeline_config, pipeline_steps


# Edit this one boundary rather than searching through the pipeline declarations.
example_inputs = LooseOperaPhenixNeuriteInputs(
    plate_path=Path("path/to/loose_operaphenix_export"),
    output_root=Path("openhcs_neurite_output"),
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
pipeline_config, pipeline_steps = build_loose_operaphenix_neurite_pipeline(
    example_inputs
)
