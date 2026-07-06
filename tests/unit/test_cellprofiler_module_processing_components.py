import pytest

from openhcs.constants.constants import AllComponents
from openhcs.constants.constants import GroupBy
from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
from openhcs.interop.cellprofiler.module_processing_components import (
    GeneratedStepSettings,
    ModuleProcessingComponents,
    ModuleProcessingComponentRequest,
    RuntimeArtifactLineageScope,
)
from openhcs.interop.cellprofiler.symbol_table import (
    CellProfilerSymbol,
    ModuleArtifactContracts,
)
from openhcs.core.artifacts import ArtifactSpec, ImageArtifactType, ObjectLabelsArtifactType
from openhcs.core.pipeline_image_schema import PipelineImageSchema, SourceImageStackPlan
from openhcs.core.source_bindings import (
    MetadataSelector,
    NamedSourceBinding,
    SourceSelector,
    StepSourceBindingsConfig,
)
from openhcs.processing.backends.cellprofiler.intensity import (
    MeasureObjectIntensityModule,
)
from openhcs.processing.backends.cellprofiler.image_geometry import MaskImageModule
from openhcs.processing.backends.cellprofiler.morphology import ResizeObjectsModule


def test_generated_group_by_collapses_variable_component_conflict_to_none():
    components = ModuleProcessingComponents((AllComponents.CHANNEL,))

    assert CellProfilerModule.generated_group_by(components) is GroupBy.NONE


def test_plane_runtime_artifact_modules_preserve_stack_axes_as_variable_components():
    contract = ModuleArtifactContracts(
        "ResizeObjects",
        2,
        input_symbols=(
            CellProfilerSymbol(ArtifactSpec.input("Parents", ObjectLabelsArtifactType), producer_module_num=1),
        ),
    )
    request = ModuleProcessingComponentRequest(
        module_type=ResizeObjectsModule,
        function_name="resize_objects",
        runtime_lineage=RuntimeArtifactLineageScope(
            contract,
            (AllComponents.SITE, AllComponents.Z_INDEX),
        ),
        bound_settings=GeneratedStepSettings(),
        source_schema=PipelineImageSchema.empty(),
    )

    components = ResizeObjectsModule.processing_components(request)

    assert components.variable_components == (AllComponents.SITE, AllComponents.Z_INDEX)
    assert components.group_by_component is GroupBy.CHANNEL


def test_runtime_image_artifact_inputs_keep_channel_batch_identity():
    contract = ModuleArtifactContracts(
        "MaskImage",
        2,
        input_symbols=(
            CellProfilerSymbol(ArtifactSpec.input("BF_image", ImageArtifactType), source_bound=True),
            CellProfilerSymbol(ArtifactSpec.input("Mask_image", ImageArtifactType), producer_module_num=1),
        ),
    )
    request = ModuleProcessingComponentRequest(
        module_type=MaskImageModule,
        function_name="mask_image",
        runtime_lineage=RuntimeArtifactLineageScope(
            contract,
            (AllComponents.SITE,),
        ),
        bound_settings=GeneratedStepSettings(),
        source_schema=PipelineImageSchema.empty(),
    )

    components = MaskImageModule.processing_components(request)

    assert components.variable_components == (AllComponents.SITE,)
    assert components.group_by_component is GroupBy.CHANNEL


def test_source_bound_runtime_artifact_modules_use_source_stack_axis():
    contract = ModuleArtifactContracts(
        "MeasureObjectIntensity",
        26,
        input_symbols=(
            CellProfilerSymbol(
                ArtifactSpec.input("DNA", ImageArtifactType),
                source_bound=True,
            ),
            CellProfilerSymbol(
                ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),
                producer_module_num=11,
            ),
        ),
        source_bindings=StepSourceBindingsConfig(
            enabled=True,
            bindings=(
                NamedSourceBinding(
                    alias="DNA",
                    selector=SourceSelector(
                        metadata=(MetadataSelector("ChannelNumber", "2"),),
                    ),
                ),
            ),
        ),
    )
    request = ModuleProcessingComponentRequest(
        module_type=MeasureObjectIntensityModule,
        function_name="measure_object_intensity",
        runtime_lineage=RuntimeArtifactLineageScope(
            contract,
            (AllComponents.SITE, AllComponents.Z_INDEX),
        ),
        bound_settings=GeneratedStepSettings(),
        source_schema=PipelineImageSchema(
            source_image_stack=SourceImageStackPlan((AllComponents.Z_INDEX,)),
        ),
    )

    components = MeasureObjectIntensityModule.processing_components(request)

    assert components.variable_components == (AllComponents.Z_INDEX,)
    assert components.group_by_component is AllComponents.CHANNEL
