from openhcs.constants.constants import AllComponents
from openhcs.interop.cellprofiler.module_processing_components import (
    GeneratedStepSettings,
    ModuleProcessingComponentRequest,
    RuntimeArtifactLineageScope,
)
from openhcs.interop.cellprofiler.symbol_table import (
    CellProfilerSymbol,
    CellProfilerSymbolKind,
    ModuleArtifactContracts,
)
from openhcs.core.pipeline_image_schema import PipelineImageSchema
from openhcs.processing.backends.cellprofiler.morphology import ResizeObjectsModule


def test_plane_runtime_artifact_modules_preserve_stack_axes_as_variable_components():
    contract = ModuleArtifactContracts(
        "ResizeObjects",
        2,
        input_symbols=(
            CellProfilerSymbol(
                "Parents",
                CellProfilerSymbolKind.OBJECTS,
                producer_module_num=1,
            ),
        ),
    )
    request = ModuleProcessingComponentRequest(
        category="object_processing",
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
    assert components.group_by_component is None
