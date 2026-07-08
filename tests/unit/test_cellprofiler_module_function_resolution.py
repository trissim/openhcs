"""Focused tests for CellProfiler module function-resolution policies."""

from openhcs.interop.cellprofiler import module_function_resolution
from openhcs.constants.constants import AllComponents
from openhcs.interop.cellprofiler.parser import ModuleBlock
from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
from openhcs.interop.cellprofiler.module_processing_components import (
    GeneratedStepSettings,
    ModuleProcessingComponentRequest,
    RuntimeArtifactLineageScope,
)
from openhcs.interop.cellprofiler.symbol_table import ModuleArtifactContracts
from openhcs.core.pipeline_image_schema import PipelineImageSchema
from openhcs.processing.backends.cellprofiler.morphology import DilateObjectsModule


def _module(name: str, settings: dict[str, str]) -> ModuleBlock:
    return ModuleBlock(name=name, module_num=1, settings=settings)


def _resolved_function_name(
    module: ModuleBlock, default_function_name: str = "default_function"
) -> str:
    module_class = CellProfilerModule.for_module(module.name)
    assert module_class is not None
    return module_class.resolve_function(
        module, default_function_name=default_function_name
    ).function_name


def test_function_resolution_has_no_parallel_rule_table() -> None:
    assert "MODULE_FUNCTION_RESOLUTION_RULES" not in vars(module_function_resolution)
    assert not any(
        (
            name.endswith("ModuleFunctionResolutionStrategy")
            for name in vars(module_function_resolution)
        )
    )


def test_measure_texture_resolution_is_declared_on_module_class() -> None:
    assert (
        _resolved_function_name(
            _module(
                "MeasureTexture",
                {
                    "Measure images or objects?": "Objects",
                    "Select objects to measure": "Nuclei",
                },
            )
        )
        == "measure_texture_objects"
    )
    assert (
        _resolved_function_name(
            _module(
                "MeasureTexture",
                {
                    "Measure images or objects?": "Images",
                    "Select objects to measure": "Nuclei",
                },
            )
        )
        == "default_function"
    )


def test_measure_colocalization_resolution_is_declared_on_module_class() -> None:
    assert (
        _resolved_function_name(
            _module(
                "MeasureColocalization",
                {
                    "Select where to measure correlation": "Both",
                    "Select objects to measure": "Nuclei",
                },
            )
        )
        == "measure_colocalization_objects"
    )


def test_measure_granularity_resolution_is_declared_on_module_class() -> None:
    assert (
        _resolved_function_name(
            _module("MeasureGranularity", {"Select objects to measure": "Nuclei"})
        )
        == "measure_granularity_objects"
    )


def test_resize_resolution_is_declared_on_module_class() -> None:
    assert (
        _resolved_function_name(_module("Resize", {"Z Resizing factor": "1.0"}))
        == "resize_volumetric"
    )


def test_resize_objects_resolution_is_declared_on_module_class() -> None:
    assert (
        _resolved_function_name(_module("ResizeObjects", {"Planes (Z)": "10"}))
        == "resize_objects_3d"
    )


def test_dilate_objects_uses_3d_variant_for_z_stack_lineage() -> None:
    request = ModuleProcessingComponentRequest(
        module_type=DilateObjectsModule,
        function_name="dilate_objects",
        runtime_lineage=RuntimeArtifactLineageScope(
            ModuleArtifactContracts("DilateObjects", 1),
            (AllComponents.Z_INDEX,),
        ),
        bound_settings=GeneratedStepSettings(),
        source_schema=PipelineImageSchema.empty(),
    )

    resolved = DilateObjectsModule.resolve_semantic_function(
        _module("DilateObjects", {}),
        default_function_name="dilate_objects",
        request=request,
    )

    assert resolved.function_name == "dilate_objects_3d"
