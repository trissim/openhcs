"""
CellProfiler → OpenHCS Converter

Two commands:
    python -m benchmark.converter.absorb   # One-time: absorb CP library
    python -m benchmark.converter.convert  # Instant: convert .cppipe files

Architecture:
    1. ABSORB: LLM converts entire CP library once → benchmark/cellprofiler_library/
    2. CONVERT: Lookup functions in registry, bind settings, generate pipeline

No fallback. No modes. Absorb first, then convert.
"""

from .parser import CPPipeParser, ModuleBlock
from .source_locator import SourceLocator
from .llm_converter import LLMFunctionConverter
from .pipeline_generator import PipelineGenerator
from .library_absorber import LibraryAbsorber
from .contract_inference import ContractInference, infer_contract
from .runtime_pipeline import (
    CPPipeModulePartition,
    CPPipePipelineGenerationRequest,
    CPPipePipelinePreparationRequest,
    DirectPipelineExecution,
    GeneratedCPPipePipeline,
    PreparedGeneratedPipeline,
    execute_pipeline_direct,
    generate_pipeline_from_cppipe,
    prepare_generated_pipeline,
)
from .settings_binder import SettingsBinder
from .source_schema import (
    compile_image_schema,
)
from openhcs.core.pipeline_image_schema import (
    GroupingPlan,
    ImageAssignment,
    ImagesRule,
    PipelineImageSchema,
)
from openhcs.core.source_bindings import (
    MetadataExtractionRule,
    MetadataSource,
)
from .symbol_table import (
    CellProfilerSymbol,
    CellProfilerSymbolKind,
    CellProfilerSymbolTable,
    ModuleArtifactContracts,
)


def _is_public_api_export(name: str, value: object) -> bool:
    return not name.startswith("_") and (
        getattr(value, "__module__", __name__).startswith("benchmark.converter")
        or name in _COMPATIBILITY_EXPORTS
    )


_COMPATIBILITY_EXPORTS = frozenset(
    {
        "CPPipeParser",
        "GroupingPlan",
        "ImageAssignment",
        "ImagesRule",
        "MetadataExtractionRule",
        "MetadataSource",
        "ModuleBlock",
        "PipelineImageSchema",
        "compile_image_schema",
    }
)


__all__ = sorted(
    name
    for name, value in globals().items()
    if _is_public_api_export(name, value)
)
