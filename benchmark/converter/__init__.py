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

from openhcs.interop.cellprofiler.parser import CPPipeParser, ModuleBlock
from openhcs.interop.cellprofiler.pipeline_generator import PipelineGenerator
from openhcs.interop.cellprofiler.settings_binder import SettingsBinder
from openhcs.interop.cellprofiler.source_schema import compile_image_schema
from openhcs.interop.cellprofiler.symbol_table import (
    CellProfilerSymbol,
    CellProfilerSymbolKind,
    CellProfilerSymbolTable,
    ModuleArtifactContracts,
)
from openhcs.core.public_api import public_names_from_objects

from .source_locator import SourceLocator
from .llm_converter import LLMFunctionConverter
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
    prepare_generated_pipeline,
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


__all__ = public_names_from_objects(
    SourceLocator,
    LLMFunctionConverter,
    LibraryAbsorber,
    ContractInference,
    infer_contract,
    CPPipeModulePartition,
    CPPipePipelineGenerationRequest,
    CPPipePipelinePreparationRequest,
    DirectPipelineExecution,
    GeneratedCPPipePipeline,
    PreparedGeneratedPipeline,
    execute_pipeline_direct,
    prepare_generated_pipeline,
    CellProfilerSymbol,
    CellProfilerSymbolKind,
    CellProfilerSymbolTable,
    ModuleArtifactContracts,
    CPPipeParser,
    ModuleBlock,
    PipelineGenerator,
    SettingsBinder,
    compile_image_schema,
    GroupingPlan,
    ImageAssignment,
    ImagesRule,
    PipelineImageSchema,
    MetadataExtractionRule,
    MetadataSource,
)
