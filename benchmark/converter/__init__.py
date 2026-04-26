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
from .settings_binder import SettingsBinder, bind_settings
from .symbol_table import (
    CellProfilerSymbol,
    CellProfilerSymbolKind,
    CellProfilerSymbolTable,
    ModuleArtifactContracts,
)

__all__ = [
    # Core
    'CPPipeParser',
    'ModuleBlock',
    'PipelineGenerator',

    # Absorption
    'LibraryAbsorber',
    'LLMFunctionConverter',
    'SourceLocator',

    # Utilities
    'ContractInference',
    'infer_contract',
    'SettingsBinder',
    'bind_settings',
    'CellProfilerSymbol',
    'CellProfilerSymbolKind',
    'CellProfilerSymbolTable',
    'ModuleArtifactContracts',
]
