"""Compatibility aliases for CellProfiler symbol-table compilation."""

from openhcs.interop.cellprofiler.symbol_table import (
    IDENTIFY_PRIMARY_OUTPUT_OBJECTS_SETTING,
    INPUT_IMAGE_SETTING,
    INPUT_OBJECTS_SETTING,
    OUTPUT_IMAGE_SETTING,
    OUTPUT_OBJECTS_SETTING,
    CellProfilerSymbol,
    CellProfilerSymbolKind,
    CellProfilerSymbolTable,
    ModuleArtifactContracts,
    ModuleContractBuilder,
    module_contract_literal,
    source_bindings_literal,
)

__all__ = (
    "IDENTIFY_PRIMARY_OUTPUT_OBJECTS_SETTING",
    "INPUT_IMAGE_SETTING",
    "INPUT_OBJECTS_SETTING",
    "OUTPUT_IMAGE_SETTING",
    "OUTPUT_OBJECTS_SETTING",
    "CellProfilerSymbol",
    "CellProfilerSymbolKind",
    "CellProfilerSymbolTable",
    "ModuleArtifactContracts",
    "ModuleContractBuilder",
    "module_contract_literal",
    "source_bindings_literal",
)
