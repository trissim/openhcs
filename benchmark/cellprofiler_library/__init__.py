"""Compatibility import path for absorbed CellProfiler function registry."""

from openhcs.processing.backends.cellprofiler.library import (
    AbsorbedFunctionLocation,
    AbsorbedFunctionMetadata,
    canonical_module_name,
    coerce_absorbed_processing_contract,
    coerce_registered_absorbed_processing_contract,
    function_inventory,
    get_contract,
    get_function,
    list_modules,
    require_function,
    validated_contracts,
)

__all__ = (
    "AbsorbedFunctionLocation",
    "AbsorbedFunctionMetadata",
    "canonical_module_name",
    "coerce_absorbed_processing_contract",
    "coerce_registered_absorbed_processing_contract",
    "function_inventory",
    "get_contract",
    "get_function",
    "list_modules",
    "require_function",
    "validated_contracts",
)
