"""Product-owned access point for absorbed CellProfiler function metadata.

The absorbed function implementation tree is still physically hosted under the
benchmark package during the compiler/runtime surface migration.  This module
is the OpenHCS-owned boundary that product backend code depends on; benchmark
imports can then be collapsed behind the compatibility layer instead of leaking
through backend registration.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import Any

from openhcs.processing.backends.lib_registry.unified_registry import (
    ProcessingContract,
)

_ABSORBED_LIBRARY_MODULE = "benchmark.cellprofiler_library"
_absorbed_library = importlib.import_module(_ABSORBED_LIBRARY_MODULE)

AbsorbedFunctionLocation = _absorbed_library.AbsorbedFunctionLocation
AbsorbedFunctionMetadata = _absorbed_library.AbsorbedFunctionMetadata


def canonical_module_name(module_name: str) -> str:
    """Return the canonical absorbed module name for a CellProfiler module."""
    return _absorbed_library.canonical_module_name(module_name)


def get_function(
    module_name: str,
    *,
    function_name: str | None = None,
) -> Callable[..., Any] | None:
    """Return the absorbed function for a CellProfiler module, if registered."""
    return _absorbed_library.get_function(module_name, function_name=function_name)


def require_function(
    module_name: str,
    *,
    function_name: str | None = None,
) -> Callable[..., Any]:
    """Return one absorbed function or raise a precise registry error."""
    return _absorbed_library.require_function(
        module_name,
        function_name=function_name,
    )


def get_contract(module_name: str) -> dict[str, Any] | None:
    """Return contract metadata for one absorbed CellProfiler module."""
    return _absorbed_library.get_contract(module_name)


def validated_contracts() -> Mapping[str, dict[str, Any]]:
    """Return validated absorbed module contracts keyed by canonical module name."""
    return {
        module_name: contract
        for module_name in list_modules()
        if (contract := get_contract(module_name)) is not None
        and contract.get("validated", False)
    }


def list_modules() -> list[str]:
    """List absorbed CellProfiler module names."""
    return _absorbed_library.list_modules()


def function_inventory() -> Mapping[str, Any]:
    """Return the derived absorbed function location index."""
    return _absorbed_library.function_inventory()


def coerce_absorbed_processing_contract(
    module_name: str,
    function_name: str,
    function: Callable[..., Any],
) -> ProcessingContract | None:
    """Return or install nominal processing metadata for an executable function."""
    return _absorbed_library.coerce_absorbed_processing_contract(
        module_name,
        function_name,
        function,
    )


def coerce_registered_absorbed_processing_contract(
    function_name: str,
    function: Callable[..., Any],
) -> ProcessingContract | None:
    """Install nominal processing metadata for a registered absorbed function."""
    return _absorbed_library.coerce_registered_absorbed_processing_contract(
        function_name,
        function,
    )
