"""Resolve absorbed function contract declarations to OpenHCS contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from inspect import unwrap

from openhcs.core.function_contract_metadata import FunctionContractAttribute
from openhcs.processing.backends.cellprofiler import CellProfilerFunctionCatalog
from openhcs.processing.backends.lib_registry.unified_registry import (
    ProcessingContract,
)

UNKNOWN_PROCESSING_CONTRACT_NAME = "unknown"


class ProcessingContractResolutionSource(str, Enum):
    """Authority that resolved one executable processing contract."""

    CALLABLE_METADATA = "callable_metadata"


@dataclass(frozen=True, slots=True)
class ResolvedProcessingContract:
    """Executable OpenHCS contract plus provenance."""

    contract: ProcessingContract
    source: ProcessingContractResolutionSource


def resolve_processing_contract(
    module_name: str,
    function_name: str,
    *,
    function_resolver: Callable[
        ..., Callable[..., object]
    ] = CellProfilerFunctionCatalog.require_function,
) -> ResolvedProcessingContract:
    """Resolve one absorbed module to an executable OpenHCS contract."""
    callable_contract = _callable_processing_contract(
        function_resolver(module_name, function_name=function_name)
    )
    if callable_contract is not None:
        return ResolvedProcessingContract(
            contract=callable_contract,
            source=ProcessingContractResolutionSource.CALLABLE_METADATA,
        )

    raise ValueError(
        f"Module {module_name} resolved executable {function_name} without "
        "__processing_contract__ metadata. Coerce the catalog declaration into "
        "callable metadata at the absorbed-library boundary or annotate the "
        "absorbed function directly."
    )


def _callable_processing_contract(
    function: Callable[..., object],
) -> ProcessingContract | None:
    raw_value = vars(unwrap(function)).get(
        FunctionContractAttribute.processing_contract
    )
    if isinstance(raw_value, ProcessingContract):
        return raw_value
    value = vars(function).get(FunctionContractAttribute.processing_contract)
    if isinstance(value, ProcessingContract):
        return value
    return None
