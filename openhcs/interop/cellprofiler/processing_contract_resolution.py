"""Resolve absorbed function contract declarations to OpenHCS contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from openhcs.processing.backends.cellprofiler import require_cellprofiler_function
from openhcs.processing.backends.lib_registry.unified_registry import (
    ProcessingContract,
)

UNKNOWN_PROCESSING_CONTRACT_NAME = "unknown"


class ProcessingContractResolutionSource(str, Enum):
    """Authority that resolved one executable processing contract."""

    REGISTRY = "registry"
    CALLABLE_METADATA = "callable_metadata"


@dataclass(frozen=True, slots=True)
class ResolvedProcessingContract:
    """Executable OpenHCS contract plus provenance."""

    contract: ProcessingContract
    source: ProcessingContractResolutionSource


def resolve_processing_contract(
    module_name: str,
    function_name: str,
    declared_contract: str,
    *,
    function_resolver: Callable[..., Callable[..., object]] = require_cellprofiler_function,
) -> ResolvedProcessingContract:
    """Resolve one absorbed module to an executable OpenHCS contract."""
    normalized_contract = declared_contract.strip().lower()
    if normalized_contract != UNKNOWN_PROCESSING_CONTRACT_NAME:
        registry_contract = ProcessingContract.from_declared_name(normalized_contract)
        if registry_contract is None:
            raise ValueError(
                f"Module {module_name} declares unsupported processing contract "
                f"{declared_contract!r}."
            )
        return ResolvedProcessingContract(
            contract=registry_contract,
            source=ProcessingContractResolutionSource.REGISTRY,
        )

    callable_contract = _callable_processing_contract(
        function_resolver(module_name, function_name=function_name)
    )
    if callable_contract is not None:
        return ResolvedProcessingContract(
            contract=callable_contract,
            source=ProcessingContractResolutionSource.CALLABLE_METADATA,
        )

    raise ValueError(
        f"Module {module_name} declares unknown processing contract and "
        f"{function_name} has no __processing_contract__ metadata. Add an "
        "explicit registry contract or annotate the absorbed function."
    )


def _callable_processing_contract(
    function: Callable[..., object],
) -> ProcessingContract | None:
    value = vars(function).get("__processing_contract__")
    if isinstance(value, ProcessingContract):
        return value
    return None
