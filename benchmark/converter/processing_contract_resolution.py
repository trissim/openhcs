"""Typed CellProfiler processing-contract resolution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from benchmark.cellprofiler_library import require_function
from openhcs.processing.backends.lib_registry.unified_registry import (
    ProcessingContract,
)


class CellProfilerProcessingContractName(str, Enum):
    """Processing-contract names accepted from the absorbed registry."""

    PURE_2D = "pure_2d"
    PURE_3D = "pure_3d"
    FLEXIBLE = "flexible"
    VOLUMETRIC_TO_SLICE = "volumetric_to_slice"
    UNKNOWN = "unknown"

    @classmethod
    def from_registry_value(
        cls,
        value: str,
        *,
        module_name: str,
    ) -> "CellProfilerProcessingContractName":
        try:
            return cls(value.strip().lower())
        except ValueError as error:
            raise ValueError(
                f"Module {module_name} declares unsupported processing contract "
                f"{value!r}."
            ) from error

    def to_openhcs_contract(self) -> ProcessingContract:
        if self is CellProfilerProcessingContractName.UNKNOWN:
            raise ValueError("Unknown CellProfiler contracts are not executable.")
        return ProcessingContract[self.name]


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
) -> ResolvedProcessingContract:
    """Resolve one absorbed module to an executable OpenHCS contract."""
    registry_contract = CellProfilerProcessingContractName.from_registry_value(
        declared_contract,
        module_name=module_name,
    )
    if registry_contract is not CellProfilerProcessingContractName.UNKNOWN:
        return ResolvedProcessingContract(
            contract=registry_contract.to_openhcs_contract(),
            source=ProcessingContractResolutionSource.REGISTRY,
        )

    callable_contract = _callable_processing_contract(
        require_function(module_name, function_name=function_name)
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
    function: Callable[..., Any],
) -> ProcessingContract | None:
    value = vars(function).get("__processing_contract__")
    if isinstance(value, ProcessingContract):
        return value
    return None
