import pytest

from benchmark.converter.processing_contract_resolution import (
    ProcessingContractResolutionSource,
    resolve_processing_contract,
)
from openhcs.processing.backends.lib_registry.unified_registry import (
    ProcessingContract,
)


def test_resolve_processing_contract_uses_registry_contract() -> None:
    resolved = resolve_processing_contract(
        "MeasureObjectIntensity",
        "measure_object_intensity",
        "pure_2d",
    )

    assert resolved.contract is ProcessingContract.PURE_2D
    assert resolved.source is ProcessingContractResolutionSource.REGISTRY


def test_resolve_processing_contract_uses_callable_metadata() -> None:
    resolved = resolve_processing_contract("Opening", "opening", "unknown")

    assert resolved.contract is ProcessingContract.PURE_2D
    assert resolved.source is ProcessingContractResolutionSource.CALLABLE_METADATA


def test_resolve_processing_contract_rejects_unresolved_unknown() -> None:
    with pytest.raises(ValueError, match="declares unknown processing contract"):
        resolve_processing_contract("ColorToGray", "color_to_gray", "unknown")
