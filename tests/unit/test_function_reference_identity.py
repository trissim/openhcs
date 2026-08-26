"""Proof boundaries for transported registry callable identity."""

from __future__ import annotations

import pytest

from openhcs.core.callable_contract import (
    CallableContract,
    CallableImportIdentity,
    CallableMetadata,
)
from openhcs.core.function_reference import (
    FunctionReferenceTransportAuthority,
    RegistryFunctionReference,
)
from openhcs.processing.backends import cellprofiler as cellprofiler_backend
from openhcs.processing.backends.lib_registry.registry_service import RegistryService
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


def test_registry_reference_derives_owner_from_canonical_key() -> None:
    with pytest.raises(ValueError, match="<registry>:<key>"):
        RegistryFunctionReference(
            import_identity=CallableImportIdentity(
                module_name="example.filters",
                function_name="filter_image",
            ),
            composite_key="filter_image",
        )


def test_cached_resolution_rejects_forged_import_identity(monkeypatch) -> None:
    composite_key, metadata = RegistryService.declared_metadata_for_callable(
        cellprofiler_backend.crop
    )
    monkeypatch.setattr(RegistryService, "_metadata_cache", {composite_key: metadata})
    monkeypatch.setattr(RegistryService, "_resolved_reference_callables", {})
    canonical_reference = FunctionReferenceTransportAuthority.function_reference(
        cellprofiler_backend.crop
    )
    assert canonical_reference.resolve() is metadata.func
    reference = RegistryFunctionReference(
        import_identity=CallableImportIdentity(
            module_name="forged.module",
            function_name="forged_name",
        ),
        composite_key=composite_key,
        metadata=FunctionReferenceTransportAuthority.callable_metadata(metadata.func),
    )

    with pytest.raises(RuntimeError, match="contradicts canonical identity"):
        reference.resolve()


def test_cold_resolution_rejects_forged_composite_key(monkeypatch) -> None:
    metadata = RegistryService.declared_metadata_for_callable(
        cellprofiler_backend.crop
    )[1]
    monkeypatch.setattr(RegistryService, "_metadata_cache", None)
    monkeypatch.setattr(RegistryService, "_resolved_reference_callables", {})
    reference = RegistryFunctionReference(
        import_identity=metadata.import_identity,
        composite_key="skimage:cellprofiler_crop",
        metadata=FunctionReferenceTransportAuthority.callable_metadata(metadata.func),
    )

    with pytest.raises(RuntimeError, match="contradicts declaration-owned identity"):
        reference.resolve()


def test_cold_external_resolution_retains_registry_classified_contract(
    monkeypatch,
) -> None:
    reference = RegistryFunctionReference(
        import_identity=CallableImportIdentity(
            module_name="skimage.filters.thresholding",
            function_name="threshold_otsu",
        ),
        composite_key="skimage:filters.threshold_otsu",
        metadata=CallableMetadata(
            processing_contract=ProcessingContract.FLEXIBLE,
        ),
    )
    monkeypatch.setattr(RegistryService, "_metadata_cache", None)
    monkeypatch.setattr(RegistryService, "_resolved_reference_callables", {})

    resolved = reference.resolve()

    assert (
        CallableContract.from_callable(resolved).processing_contract
        is ProcessingContract.FLEXIBLE
    )
