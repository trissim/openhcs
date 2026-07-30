from __future__ import annotations

from functools import wraps
import importlib
import inspect
from types import SimpleNamespace

from openhcs.core.function_reference import FunctionReferenceTransportAuthority
from openhcs.core.function_step_transport import FunctionStepTransportAuthority
from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.backends import cellprofiler as cellprofiler_backend
from openhcs.processing.backends.lib_registry.registry_service import RegistryService
from python_introspect import Enableable, is_enableable, mark_enableable


def _registered_callable_pair():
    def declared(image):
        return image

    @wraps(declared)
    def registered(image, *, enabled: bool = True):
        del enabled
        return declared(image)

    registered.__signature__ = inspect.signature(declared).replace(
        parameters=(
            *inspect.signature(declared).parameters.values(),
            Enableable.parameter(),
        )
    )
    mark_enableable(registered)
    return declared, registered


def test_function_step_transport_projects_registered_callable_authority(
    monkeypatch,
) -> None:
    """Imported declarations normalize to their registered callable authority."""
    declared, registered = _registered_callable_pair()
    metadata = SimpleNamespace(
        func=registered,
        registry=SimpleNamespace(library_name="test", MEMORY_TYPE="numpy"),
    )
    monkeypatch.setattr(
        RegistryService,
        "get_all_functions_with_metadata",
        classmethod(lambda cls: {"test:declared": metadata}),
    )

    normalized_step = FunctionStepTransportAuthority.normalize_step(
        FunctionStep(func=declared)
    )

    assert normalized_step.func is registered
    assert is_enableable(normalized_step.func)


def test_registered_exports_preserve_enableable_identity_through_transport(
    monkeypatch,
) -> None:
    """Every registry owner survives reference and FunctionStep transport."""
    monkeypatch.setenv("OPENHCS_CPU_ONLY", "true")
    RegistryService.clear_metadata_cache()
    try:
        metadata_by_key = RegistryService.get_all_functions_with_metadata()
        assert metadata_by_key

        for composite_key, metadata in metadata_by_key.items():
            declared = vars(importlib.import_module(metadata.module))[
                metadata.original_name
            ]

            assert is_enableable(metadata.func)
            assert (declared is metadata.func) is is_enableable(declared)

            reference = FunctionReferenceTransportAuthority.function_reference(declared)
            assert reference.composite_key == composite_key
            assert reference.resolve() is metadata.func

            (normalized_step,) = FunctionStepTransportAuthority.normalize_pipeline(
                [FunctionStep(func=declared)]
            )
            assert normalized_step.func is metadata.func
    finally:
        RegistryService.clear_metadata_cache()


def test_local_projection_and_catalog_warmup_share_callable_identity(
    monkeypatch,
) -> None:
    """Cold source parsing and later catalog warmup reuse one wrapper."""

    monkeypatch.setenv("OPENHCS_CPU_ONLY", "true")
    RegistryService.clear_metadata_cache()
    try:
        local = RegistryService.registered_callable(cellprofiler_backend.crop)
        assert is_enableable(local)

        RegistryService.get_all_functions_with_metadata()
        warmed = RegistryService.registered_callable(cellprofiler_backend.crop)

        assert warmed is local
    finally:
        RegistryService.clear_metadata_cache()
