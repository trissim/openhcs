"""Exact reconciliation tests for generated function import projections."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import openhcs.processing.func_registry as func_registry
from openhcs.core.memory import numpy
from openhcs.processing.backends.lib_registry.registry_service import RegistryService


class _ExternalProjectionOwner:
    def public_projection_module(self, metadata):
        return metadata.public_module


def test_external_projection_removes_stale_exports_and_modules(monkeypatch) -> None:
    module_name = "openhcs.codex_external.filters"

    def transient_filter(image):
        return image

    metadata = SimpleNamespace(
        func=transient_filter,
        registry=_ExternalProjectionOwner(),
        public_module=module_name,
    )
    monkeypatch.setattr(func_registry, "_external_projection_exports", {})
    monkeypatch.setattr(func_registry, "_external_projection_modules", set())

    func_registry._create_external_virtual_modules({"external:filter": metadata})
    assert sys.modules[module_name].transient_filter is transient_filter

    func_registry._create_external_virtual_modules({})
    assert module_name not in sys.modules
    assert "openhcs.codex_external" not in sys.modules


def test_legacy_name_lookup_fails_with_canonical_candidates(monkeypatch) -> None:
    @numpy
    def first_crop(image):
        return image

    @numpy
    def second_crop(image):
        return image

    metadata = {
        "openhcs:numpy_crop": SimpleNamespace(
            func=first_crop,
            display_name="crop",
        ),
        "openhcs:cellprofiler_crop": SimpleNamespace(
            func=second_crop,
            display_name="crop",
        ),
    }
    monkeypatch.setattr(
        RegistryService,
        "get_all_functions_with_metadata",
        classmethod(lambda cls: metadata),
    )

    with pytest.raises(LookupError, match="canonical function IDs"):
        func_registry.get_function_by_name("crop", "numpy")
    assert func_registry.get_function("openhcs:numpy_crop") is first_crop
