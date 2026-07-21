from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from openhcs.agent.services.function_catalog_service import FunctionCatalogService
from openhcs.processing.backends.cellprofiler.feature_enhancement import (
    enhance_or_suppress_features,
)
from openhcs.processing.backends.cellprofiler.primary_objects import (
    identify_primary_objects,
)


@dataclass(frozen=True)
class _Metadata:
    func: Callable
    tags: tuple[str, ...]
    library: str = "openhcs"

    @property
    def original_name(self) -> str:
        return self.func.__name__

    @property
    def name(self) -> str:
        return self.func.__name__

    @property
    def display_name(self) -> str:
        return self.func.__name__

    @property
    def module(self) -> str:
        return self.func.__module__

    @property
    def doc(self) -> str:
        return self.func.__doc__ or ""

    def get_registry_name(self) -> str:
        return self.library


def _catalog(monkeypatch, function_id: str, func: Callable) -> FunctionCatalogService:
    monkeypatch.setattr(
        FunctionCatalogService,
        "_all_metadata",
        lambda self: {
            function_id: _Metadata(func=func, tags=("openhcs", "cellprofiler"))
        },
    )
    return FunctionCatalogService()


def test_library_selector_accepts_declaration_owned_backend_tag(monkeypatch):
    catalog = _catalog(
        monkeypatch,
        "openhcs:cellprofiler_identify_primary_objects",
        identify_primary_objects,
    )

    by_registry = catalog.search(library="openhcs")
    by_backend_tag = catalog.search(library="cellprofiler")

    assert by_registry.total == 1
    assert by_backend_tag.total == 1
    assert by_backend_tag.items[0].backend_tags == ("openhcs", "cellprofiler")


def test_function_parameters_project_enum_import_members_and_values(monkeypatch):
    catalog = _catalog(
        monkeypatch,
        "openhcs:cellprofiler_enhance_or_suppress_features",
        enhance_or_suppress_features,
    )

    detail = catalog.get("openhcs:cellprofiler_enhance_or_suppress_features")
    parameters = {parameter.name: parameter for parameter in detail.parameters}
    neurite_method = parameters["neurite_method"]

    assert neurite_method.enum_import_path == (
        "openhcs.processing.backends.cellprofiler.feature_enhancement.NeuriteMethod"
    )
    assert neurite_method.enum_members == ("GRADIENT", "TUBENESS")
    assert neurite_method.enum_values == ("Line structures", "Tubeness")


def test_cellprofiler_detail_distinguishes_static_and_compiled_artifacts(monkeypatch):
    catalog = _catalog(
        monkeypatch,
        "openhcs:cellprofiler_identify_primary_objects",
        identify_primary_objects,
    )

    detail = catalog.get("openhcs:cellprofiler_identify_primary_objects")

    assert detail.runtime_contract is not None
    runtime_contract = detail.runtime_contract
    assert runtime_contract.artifact_inputs == ()
    assert runtime_contract.artifact_outputs == ()
    assert runtime_contract.cellprofiler_module is not None
    module = runtime_contract.cellprofiler_module
    assert module.exact_artifact_contract_requires_compilation is True
    assert {
        (binding.direction, binding.kind)
        for binding in module.artifact_bindings
    } >= {
        ("input", "image"),
        ("output", "object_labels"),
    }
    assert runtime_contract.source_binding_rule is not None
    assert "Callable-level artifact arrays can therefore be empty" in (
        runtime_contract.source_binding_rule
    )
    assert runtime_contract.pattern_compatibility_rule is not None
    assert "may intentionally cover only a subset" in (
        runtime_contract.pattern_compatibility_rule
    )
