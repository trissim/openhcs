"""Canonical FunctionStep Python document contracts."""

from __future__ import annotations

import pytest

from openhcs.core.function_step_document import (
    FunctionStepDocument,
    FunctionStepDocumentAuthority,
    FunctionStepDocumentField,
)
from openhcs.core.function_step_transport import FunctionStepTransportAuthority
from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.backends import cellprofiler as cellprofiler_backend
from openhcs.processing.backends.lib_registry.registry_service import RegistryService
from openhcs.pyqt_gui.services.reactor_providers import OpenHCSCodegenProvider


def _step() -> FunctionStep:
    return FunctionStep(func=cellprofiler_backend.crop, name="Crop")


def test_function_step_document_field_owns_exact_public_name() -> None:
    assert tuple(field.value for field in FunctionStepDocumentField) == ("step",)


def test_function_step_document_round_trip() -> None:
    document = FunctionStepDocumentAuthority.from_value(_step())

    source = FunctionStepDocumentAuthority.render(document)
    restored = FunctionStepDocumentAuthority.from_source(source)

    assert "step = FunctionStep(" in source
    assert restored.step.func is RegistryService.registered_callable(
        cellprofiler_backend.crop
    )
    assert FunctionStepDocumentAuthority.render(restored) == source


def test_from_value_delegates_transport_normalization(monkeypatch) -> None:
    step = _step()
    calls: list[FunctionStep] = []

    def normalize_step(value):
        calls.append(value)
        return value

    monkeypatch.setattr(
        FunctionStepTransportAuthority,
        "normalize_step",
        normalize_step,
    )

    document = FunctionStepDocumentAuthority.from_value(step)

    assert calls == [step]
    assert document.step is step


@pytest.mark.parametrize(
    ("namespace", "error_type", "message"),
    (
        ({}, ValueError, "step"),
        ({"step": object()}, TypeError, "FunctionStep"),
        ({"function_step": _step()}, ValueError, "step"),
    ),
)
def test_from_namespace_requires_exact_field_and_type(
    namespace: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        FunctionStepDocumentAuthority.from_namespace(namespace)


def test_render_revalidates_direct_document_instances() -> None:
    invalid = FunctionStepDocument(step=object())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="FunctionStep"):
        FunctionStepDocumentAuthority.render(invalid)


def test_openhcs_codegen_provider_delegates_step_documents() -> None:
    step = _step()

    assert OpenHCSCodegenProvider().generate_step_code(step) == (
        FunctionStepDocumentAuthority.render(
            FunctionStepDocumentAuthority.from_value(step)
        )
    )
