"""PyQt adapter tests for FunctionStep code documents."""

from __future__ import annotations

import pytest

from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.backends import cellprofiler as cellprofiler_backend
from openhcs.processing.backends.lib_registry.registry_service import RegistryService
from openhcs.pyqt_gui.services.function_step_code_document import (
    FunctionStepCodeDocumentDriver,
)


def _step(name: str = "Crop") -> FunctionStep:
    return FunctionStep(func=cellprofiler_backend.crop, name=name)


def test_function_step_driver_round_trip_uses_semantic_document() -> None:
    current = _step()
    applied: list[FunctionStep] = []
    reads: list[bool] = []
    driver = FunctionStepCodeDocumentDriver(
        title="Edit Step: Crop",
        current_step=lambda: current,
        apply_step=applied.append,
        before_read=lambda: reads.append(True),
    )

    document = driver.read_document()
    driver.validate_source(document.source)
    driver.apply_source(document.source)

    assert document.title == "Edit Step: Crop"
    assert reads == [True]
    assert len(applied) == 1
    assert applied[0].func is RegistryService.registered_callable(
        cellprofiler_backend.crop
    )


def test_function_step_driver_rejects_nonsemantic_source() -> None:
    driver = FunctionStepCodeDocumentDriver(
        title="Edit Step",
        current_step=_step,
        apply_step=lambda _step: None,
    )

    with pytest.raises(ValueError, match="step"):
        driver.validate_source("function_step = object()")


def test_function_step_driver_authorizes_before_apply() -> None:
    current = _step()
    applied: list[FunctionStep] = []
    driver = FunctionStepCodeDocumentDriver(
        title="Edit Step",
        current_step=lambda: current,
        apply_step=applied.append,
        before_apply=lambda: (_ for _ in ()).throw(RuntimeError("mutation rejected")),
    )
    source = driver.read_document().source

    with pytest.raises(RuntimeError, match="mutation rejected"):
        driver.apply_source(source)

    assert applied == []
