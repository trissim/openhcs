"""Tests for declaration-owned code-document normalization."""

from types import FunctionType

import pytest

from openhcs.core.config import GlobalPipelineConfig
from openhcs.core.pipeline_document import PipelineDocument
from openhcs.core.steps.function_step import FunctionStep
from openhcs.pyqt_gui.services.code_document_normalization import (
    CodeDocumentNormalizationStrategy,
)
from openhcs.ui.shared.plate_manager_code_document import (
    PlateManagerOrchestratorCodePayload,
)


def test_normalization_dispatch_descends_from_nominal_declaration_types() -> None:
    declaration_types = (
        PipelineDocument,
        FunctionStep,
        GlobalPipelineConfig,
        FunctionType,
        PlateManagerOrchestratorCodePayload,
    )
    strategies = tuple(
        CodeDocumentNormalizationStrategy.for_declaration_type(declaration_type)
        for declaration_type in declaration_types
    )

    assert len({type(strategy) for strategy in strategies}) == len(declaration_types)


def test_normalization_strategy_is_abstract() -> None:
    with pytest.raises(TypeError):
        CodeDocumentNormalizationStrategy()


def test_custom_function_normalization_validates_source() -> None:
    strategy = CodeDocumentNormalizationStrategy.for_declaration_type(FunctionType)

    with pytest.raises(SyntaxError):
        strategy.normalize_source(
            "def incomplete(",
            declaration_type=FunctionType,
            clean_mode=True,
        )
