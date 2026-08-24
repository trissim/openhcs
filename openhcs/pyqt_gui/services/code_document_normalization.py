"""OpenHCS normalization strategies for editable code documents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import FunctionType
from typing import TypeVar

from metaclass_registry import AutoRegisterMeta
from pyqt_reactive.services.function_pattern_code_document import FunctionPatternValue
from python_introspect import AnnotatedDataclassValidationMixin

from openhcs.core.config_document import ConfigDocumentAuthority
from openhcs.core.function_step_document import (
    FunctionStepDocument,
    FunctionStepDocumentAuthority,
)
from openhcs.core.pipeline_document import PipelineDocument, PipelineDocumentAuthority
from openhcs.core.registry_strategies import NominalTypeStrategyFamilyMixin
from openhcs.core.steps.function_step import FunctionStep
from openhcs.ui.shared.plate_manager_code_document import (
    PlateManagerCodeDocumentAuthority,
    PlateManagerOrchestratorCodePayload,
)

DeclarationT = TypeVar("DeclarationT")


class CodeDocumentNormalizationStrategy(
    NominalTypeStrategyFamilyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Normalize source through the authority owned by its declaration type."""

    @classmethod
    def for_declaration_type(
        cls,
        declaration_type: type[DeclarationT],
    ) -> CodeDocumentNormalizationStrategy:
        strategy_types = cls.strategy_types_for_nominal_type(declaration_type)
        if not strategy_types:
            raise TypeError(
                "Code normalization has no registered strategy for declaration type "
                f"{declaration_type.__module__}.{declaration_type.__qualname__}."
            )
        return strategy_types[0]()

    @abstractmethod
    def normalize_source(
        self,
        source: str,
        *,
        declaration_type: type[DeclarationT],
        clean_mode: bool,
    ) -> str:
        """Render source through the declaration authority being edited."""
        raise NotImplementedError


class PipelineDocumentNormalizationStrategy(CodeDocumentNormalizationStrategy):
    value_type = PipelineDocument

    def normalize_source(
        self,
        source: str,
        *,
        declaration_type: type[DeclarationT],
        clean_mode: bool,
    ) -> str:
        del declaration_type
        return PipelineDocumentAuthority.render(
            PipelineDocumentAuthority.from_source(source),
            clean_mode=clean_mode,
        )


class StepDocumentNormalizationStrategy(CodeDocumentNormalizationStrategy):
    value_type = (FunctionStep, FunctionStepDocument)

    def normalize_source(
        self,
        source: str,
        *,
        declaration_type: type[DeclarationT],
        clean_mode: bool,
    ) -> str:
        del declaration_type
        return FunctionStepDocumentAuthority.render(
            FunctionStepDocumentAuthority.from_source(source),
            clean_mode=clean_mode,
        )


class ConfigDocumentNormalizationStrategy(CodeDocumentNormalizationStrategy):
    value_type = AnnotatedDataclassValidationMixin

    def normalize_source(
        self,
        source: str,
        *,
        declaration_type: type[DeclarationT],
        clean_mode: bool,
    ) -> str:
        config = ConfigDocumentAuthority.from_source(
            source,
            expected_config_type=declaration_type,
        )
        return ConfigDocumentAuthority.render(
            config,
            expected_config_type=declaration_type,
            clean_mode=clean_mode,
        )


class CustomFunctionDocumentNormalizationStrategy(CodeDocumentNormalizationStrategy):
    value_type = FunctionType

    def normalize_source(
        self,
        source: str,
        *,
        declaration_type: type[DeclarationT],
        clean_mode: bool,
    ) -> str:
        del declaration_type, clean_mode
        compile(source, "<openhcs-custom-function-document>", "exec")
        return source


class FunctionPatternDocumentNormalizationStrategy(
    CustomFunctionDocumentNormalizationStrategy
):
    value_type = FunctionPatternValue

    def normalize_source(
        self,
        source: str,
        *,
        declaration_type: type[DeclarationT],
        clean_mode: bool,
    ) -> str:
        del declaration_type
        from pyqt_reactive.services.function_pattern_code_document import (
            FunctionPatternCodeDocumentService,
        )

        service = FunctionPatternCodeDocumentService()
        return service.generate_complete_function_pattern_code(
            service.pattern_from_source(source),
            clean_mode=clean_mode,
        )


class PlateManagerDocumentNormalizationStrategy(CodeDocumentNormalizationStrategy):
    value_type = PlateManagerOrchestratorCodePayload

    def normalize_source(
        self,
        source: str,
        *,
        declaration_type: type[DeclarationT],
        clean_mode: bool,
    ) -> str:
        del declaration_type
        return PlateManagerCodeDocumentAuthority.render(
            PlateManagerCodeDocumentAuthority.from_source(source),
            clean_mode=clean_mode,
        )
