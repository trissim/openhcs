"""Canonical Python document contract for one OpenHCS FunctionStep."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from openhcs.core.function_step_transport import FunctionStepTransportAuthority
from openhcs.core.steps.function_step import FunctionStep


class FunctionStepDocumentField(str, Enum):
    """Exact public assignment in a FunctionStep Python document."""

    STEP = "step"


@dataclass(frozen=True, slots=True)
class FunctionStepDocument:
    """Normalized public declaration for one FunctionStep."""

    step: FunctionStep


class FunctionStepDocumentAuthority:
    """Validate, parse, and render the canonical FunctionStep document shape."""

    HEADER = "# Function Step"

    @classmethod
    def from_value(cls, step: FunctionStep) -> FunctionStepDocument:
        """Build a document from one typed and transport-normalized step."""

        cls._require_step(step)
        normalized_step = FunctionStepTransportAuthority.normalize_step(step)
        cls._require_step(normalized_step)
        return FunctionStepDocument(step=normalized_step)

    @classmethod
    def from_namespace(
        cls,
        namespace: Mapping[str, object],
    ) -> FunctionStepDocument:
        """Read the exact public step assignment from an executed namespace."""

        field_name = FunctionStepDocumentField.STEP.value
        if field_name not in namespace:
            raise ValueError(f"FunctionStep document must define {field_name!r}.")
        return cls.from_value(namespace[field_name])

    @classmethod
    def from_source(cls, source: str) -> FunctionStepDocument:
        """Execute Python source and read its canonical FunctionStep assignment."""

        namespace: dict[str, object] = {}
        code = compile(source, "<openhcs-function-step-document>", "exec")
        exec(code, namespace)
        return cls.from_namespace(namespace)

    @classmethod
    def render(
        cls,
        document: FunctionStepDocument,
        *,
        clean_mode: bool = True,
    ) -> str:
        """Render a validated FunctionStep document as reviewable Python source."""

        import openhcs.serialization.pycodify_formatters  # noqa: F401
        from pycodify import Assignment
        from openhcs.serialization.source_path_factoring import (
            OpenHCSPythonSourceDocument,
        )

        normalized = cls.from_value(document.step)
        return OpenHCSPythonSourceDocument(
            Assignment(FunctionStepDocumentField.STEP.value, normalized.step),
            header=cls.HEADER,
            clean_mode=clean_mode,
        ).render()

    @staticmethod
    def _require_step(value: object) -> None:
        if not isinstance(value, FunctionStep):
            raise TypeError(f"step must be a FunctionStep, got {type(value).__name__}.")
