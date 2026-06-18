"""Shared Python source rendering authority for OpenHCS agent services."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
    from openhcs.core.steps.function_step import FunctionStep


ConfigSourceValue: TypeAlias = "GlobalPipelineConfig | PipelineConfig"
PipelineSourceValue: TypeAlias = Sequence["FunctionStep"]
PythonSourceValue: TypeAlias = "ConfigSourceValue | PipelineSourceValue"


class PythonSourceAssignmentKind(Enum):
    CONFIG = ("config", "# OpenHCS configuration")
    PIPELINE_STEPS = (
        "pipeline_steps",
        "# Edit this pipeline and save to apply changes",
    )

    @property
    def assignment_name(self) -> str:
        return self.value[0]

    @property
    def header(self) -> str:
        return self.value[1]

    def assignment(
        self,
        value: PythonSourceValue,
        clean_mode: bool,
    ) -> "PythonSourceAssignment":
        return PythonSourceAssignment(self, value, clean_mode)


@dataclass(frozen=True, slots=True)
class PythonSourceAssignment:
    """Nominal pycodify request for reviewable OpenHCS Python source."""

    kind: PythonSourceAssignmentKind
    value: PythonSourceValue
    clean_mode: bool

    def render(self) -> str:
        import openhcs.serialization.pycodify_formatters  # noqa: F401
        from pycodify import Assignment, generate_python_source

        return generate_python_source(
            Assignment(self.kind.assignment_name, self.value),
            self.kind.header,
            self.clean_mode,
        )
