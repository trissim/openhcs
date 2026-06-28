"""Shared CellProfiler runtime binding authorities."""

from __future__ import annotations

from dataclasses import dataclass

from openhcs.interop.cellprofiler.runtime.bound_parameters import (
    RuntimeBoundParameterName,
)
from openhcs.interop.cellprofiler.runtime.payload_types import CellProfilerKwargs


class CellProfilerInvocationOverrideKwarg:
    """Runtime-internal kwargs that override CellProfiler invocation construction."""

    image = RuntimeBoundParameterName("_cellprofiler_image_override")
    execution_mode = RuntimeBoundParameterName("_cellprofiler_execution_mode_override")
    measurement_target_scope = RuntimeBoundParameterName(
        "_cellprofiler_measurement_target_scope"
    )


@dataclass(frozen=True, slots=True)
class CellProfilerOptionalNonemptyString:
    """Optional string text after CellProfiler kwarg type validation."""

    value: str

    def normalized_or_none(self) -> str | None:
        normalized = self.value.strip()
        if not normalized:
            return None
        return normalized


class CellProfilerStringKwargAuthority:
    """Typed string-kwarg validation shared by CellProfiler binding policies."""

    @staticmethod
    def required(
        kwargs: CellProfilerKwargs,
        name: str,
        module_name: str,
    ) -> str:
        value = kwargs.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{module_name} requires non-empty kwarg {name!r}.")
        return value

    @staticmethod
    def optional(
        kwargs: CellProfilerKwargs,
        name: str,
    ) -> str | None:
        raw_value = kwargs.get(name)
        if raw_value is None:
            return None
        if not isinstance(raw_value, str):
            raise TypeError(
                f"Expected string kwarg {name!r}, got {type(raw_value).__name__}."
            )
        return CellProfilerOptionalNonemptyString(raw_value).normalized_or_none()
