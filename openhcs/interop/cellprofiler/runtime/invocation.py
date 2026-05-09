"""Typed CellProfiler invocation records and execution-mode utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from collections.abc import Mapping

import numpy as np

from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.runtime_semantics import MeasurementImageReferenceDomain
from openhcs.core.runtime_invocation import (
    RuntimeImageExecutionContext,
    RuntimeInvocationOptions,
    RuntimeSliceAlignedValues,
    requested_image_execution_mode,
)


CELLPROFILER_GRID_CYCLE_SCOPE_KWARG = "_cellprofiler_grid_cycle_scope"


class CellProfilerGridCycleScope(str, Enum):
    """Closed DefineGrid execution scopes from CellProfiler."""

    EACH_CYCLE = "each_cycle"
    ONCE = "once"


@dataclass(frozen=True, slots=True, kw_only=True)
class CellProfilerInvocationOptions(RuntimeInvocationOptions):
    """Typed CellProfiler controls that are not absorbed function arguments."""

    grid_cycle_scope: CellProfilerGridCycleScope = CellProfilerGridCycleScope.EACH_CYCLE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "grid_cycle_scope",
            coerce_cellprofiler_grid_cycle_scope(self.grid_cycle_scope),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CellProfilerImageExecutionContext(RuntimeImageExecutionContext):
    """Shared source provenance for CellProfiler image execution records."""


@dataclass(frozen=True, slots=True, kw_only=True)
class CellProfilerResolvedInputRequest(CellProfilerImageExecutionContext):
    """Shared source provenance for resolved CellProfiler invocation inputs."""

    image_count: int


@dataclass(frozen=True, slots=True, kw_only=True)
class CellProfilerImageRequest(CellProfilerResolvedInputRequest):
    """Resolved image payload and source metadata for one module invocation."""

    payload: object


@dataclass(frozen=True, slots=True, kw_only=True)
class CellProfilerInvocationRequest(CellProfilerResolvedInputRequest):
    """Resolved invocation inputs for one CellProfiler function call."""

    image: object
    kwargs: Mapping[str, object]


CellProfilerMeasurementImageDomain = MeasurementImageReferenceDomain


@dataclass(frozen=True, slots=True, kw_only=True)
class CellProfilerMeasurementImage(CellProfilerImageExecutionContext):
    """One resolved image payload used by object measurement modules."""

    payload: object
    source_image_names: tuple[str, ...] = ()
    align_to_labels: bool = True
    reference_domain: CellProfilerMeasurementImageDomain = (
        CellProfilerMeasurementImageDomain.SOURCE_IMAGE
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_image_names",
            tuple(str(name) for name in self.source_image_names),
        )
        if not isinstance(self.reference_domain, CellProfilerMeasurementImageDomain):
            raise TypeError(
                "CellProfilerMeasurementImage.reference_domain must be "
                "CellProfilerMeasurementImageDomain, got "
                f"{type(self.reference_domain).__name__}."
            )


@dataclass(frozen=True, slots=True)
class CellProfilerSliceAlignedValues(RuntimeSliceAlignedValues[np.ndarray]):
    """Non-image vector payload with one value array per object-label slice."""

    def __post_init__(self) -> None:
        slices = tuple(np.asarray(value) for value in self.slices)
        if not slices:
            raise ValueError("CellProfilerSliceAlignedValues.slices cannot be empty.")
        object.__setattr__(self, "slices", slices)


def illumination_scope_uses_all_images(value: object) -> bool:
    """Return whether a CellProfiler illumination scope means all images."""
    if value is None:
        return False
    if isinstance(value, Enum):
        value = value.value
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return normalized.startswith("all")


def coerce_cellprofiler_grid_cycle_scope(
    value: object,
    *,
    default: CellProfilerGridCycleScope = CellProfilerGridCycleScope.EACH_CYCLE,
) -> CellProfilerGridCycleScope:
    """Coerce CellProfiler's grid scope setting into a closed runtime enum."""
    if value is None:
        return default
    if isinstance(value, CellProfilerGridCycleScope):
        return value
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    if normalized in {"each", "each_cycle"}:
        return CellProfilerGridCycleScope.EACH_CYCLE
    if normalized == "once":
        return CellProfilerGridCycleScope.ONCE
    return CellProfilerGridCycleScope(normalized)
