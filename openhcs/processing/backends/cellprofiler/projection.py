"""CellProfiler-compatible image projection backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta
import numpy as np

from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.processing.materialization import csv_materializer

PROJECTION_STATS_FIELDS = [
    "projection_type",
    "input_slices",
    "output_min",
    "output_max",
    "output_mean",
]


class ProjectionType(Enum):
    AVERAGE = "average"
    MAXIMUM = "maximum"
    MINIMUM = "minimum"
    SUM = "sum"
    VARIANCE = "variance"
    POWER = "power"
    BRIGHTFIELD = "brightfield"
    MASK = "mask"


@dataclass(frozen=True, slots=True)
class ProjectionStats:
    projection_type: str
    input_slices: int
    output_min: float
    output_max: float
    output_mean: float


@dataclass(frozen=True, slots=True)
class ProjectionRequest:
    """Normalized MakeProjection request."""

    image: np.ndarray
    projection_type: ProjectionType
    frequency: float

    @property
    def stack(self) -> np.ndarray:
        return self.image[np.newaxis, :, :] if self.image.ndim == 2 else self.image

    def stats(self, result: np.ndarray) -> ProjectionStats:
        return ProjectionStats(
            projection_type=self.projection_type.value,
            input_slices=self.stack.shape[0],
            output_min=float(np.min(result)),
            output_max=float(np.max(result)),
            output_mean=float(np.mean(result)),
        )


class ProjectionStrategy(
    EnumKeyedStrategyMixin[ProjectionType],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Apply one CellProfiler MakeProjection operation."""

    __registry_key__ = "projection_type_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "projection_type"
    __enum_label_attr__ = "projection_type_label"
    projection_type: ClassVar[ProjectionType | None] = None
    projection_type_label: ClassVar[str | None] = None

    @classmethod
    def for_projection_type(cls, projection_type: ProjectionType) -> "ProjectionStrategy":
        return cls.for_enum_member(projection_type)

    @abstractmethod
    def apply(self, request: ProjectionRequest) -> np.ndarray:
        """Return the projected image."""


class Float32ProjectionStrategy(ProjectionStrategy):
    """Template for projection algorithms that materialize float32 output."""

    def apply(self, request: ProjectionRequest) -> np.ndarray:
        return self.project(request).astype(np.float32)

    @abstractmethod
    def project(self, request: ProjectionRequest) -> np.ndarray:
        """Return the projection before final CellProfiler float32 materialization."""


class AverageProjectionStrategy(Float32ProjectionStrategy):
    projection_type = ProjectionType.AVERAGE

    def project(self, request: ProjectionRequest) -> np.ndarray:
        return np.mean(request.stack, axis=0)


class MaximumProjectionStrategy(Float32ProjectionStrategy):
    projection_type = ProjectionType.MAXIMUM

    def project(self, request: ProjectionRequest) -> np.ndarray:
        return np.max(request.stack, axis=0)


class MinimumProjectionStrategy(Float32ProjectionStrategy):
    projection_type = ProjectionType.MINIMUM

    def project(self, request: ProjectionRequest) -> np.ndarray:
        return np.min(request.stack, axis=0)


class SumProjectionStrategy(Float32ProjectionStrategy):
    projection_type = ProjectionType.SUM

    def project(self, request: ProjectionRequest) -> np.ndarray:
        return np.sum(request.stack, axis=0)


class VarianceProjectionStrategy(Float32ProjectionStrategy):
    projection_type = ProjectionType.VARIANCE

    def project(self, request: ProjectionRequest) -> np.ndarray:
        return np.var(request.stack.astype(np.float64), axis=0)


class PowerProjectionStrategy(Float32ProjectionStrategy):
    projection_type = ProjectionType.POWER

    def project(self, request: ProjectionRequest) -> np.ndarray:
        stack = request.stack.astype(np.float64)
        depth, height, width = stack.shape
        summed = np.sum(stack, axis=0)
        power_image = np.zeros((height, width), dtype=np.complex128)
        power_mask = np.zeros((height, width), dtype=np.complex128)
        for index in range(depth):
            multiplier = np.exp(2j * np.pi * float(index) / request.frequency)
            power_image += multiplier * stack[index]
            power_mask += multiplier
        power_image -= summed * power_mask / depth
        return (power_image * np.conj(power_image)).real


class BrightfieldProjectionStrategy(Float32ProjectionStrategy):
    projection_type = ProjectionType.BRIGHTFIELD

    def project(self, request: ProjectionRequest) -> np.ndarray:
        stack = request.stack.astype(np.float64)
        norm0 = np.mean(stack[0])
        bright_max = stack[0].copy()
        bright_min = stack[0].copy()
        for index in range(1, stack.shape[0]):
            norm = np.mean(stack[index])
            normalized = stack[index] * norm0 / norm if norm > 0 else stack[index]
            max_mask = bright_max < normalized
            min_mask = bright_min > normalized
            bright_min[min_mask] = normalized[min_mask]
            bright_max[max_mask] = normalized[max_mask]
            bright_min[max_mask] = bright_max[max_mask]
        return bright_max - bright_min


class MaskProjectionStrategy(Float32ProjectionStrategy):
    projection_type = ProjectionType.MASK

    def project(self, request: ProjectionRequest) -> np.ndarray:
        return np.all(request.stack > 0, axis=0)


@numpy
@special_outputs(
    (
        "projection_stats",
        csv_materializer(
            fields=PROJECTION_STATS_FIELDS,
            analysis_type="projection",
        ),
    )
)
def make_projection(
    image: np.ndarray,
    projection_type: ProjectionType = ProjectionType.AVERAGE,
    frequency: float = 6.0,
) -> tuple[np.ndarray, ProjectionStats]:
    """Combine a stack of 2-D images into a single 2-D projection image."""
    projection_type = coerce_cellprofiler_enum(ProjectionType, projection_type)
    request = ProjectionRequest(
        image=image,
        projection_type=projection_type,
        frequency=frequency,
    )
    result = ProjectionStrategy.for_projection_type(projection_type).apply(request)
    return result, request.stats(result)


__all__ = [
    "AverageProjectionStrategy",
    "BrightfieldProjectionStrategy",
    "MaskProjectionStrategy",
    "MaximumProjectionStrategy",
    "MinimumProjectionStrategy",
    "PROJECTION_STATS_FIELDS",
    "PowerProjectionStrategy",
    "ProjectionRequest",
    "ProjectionStats",
    "ProjectionStrategy",
    "ProjectionType",
    "SumProjectionStrategy",
    "VarianceProjectionStrategy",
    "make_projection",
]
