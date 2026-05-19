"""Shared model types for cell-counting analysis backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, ClassVar

from metaclass_registry import AutoRegisterMeta

from openhcs.core.registry_strategies import EnumKeyedStrategyMixin


class DetectionMethod(Enum):
    """Cell detection methods available."""

    BLOB_LOG = "blob_log"
    BLOB_DOG = "blob_dog"
    BLOB_DOH = "blob_doh"
    WATERSHED = "watershed"
    THRESHOLD = "threshold"


class ColocalizationMethod(Enum):
    """Methods for multi-channel colocalization analysis."""

    OVERLAP_AREA = "overlap_area"
    DISTANCE_BASED = "distance_based"
    INTENSITY_CORRELATION = "intensity_correlation"
    MANDERS_COEFFICIENTS = "manders_coefficients"


class ThresholdMethod(Enum):
    """Automatic thresholding methods for watershed segmentation."""

    OTSU = "otsu"
    LI = "li"
    MANUAL = "manual"


@dataclass
class CellCountResult:
    """Results for single-channel cell counting."""

    slice_index: int
    method: str
    cell_count: int
    cell_positions: list[tuple[float, float]]
    cell_areas: list[float]
    cell_intensities: list[float]
    detection_confidence: list[float]
    parameters_used: dict[str, Any]
    binary_mask: Any | None = None

    @classmethod
    def from_measurements(
        cls,
        slice_index: int,
        method: str,
        positions: list[tuple[float, float]],
        areas: list[float],
        intensities: list[float],
        confidences: list[float],
        parameters: dict[str, Any],
        binary_mask: Any | None = None,
    ) -> "CellCountResult":
        """Build a count result from parallel detection measurement vectors."""
        return cls(
            slice_index=slice_index,
            method=method,
            cell_count=len(positions),
            cell_positions=positions,
            cell_areas=areas,
            cell_intensities=intensities,
            detection_confidence=confidences,
            parameters_used=parameters,
            binary_mask=binary_mask,
        )


@dataclass
class MultiChannelResult:
    """Results for multi-channel cell counting and colocalization."""

    slice_index: int
    chan_1_results: CellCountResult
    chan_2_results: CellCountResult
    colocalization_method: str
    colocalized_count: int
    colocalization_percentage: float
    chan_1_only_count: int
    chan_2_only_count: int
    colocalization_metrics: dict[str, Any]
    overlap_positions: list[tuple[float, float]]


@dataclass(frozen=True, slots=True)
class DistanceColocalizationMetrics:
    """Metrics reported by distance-based colocalization."""

    average_colocalization_distance: float
    max_colocalization_distance: float
    distance_threshold_used: float

    def as_dict(self) -> dict[str, float]:
        return {
            "average_colocalization_distance": self.average_colocalization_distance,
            "max_colocalization_distance": self.max_colocalization_distance,
            "distance_threshold_used": self.distance_threshold_used,
        }


@dataclass(frozen=True, slots=True)
class IntensityColocalizationMetrics:
    """Metrics reported by intensity-threshold colocalization."""

    intensity_threshold_used: float
    correlation_method: str = "threshold_based"

    def as_dict(self) -> dict[str, float | str]:
        return {
            "intensity_threshold_used": self.intensity_threshold_used,
            "correlation_method": self.correlation_method,
        }


@dataclass(frozen=True)
class WatershedThresholdBackend:
    """Backend-specific threshold primitives for watershed segmentation."""

    otsu: Callable[[Any], Any]
    li: Callable[[Any], Any]


class WatershedThresholdMethodStrategy(
    EnumKeyedStrategyMixin[ThresholdMethod],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal watershed threshold method dispatcher shared by backends."""

    __registry_key__ = "method_value"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "method"

    method: ClassVar[ThresholdMethod]
    method_value: ClassVar[str | None] = None

    @classmethod
    def for_method_value(
        cls,
        method: str,
    ) -> "WatershedThresholdMethodStrategy":
        try:
            return cls.for_enum_member(ThresholdMethod(method))
        except ValueError:
            return ManualWatershedThresholdMethodStrategy()

    @abstractmethod
    def threshold(
        self,
        backend: WatershedThresholdBackend,
        image: Any,
        raw_method: str,
    ) -> Any:
        """Return the threshold value for one watershed method."""


class OtsuWatershedThresholdMethodStrategy(WatershedThresholdMethodStrategy):
    method = ThresholdMethod.OTSU
    method_value = method.value

    def threshold(
        self,
        backend: WatershedThresholdBackend,
        image: Any,
        raw_method: str,
    ) -> Any:
        return backend.otsu(image)


class LiWatershedThresholdMethodStrategy(WatershedThresholdMethodStrategy):
    method = ThresholdMethod.LI
    method_value = method.value

    def threshold(
        self,
        backend: WatershedThresholdBackend,
        image: Any,
        raw_method: str,
    ) -> Any:
        return backend.li(image)


class ManualWatershedThresholdMethodStrategy(WatershedThresholdMethodStrategy):
    method = ThresholdMethod.MANUAL
    method_value = method.value

    def threshold(
        self,
        backend: WatershedThresholdBackend,
        image: Any,
        raw_method: str,
    ) -> float:
        return float(raw_method)
