"""Shared model types for cell-counting analysis backends."""

from __future__ import annotations

import math
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


def detection_method_catalog(
    *,
    blob_log: Callable[..., CellCountResult],
    blob_dog: Callable[..., CellCountResult],
    blob_doh: Callable[..., CellCountResult],
    watershed: Callable[..., CellCountResult],
    threshold: Callable[..., CellCountResult],
) -> dict[str, Callable[..., CellCountResult]]:
    """Build the detector catalog from the single detection enum authority."""
    return {
        DetectionMethod.BLOB_LOG.value: blob_log,
        DetectionMethod.BLOB_DOG.value: blob_dog,
        DetectionMethod.BLOB_DOH.value: blob_doh,
        DetectionMethod.WATERSHED.value: watershed,
        DetectionMethod.THRESHOLD.value: threshold,
    }


class ColocalizationMethod(Enum):
    """Methods for multi-channel colocalization analysis."""

    OVERLAP_AREA = "overlap_area"
    DISTANCE_BASED = "distance_based"
    INTENSITY_CORRELATION = "intensity_correlation"
    MANDERS_COEFFICIENTS = "manders_coefficients"


def colocalization_analyzer_catalog(
    *,
    distance_based: Callable[..., MultiChannelResult],
    overlap_area: Callable[..., MultiChannelResult],
    intensity_correlation: Callable[..., MultiChannelResult],
    manders_coefficients: Callable[..., MultiChannelResult],
) -> dict[str, Callable[..., MultiChannelResult]]:
    """Build the analyzer catalog from the single colocalization enum authority."""
    return {
        ColocalizationMethod.DISTANCE_BASED.value: distance_based,
        ColocalizationMethod.OVERLAP_AREA.value: overlap_area,
        ColocalizationMethod.INTENSITY_CORRELATION.value: intensity_correlation,
        ColocalizationMethod.MANDERS_COEFFICIENTS.value: manders_coefficients,
    }


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


@dataclass(frozen=True, slots=True)
class AreaFilterRequest:
    """Parallel per-object measurements filtered by cell area bounds."""

    positions: list[tuple[float, float]]
    areas: list[float]
    intensities: list[float]
    confidences: list[float]
    min_area: float
    max_area: float

    @classmethod
    def from_measurements(
        cls,
        positions: list[tuple[float, float]],
        areas: list[float],
        intensities: list[float],
        confidences: list[float],
        *,
        min_area: float,
        max_area: float,
    ) -> "AreaFilterRequest":
        return cls(
            positions=positions,
            areas=areas,
            intensities=intensities,
            confidences=confidences,
            min_area=min_area,
            max_area=max_area,
        )


@dataclass(frozen=True, slots=True)
class AreaFilterResult:
    """Area-filtered measurement vectors for detected cell candidates."""

    positions: list[tuple[float, float]]
    areas: list[float]
    intensities: list[float]
    confidences: list[float]

    def as_measurement_args(
        self,
    ) -> tuple[list[tuple[float, float]], list[float], list[float], list[float]]:
        return self.positions, self.areas, self.intensities, self.confidences


class AreaFilter:
    """Authoritative area gate shared by cell-counting detector backends."""

    def apply(self, request: AreaFilterRequest) -> AreaFilterResult:
        filtered_positions: list[tuple[float, float]] = []
        filtered_areas: list[float] = []
        filtered_intensities: list[float] = []
        filtered_confidences: list[float] = []

        for position, area, intensity, confidence in zip(
            request.positions,
            request.areas,
            request.intensities,
            request.confidences,
        ):
            if request.min_area <= area <= request.max_area:
                filtered_positions.append(position)
                filtered_areas.append(area)
                filtered_intensities.append(intensity)
                filtered_confidences.append(confidence)

        return AreaFilterResult(
            positions=filtered_positions,
            areas=filtered_areas,
            intensities=filtered_intensities,
            confidences=filtered_confidences,
        )


class ColocalizationAnalysis:
    """Shared cell-counting colocalization semantics for CPU and GPU callers."""

    DISTANCE_BASED_METHOD = "distance_based"
    OVERLAP_AREA_METHOD = "overlap_area"
    INTENSITY_CORRELATION_METHOD = "intensity_correlation"
    MANDERS_COEFFICIENTS_METHOD = "manders_coefficients"
    OVERLAP_DISTANCE_THRESHOLD = 2.0
    INTENSITY_PAIRING_DISTANCE_THRESHOLD = 5.0

    def distance_based(
        self,
        chan_1_result: CellCountResult,
        chan_2_result: CellCountResult,
        max_distance: float,
    ) -> MultiChannelResult:
        if not chan_1_result.cell_positions or not chan_2_result.cell_positions:
            return self.empty_result(
                chan_1_result,
                chan_2_result,
                self.DISTANCE_BASED_METHOD,
            )

        colocalized_pairs = self.nearest_pairs(
            chan_1_result.cell_positions,
            chan_2_result.cell_positions,
            max_distance,
        )
        colocalized_count = len(colocalized_pairs)
        total_cells = chan_1_result.cell_count + chan_2_result.cell_count
        colocalization_percentage = (
            2 * colocalized_count / total_cells * 100 if total_cells > 0 else 0
        )
        pair_distances = [
            self.distance(
                chan_1_result.cell_positions[first_index],
                chan_2_result.cell_positions[second_index],
            )
            for first_index, second_index in colocalized_pairs
        ]
        metrics = DistanceColocalizationMetrics(
            average_colocalization_distance=(
                sum(pair_distances) / len(pair_distances) if pair_distances else 0.0
            ),
            max_colocalization_distance=max(pair_distances) if pair_distances else 0.0,
            distance_threshold_used=max_distance,
        )

        return MultiChannelResult(
            slice_index=chan_1_result.slice_index,
            chan_1_results=chan_1_result,
            chan_2_results=chan_2_result,
            colocalization_method=self.DISTANCE_BASED_METHOD,
            colocalized_count=colocalized_count,
            colocalization_percentage=colocalization_percentage,
            chan_1_only_count=chan_1_result.cell_count - colocalized_count,
            chan_2_only_count=chan_2_result.cell_count - colocalized_count,
            colocalization_metrics=metrics.as_dict(),
            overlap_positions=[
                self.average_position(
                    chan_1_result.cell_positions[first_index],
                    chan_2_result.cell_positions[second_index],
                )
                for first_index, second_index in colocalized_pairs
            ],
        )

    def overlap_based(
        self,
        chan_1_result: CellCountResult,
        chan_2_result: CellCountResult,
        min_overlap_area: float,
    ) -> MultiChannelResult:
        result = self.distance_based(
            chan_1_result,
            chan_2_result,
            self.OVERLAP_DISTANCE_THRESHOLD,
        )
        result.colocalization_method = self.OVERLAP_AREA_METHOD
        result.colocalization_metrics["min_overlap_threshold"] = min_overlap_area
        result.colocalization_metrics["note"] = (
            "Approximated using distance-based method"
        )
        return result

    def intensity_based(
        self,
        chan_1_result: CellCountResult,
        chan_2_result: CellCountResult,
        intensity_threshold: float,
    ) -> MultiChannelResult:
        if not chan_1_result.cell_positions or not chan_2_result.cell_positions:
            return self.empty_result(
                chan_1_result,
                chan_2_result,
                self.INTENSITY_CORRELATION_METHOD,
            )

        colocalized_pairs: list[tuple[int, int]] = []
        overlap_positions: list[tuple[float, float]] = []
        for first_index, first_position in enumerate(chan_1_result.cell_positions):
            for second_index, second_position in enumerate(chan_2_result.cell_positions):
                if (
                    self.distance(first_position, second_position)
                    > self.INTENSITY_PAIRING_DISTANCE_THRESHOLD
                ):
                    continue
                if (
                    chan_1_result.cell_intensities[first_index] >= intensity_threshold
                    and chan_2_result.cell_intensities[second_index]
                    >= intensity_threshold
                ):
                    colocalized_pairs.append((first_index, second_index))
                    overlap_positions.append(
                        self.average_position(first_position, second_position)
                    )
                    break

        colocalized_count = len(colocalized_pairs)
        total_cells = chan_1_result.cell_count + chan_2_result.cell_count
        metrics = IntensityColocalizationMetrics(
            intensity_threshold_used=intensity_threshold,
        )
        return MultiChannelResult(
            slice_index=chan_1_result.slice_index,
            chan_1_results=chan_1_result,
            chan_2_results=chan_2_result,
            colocalization_method=self.INTENSITY_CORRELATION_METHOD,
            colocalized_count=colocalized_count,
            colocalization_percentage=(
                2 * colocalized_count / total_cells * 100 if total_cells > 0 else 0
            ),
            chan_1_only_count=chan_1_result.cell_count - colocalized_count,
            chan_2_only_count=chan_2_result.cell_count - colocalized_count,
            colocalization_metrics=metrics.as_dict(),
            overlap_positions=overlap_positions,
        )

    def manders(
        self,
        chan_1_result: CellCountResult,
        chan_2_result: CellCountResult,
        intensity_threshold: float,
    ) -> MultiChannelResult:
        if not chan_1_result.cell_positions or not chan_2_result.cell_positions:
            return self.empty_result(
                chan_1_result,
                chan_2_result,
                self.MANDERS_COEFFICIENTS_METHOD,
            )

        result = self.intensity_based(
            chan_1_result,
            chan_2_result,
            intensity_threshold,
        )
        total_intensity_1 = sum(chan_1_result.cell_intensities)
        total_intensity_2 = sum(chan_2_result.cell_intensities)
        coloc_intensity_1 = (
            chan_1_result.cell_intensities[0]
            if chan_1_result.cell_intensities and chan_2_result.cell_positions
            else 0.0
        )
        result.colocalization_method = self.MANDERS_COEFFICIENTS_METHOD
        result.colocalization_metrics.update(
            {
                "manders_m1": (
                    coloc_intensity_1 / total_intensity_1
                    if total_intensity_1 > 0
                    else 0
                ),
                "manders_m2": (
                    coloc_intensity_1 / total_intensity_2
                    if total_intensity_2 > 0
                    else 0
                ),
                "note": "Simplified cell-based Manders calculation",
            }
        )
        return result

    def empty_result(
        self,
        chan_1_result: CellCountResult,
        chan_2_result: CellCountResult,
        method: str,
    ) -> MultiChannelResult:
        return MultiChannelResult(
            slice_index=chan_1_result.slice_index,
            chan_1_results=chan_1_result,
            chan_2_results=chan_2_result,
            colocalization_method=method,
            colocalized_count=0,
            colocalization_percentage=0.0,
            chan_1_only_count=chan_1_result.cell_count,
            chan_2_only_count=chan_2_result.cell_count,
            colocalization_metrics={},
            overlap_positions=[],
        )

    def nearest_pairs(
        self,
        positions_1: list[tuple[float, float]],
        positions_2: list[tuple[float, float]],
        max_distance: float,
    ) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        used_second_indices: set[int] = set()
        for first_index, first_position in enumerate(positions_1):
            candidate_distances = [
                self.distance(first_position, second_position)
                for second_position in positions_2
            ]
            second_index = min(
                range(len(candidate_distances)),
                key=candidate_distances.__getitem__,
            )
            if (
                candidate_distances[second_index] <= max_distance
                and second_index not in used_second_indices
            ):
                pairs.append((first_index, second_index))
                used_second_indices.add(second_index)
        return pairs

    @staticmethod
    def distance(
        first_position: tuple[float, float],
        second_position: tuple[float, float],
    ) -> float:
        return math.sqrt(
            (first_position[0] - second_position[0]) ** 2
            + (first_position[1] - second_position[1]) ** 2
        )

    @staticmethod
    def average_position(
        first_position: tuple[float, float],
        second_position: tuple[float, float],
    ) -> tuple[float, float]:
        return (
            (first_position[0] + second_position[0]) / 2,
            (first_position[1] + second_position[1]) / 2,
        )


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
