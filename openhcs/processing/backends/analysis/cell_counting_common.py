"""Shared model types for cell-counting analysis backends."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


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
    colocalization_metrics: dict[str, float]
    overlap_positions: list[tuple[float, float]]


@dataclass(frozen=True)
class WatershedThresholdBackend:
    """Backend-specific threshold primitives for watershed segmentation."""

    otsu: Callable[[Any], Any]
    li: Callable[[Any], Any]

    def threshold(self, image: Any, method: str) -> Any:
        if method == ThresholdMethod.OTSU.value:
            return self.otsu(image)
        if method == ThresholdMethod.LI.value:
            return self.li(image)
        return float(method)
