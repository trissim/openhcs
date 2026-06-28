"""CellProfiler-compatible local maxima detection backend."""

from __future__ import annotations
from openhcs.processing.backends.cellprofiler.module_classes import CellProfilerModule

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta
import numpy as np
import scipy.ndimage
from skimage.feature import peak_local_max

from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.memory.decorators import numpy
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.core.public_api import public_names_from_objects
from openhcs.processing.materialization import csv_materializer

MAXIMA_RESULT_FIELDS = [
    "slice_index",
    "maxima_count",
    "min_distance_used",
    "threshold_used",
]


class ExcludeMode(Enum):
    THRESHOLD = "threshold"
    MASK = "mask"
    OBJECTS = "objects"


@dataclass(frozen=True, slots=True)
class MaximaResult:
    slice_index: int
    maxima_count: int
    min_distance_used: int
    threshold_used: float


@dataclass(frozen=True, slots=True)
class MaximaRequest:
    """Normalized maxima detection request."""

    image: np.ndarray
    min_distance: int
    min_intensity: float
    label_maxima: bool

    @property
    def threshold_abs(self) -> float | None:
        return self.min_intensity if self.min_intensity > 0 else None

    def detect(self) -> tuple[np.ndarray, MaximaResult]:
        maxima_coords = peak_local_max(
            self.image,
            min_distance=self.min_distance,
            threshold_abs=self.threshold_abs,
        )
        output = np.zeros(self.image.shape, dtype=np.float32)
        if len(maxima_coords) > 0:
            output[tuple(maxima_coords.T)] = 1.0
        if self.label_maxima:
            output = scipy.ndimage.label(output > 0)[0].astype(np.float32)
        return output, MaximaResult(
            slice_index=0,
            maxima_count=len(maxima_coords),
            min_distance_used=self.min_distance,
            threshold_used=self.threshold_abs if self.threshold_abs is not None else 0.0,
        )


class MaximaInputStrategy(
    EnumKeyedStrategyMixin[ExcludeMode],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Build the effective maxima source image for one CP exclusion mode."""

    __registry_key__ = "exclude_mode_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "exclude_mode"
    __enum_label_attr__ = "exclude_mode_label"
    exclude_mode: ClassVar[ExcludeMode | None] = None
    exclude_mode_label: ClassVar[str | None] = None

    @classmethod
    def for_exclude_mode(cls, exclude_mode: ExcludeMode) -> "MaximaInputStrategy":
        return cls.for_enum_member(exclude_mode)

    @abstractmethod
    def image(self, image: np.ndarray) -> np.ndarray:
        """Return the effective image for peak detection."""


class ThresholdMaximaInputStrategy(MaximaInputStrategy):
    exclude_mode = ExcludeMode.THRESHOLD

    def image(self, image: np.ndarray) -> np.ndarray:
        return image.copy()


class MaskMaximaInputStrategy(MaximaInputStrategy):
    exclude_mode = ExcludeMode.MASK

    def image(self, image: np.ndarray) -> np.ndarray:
        intensity_image = image[0].copy()
        intensity_image[~image[1].astype(bool)] = 0
        return intensity_image


class ObjectMaximaInputStrategy(MaskMaximaInputStrategy):
    exclude_mode = ExcludeMode.OBJECTS


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(
    (
        "maxima_results",
        csv_materializer(
            fields=MAXIMA_RESULT_FIELDS,
            analysis_type="maxima_detection",
        ),
    )
)
def find_maxima(
    image: np.ndarray,
    min_distance: int = 5,
    exclude_mode: ExcludeMode = ExcludeMode.THRESHOLD,
    min_intensity: float = 0.0,
    label_maxima: bool = True,
) -> tuple[np.ndarray, MaximaResult]:
    """Find local maxima under the requested CP exclusion policy."""
    exclude_mode = coerce_cellprofiler_enum(ExcludeMode, exclude_mode)
    return MaximaRequest(
        image=MaximaInputStrategy.for_exclude_mode(exclude_mode).image(image),
        min_distance=min_distance,
        min_intensity=min_intensity,
        label_maxima=label_maxima,
    ).detect()


@numpy
@special_outputs(
    (
        "maxima_results",
        csv_materializer(
            fields=MAXIMA_RESULT_FIELDS,
            analysis_type="maxima_detection",
        ),
    )
)
def find_maxima_with_mask(
    image: np.ndarray,
    min_distance: int = 5,
    min_intensity: float = 0.0,
    label_maxima: bool = True,
) -> tuple[np.ndarray, MaximaResult]:
    """Find local maxima within a stacked mask input."""
    maxima, result = MaximaRequest(
        image=MaximaInputStrategy.for_exclude_mode(ExcludeMode.MASK).image(image),
        min_distance=min_distance,
        min_intensity=min_intensity,
        label_maxima=label_maxima,
    ).detect()
    return maxima[np.newaxis, ...], result


class FindMaximaModule(CellProfilerModule):
    module_name = 'FindMaxima'
    function_name = 'find_maxima'
    validated = True
    contract = 'unknown'
    confidence = 1.0

__all__ = public_names_from_objects(
    ExcludeMode,
    MaskMaximaInputStrategy,
    MaximaInputStrategy,
    MaximaRequest,
    MaximaResult,
    ObjectMaximaInputStrategy,
    ThresholdMaximaInputStrategy,
    find_maxima,
    find_maxima_with_mask,
    extra_names=("MAXIMA_RESULT_FIELDS",),
)
