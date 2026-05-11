"""Illumination backends for CellProfiler-compatible processing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging
import os
import time
from typing import ClassVar
import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit, prange

from openhcs.constants.constants import MemoryType
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.processing.backends.cellprofiler._backend import (
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    cellprofiler_backend_key,
)
from openhcs.processing.backends.cellprofiler.smoothing import MaskedLinearFilterRequest
from openhcs.core.runtime_values import project_image_mask_to_data_domain

_PROFILE_RUNTIME_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME"
NDIMAGE_CONSTANT_MODE = "constant"
logger = logging.getLogger(__name__)


def _profile_enabled() -> bool:
    return os.environ.get(_PROFILE_RUNTIME_ENV, "").lower() in {"1", "true", "yes"}


def _log_profile(label: str, seconds: float, **fields: object) -> None:
    if not _profile_enabled():
        return
    field_text = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("RUNTIME_PROFILE %s %.6fs %s", label, seconds, field_text)


@dataclass(frozen=True, slots=True)
class IlluminationMask:
    """Mask aligned with CellProfiler illumination input and stack slices."""

    mask: object | None
    pixel_data: np.ndarray

    @property
    def normalized(self) -> np.ndarray | None:
        if self.mask is None:
            return None
        mask_array = np.asarray(self.mask, dtype=bool)
        if mask_array.shape == self.pixel_data.shape:
            return mask_array
        if mask_array.shape == self.pixel_data.shape[-mask_array.ndim :]:
            return mask_array
        if (
            self.pixel_data.ndim == 2
            and mask_array.ndim == 3
            and mask_array.shape[0] == 1
        ):
            return mask_array[0]
        return mask_array

    def for_stack_slice(self, slice_index: int) -> np.ndarray | None:
        mask = self.normalized
        if mask is None:
            return None
        if mask.ndim >= 3 and slice_index < mask.shape[0]:
            return np.asarray(mask[slice_index], dtype=bool)
        return mask

    def for_output(self, illumination: np.ndarray) -> np.ndarray | None:
        mask = self.normalized
        if mask is None:
            return None
        if mask.shape == illumination.shape:
            return mask
        if illumination.ndim == 2 and mask.ndim >= 3:
            return np.any(mask, axis=0)
        return mask


@dataclass(frozen=True, slots=True)
class IlluminationGaussianFilter:
    """Gaussian filtering semantics for illumination smoothing and dilation."""

    pixel_data: np.ndarray
    mask: np.ndarray | None
    sigma: float

    def apply(self) -> np.ndarray:
        from scipy.ndimage import gaussian_filter

        if self.mask is None:
            return gaussian_filter(
                self.pixel_data,
                self.sigma,
                mode=NDIMAGE_CONSTANT_MODE,
                cval=0,
            )
        return MaskedLinearFilterRequest(
            pixels=self.pixel_data,
            mask=self.mask,
            operation=lambda image: gaussian_filter(
                image,
                self.sigma,
                mode=NDIMAGE_CONSTANT_MODE,
                cval=0,
            ),
        ).apply()


class IntensityChoice(Enum):
    """CellProfiler CorrectIlluminationCalculate intensity source."""

    REGULAR = "regular"
    BACKGROUND = "background"


class SmoothingMethod(Enum):
    """CellProfiler CorrectIlluminationCalculate smoothing method."""

    NONE = "none"
    CONVEX_HULL = "convex_hull"
    FIT_POLYNOMIAL = "fit_polynomial"
    MEDIAN_FILTER = "median_filter"
    GAUSSIAN_FILTER = "gaussian_filter"
    TO_AVERAGE = "to_average"
    SPLINES = "splines"


class FilterSizeMethod(Enum):
    """CellProfiler CorrectIlluminationCalculate filter-size mode."""

    AUTOMATIC = "automatic"
    OBJECT_SIZE = "object_size"
    MANUALLY = "manually"


class RescaleOption(Enum):
    """CellProfiler CorrectIlluminationCalculate output rescale mode."""

    YES = "yes"
    NO = "no"
    MEDIAN = "median"


class SplineBgMode(Enum):
    """CellProfiler CorrectIlluminationCalculate spline background mode."""

    AUTO = "auto"
    DARK = "dark"
    BRIGHT = "bright"
    GRAY = "gray"


class CalculationScope(Enum):
    """CellProfiler CorrectIlluminationCalculate image aggregation scope."""

    EACH = "each"
    ALL_FIRST_CYCLE = "all_first_cycle"
    ALL_ACROSS_CYCLES = "all_across_cycles"

    @property
    def uses_all_images(self) -> bool:
        return self in {
            CalculationScope.ALL_FIRST_CYCLE,
            CalculationScope.ALL_ACROSS_CYCLES,
        }


def coerce_illumination_enum(enum_type: type[Enum], value: object) -> Enum:
    """Coerce CellProfiler UI literals for illumination-owned enums."""
    return coerce_cellprofiler_enum(enum_type, value)


@dataclass(frozen=True, slots=True)
class SmoothingFilterSizeRequest:
    """Inputs needed to derive a smoothing filter size."""

    image_shape: tuple[int, ...]
    object_width: int
    manual_filter_size: int


class SmoothingFilterSizeStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal filter-size derivation for one closed CellProfiler mode."""

    __registry_key__ = "method_label"
    __skip_if_no_key__ = True
    method_label: ClassVar[str | None] = None
    method: ClassVar[FilterSizeMethod | None] = None

    @classmethod
    def for_method(
        cls,
        method: FilterSizeMethod,
    ) -> "SmoothingFilterSizeStrategy":
        return cls.__registry__[method.value]()

    @abstractmethod
    def calculate(self, request: SmoothingFilterSizeRequest) -> float:
        """Return the smoothing filter size."""


class ManualSmoothingFilterSizeStrategy(SmoothingFilterSizeStrategy):
    method = FilterSizeMethod.MANUALLY
    method_label = method.value

    def calculate(self, request: SmoothingFilterSizeRequest) -> float:
        return float(request.manual_filter_size)


class ObjectWidthSmoothingFilterSizeStrategy(SmoothingFilterSizeStrategy):
    method = FilterSizeMethod.OBJECT_SIZE
    method_label = method.value

    def calculate(self, request: SmoothingFilterSizeRequest) -> float:
        return request.object_width * 2.35 / 3.5


class AutomaticSmoothingFilterSizeStrategy(SmoothingFilterSizeStrategy):
    method = FilterSizeMethod.AUTOMATIC
    method_label = method.value

    def calculate(self, request: SmoothingFilterSizeRequest) -> float:
        return min(30.0, float(np.max(request.image_shape)) / 40.0)


class ConvexHullSmoothingBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Convex-hull illumination smoothing keyed by OpenHCS memory/provider."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def smooth_background_plane(
        self,
        pixel_data: np.ndarray,
        *,
        mask: np.ndarray | None,
        filter_size: float,
        morphology: object,
    ) -> np.ndarray:
        """Return a smoothed illumination background plane."""


class RankMedianSmoothingBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Rank-median illumination smoothing keyed by OpenHCS memory/provider."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def smooth_background_plane(
        self,
        pixel_data: np.ndarray,
        *,
        mask: np.ndarray | None,
        radius: int,
        morphology: object,
    ) -> np.ndarray:
        """Return a rank-median smoothed illumination background plane."""


class NumbaNumpyRankMedianSmoothingBackendStrategy(
    RankMedianSmoothingBackendStrategy,
):
    """NumPy-memory rank median matching skimage rank median border semantics."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = False

    def smooth_background_plane(
        self,
        pixel_data: np.ndarray,
        *,
        mask: np.ndarray | None,
        radius: int,
        morphology: object,
    ) -> np.ndarray:
        image = np.asarray(pixel_data, dtype=np.float32)
        if image.ndim != 2:
            raise NotImplementedError(
                "Rank-median illumination smoothing currently supports 2-D "
                f"NumPy planes, got shape {image.shape!r}."
            )
        mask = project_image_mask_to_data_domain(mask, image)
        if mask is not None and np.asarray(mask).shape != image.shape:
            raise ValueError(
                "Rank-median illumination mask must match the image shape; got "
                f"mask {np.asarray(mask).shape!r} for image {image.shape!r}."
        )
        footprint = np.asarray(morphology.disk_footprint(radius), dtype=np.bool_)
        row_offsets_y, row_radii_x = _rank_median_disk_rows(footprint)
        scaled = (image * 65535.0).astype(np.uint16)
        mask_array = (
            np.ones(image.shape, dtype=np.bool_)
            if mask is None
            else np.asarray(mask, dtype=np.bool_)
        )
        effective_scaled = scaled.copy()
        effective_scaled[~mask_array] = np.uint16(0)
        minimum_value = np.min(effective_scaled)
        phase_started_at = time.perf_counter()
        if np.all(effective_scaled == minimum_value):
            _log_profile(
                "rank_median_constant_minimum",
                time.perf_counter() - phase_started_at,
                radius=radius,
            )
            return np.full(image.shape, minimum_value, dtype=np.float32) / 65535.0
        _log_profile(
            "rank_median_constant_minimum",
            time.perf_counter() - phase_started_at,
            radius=radius,
        )

        phase_started_at = time.perf_counter()
        if _rank_median_global_minimum_is_majority_everywhere_numba(
            np.ascontiguousarray(effective_scaled),
            row_offsets_y,
            row_radii_x,
            minimum_value,
        ):
            _log_profile(
                "rank_median_minimum_majority",
                time.perf_counter() - phase_started_at,
                radius=radius,
                result=True,
            )
            return np.full(image.shape, minimum_value, dtype=np.float32) / 65535.0
        _log_profile(
            "rank_median_minimum_majority",
            time.perf_counter() - phase_started_at,
            radius=radius,
            result=False,
        )

        phase_started_at = time.perf_counter()
        values, inverse = np.unique(effective_scaled, return_inverse=True)
        _log_profile(
            "rank_median_unique_codes",
            time.perf_counter() - phase_started_at,
            radius=radius,
            value_count=int(values.size),
        )
        phase_started_at = time.perf_counter()
        code_image = inverse.reshape(image.shape).astype(np.int32, copy=False)
        result_codes = _rank_median_codes_2d_sliding_histogram_numba(
            np.ascontiguousarray(code_image),
            row_offsets_y,
            row_radii_x,
            int(values.size),
        )
        _log_profile(
            "rank_median_numba_codes",
            time.perf_counter() - phase_started_at,
            radius=radius,
            value_count=int(values.size),
        )
        result = values[result_codes]
        return result.astype(np.float32) / 65535.0


class NativeNumpyRankMedianSmoothingBackendStrategy(
    RankMedianSmoothingBackendStrategy,
):
    """Compact-domain skimage rank-median backend for NumPy planes."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NATIVE,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NATIVE
    is_default_backend = True

    def smooth_background_plane(
        self,
        pixel_data: np.ndarray,
        *,
        mask: np.ndarray | None,
        radius: int,
        morphology: object,
    ) -> np.ndarray:
        image = np.asarray(pixel_data, dtype=np.float32)
        projected_mask = project_image_mask_to_data_domain(mask, image)
        mask_array = (
            None if projected_mask is None else np.asarray(projected_mask, dtype=np.bool_)
        )
        if mask_array is not None and mask_array.shape != image.shape:
            raise ValueError(
                "Rank-median illumination mask must match the image shape; got "
                f"mask {mask_array.shape!r} for image {image.shape!r}."
            )
        footprint = np.asarray(morphology.disk_footprint(radius), dtype=np.bool_)
        row_offsets_y, row_radii_x = _rank_median_disk_rows(footprint)
        scaled = (image * 65535.0).astype(np.uint16)
        effective_scaled = scaled if mask_array is None else scaled.copy()
        if mask_array is not None:
            effective_scaled[~mask_array] = np.uint16(0)
        minimum_value = np.min(effective_scaled)
        if not np.all(effective_scaled == minimum_value) and (
            _rank_median_global_minimum_is_majority_everywhere_numba(
                np.ascontiguousarray(effective_scaled),
                row_offsets_y,
                row_radii_x,
                minimum_value,
            )
        ):
            return np.full(image.shape, minimum_value, dtype=np.float32) / 65535.0
        return self._smooth_compact_rank_median(
            effective_scaled,
            footprint,
        )

    @staticmethod
    def _smooth_compact_rank_median(
        scaled: np.ndarray,
        footprint: np.ndarray,
    ) -> np.ndarray:
        import skimage.filters

        effective_scaled = np.asarray(scaled, dtype=np.uint16)
        values, inverse = np.unique(effective_scaled, return_inverse=True)
        if values.size == 1:
            result = skimage.filters.median(
                effective_scaled,
                footprint,
                behavior="rank",
            )
            return result.astype(np.float32) / 65535.0
        code_dtype = (
            np.uint8 if values.size <= np.iinfo(np.uint8).max + 1 else np.uint16
        )
        code_image = inverse.reshape(effective_scaled.shape).astype(code_dtype)
        result_codes = skimage.filters.median(
            code_image,
            footprint,
            behavior="rank",
        )
        return values[result_codes].astype(np.float32) / 65535.0


class CentrosomeNumpyConvexHullSmoothingBackendStrategy(
    ConvexHullSmoothingBackendStrategy,
):
    """CellProfiler/centrosome reference convex-hull smoothing for NumPy planes."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.CENTROSOME,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.CENTROSOME
    is_default_backend = True

    def smooth_background_plane(
        self,
        pixel_data: np.ndarray,
        *,
        mask: np.ndarray | None,
        filter_size: float,
        morphology: object,
    ) -> np.ndarray:
        del filter_size, morphology
        import centrosome.cpmorphology
        import centrosome.filter

        image = np.asarray(pixel_data, dtype=np.float32)
        mask_array = (
            None
            if mask is None
            else np.asarray(mask, dtype=bool)
        )
        eroded = centrosome.cpmorphology.grey_erosion(
            image,
            2,
            mask_array,
        )
        transformed = centrosome.filter.convex_hull_transform(
            eroded,
            mask=mask_array,
        )
        return np.asarray(
            centrosome.cpmorphology.grey_dilation(
                transformed,
                2,
                mask_array,
            ),
            dtype=np.float32,
        )


class LegacyFastNumpyConvexHullSmoothingBackendStrategy(
    ConvexHullSmoothingBackendStrategy,
):
    """Fast CP3-compatible convex-hull smoothing for NumPy planes."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.LEGACY_FAST,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.LEGACY_FAST
    is_default_backend = False

    def smooth_background_plane(
        self,
        pixel_data: np.ndarray,
        *,
        mask: np.ndarray | None,
        filter_size: float,
        morphology: object,
    ) -> np.ndarray:
        del morphology
        from scipy.ndimage import grey_dilation, grey_erosion, maximum_filter

        image = np.asarray(pixel_data, dtype=np.float32)
        if image.ndim != 2:
            raise NotImplementedError(
                "Legacy-fast convex-hull smoothing currently supports 2-D "
                f"NumPy planes, got shape {image.shape!r}."
            )
        result = grey_dilation(
            maximum_filter(
                grey_erosion(image, size=3),
                size=max(1, int(filter_size)),
            ),
            size=3,
        )
        if mask is not None:
            result = np.asarray(result, dtype=np.float32)
            result[~np.asarray(mask, dtype=bool)] = 0
        return result.astype(np.float32, copy=False)


class ExactLevelSetNumpyConvexHullSmoothingBackendStrategy(
    ConvexHullSmoothingBackendStrategy,
):
    """Numba-accelerated exact level-set convex-hull reconstruction."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.EXACT,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.EXACT
    is_default_backend = False

    def smooth_background_plane(
        self,
        pixel_data: np.ndarray,
        *,
        mask: np.ndarray | None,
        filter_size: float,
        morphology: object,
    ) -> np.ndarray:
        del filter_size
        image = np.asarray(pixel_data, dtype=np.float32)
        if image.ndim != 2:
            raise NotImplementedError(
                "Exact convex-hull smoothing currently supports 2-D NumPy "
                f"planes, got shape {image.shape!r}."
            )
        valid_mask = (
            np.ones(image.shape, dtype=bool)
            if mask is None
            else np.asarray(mask, dtype=bool)
        )
        if valid_mask.shape != image.shape:
            raise ValueError(
                "Convex-hull smoothing requires a mask matching the 2-D "
                f"image plane, got mask {valid_mask.shape!r} for image "
                f"{image.shape!r}."
            )
        if not np.any(valid_mask):
            return np.zeros(image.shape, dtype=np.float32)

        eroded = _cellprofiler_masked_grey_erosion(
            image,
            valid_mask,
            _convex_hull_smoothing_footprint(morphology),
        )
        valid_values = eroded[valid_mask]
        thresholds = np.linspace(
            float(np.min(valid_values)),
            float(np.max(valid_values)),
            256,
            dtype=np.float32,
        )[1:]
        hull = _exact_level_set_convex_hull_smoothing_numba(
            np.ascontiguousarray(eroded, dtype=np.float32),
            np.ascontiguousarray(valid_mask, dtype=np.bool_),
            np.ascontiguousarray(thresholds, dtype=np.float32),
        )
        return _cellprofiler_masked_grey_dilation(
            hull,
            valid_mask,
            _convex_hull_smoothing_footprint(morphology),
        )


class NumbaExactLevelSetNumpyConvexHullSmoothingBackendStrategy(
    ExactLevelSetNumpyConvexHullSmoothingBackendStrategy,
):
    """Named alias for the accelerated exact NumPy provider."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA_EXACT,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA_EXACT
    is_default_backend = False


class NativeExactLevelSetNumpyConvexHullSmoothingBackendStrategy(
    ConvexHullSmoothingBackendStrategy,
):
    """Reference exact level-set convex-hull reconstruction for NumPy planes."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NATIVE_EXACT,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NATIVE_EXACT
    is_default_backend = False

    def smooth_background_plane(
        self,
        pixel_data: np.ndarray,
        *,
        mask: np.ndarray | None,
        filter_size: float,
        morphology: object,
    ) -> np.ndarray:
        del filter_size
        return _native_exact_level_set_convex_hull_smoothing(
            np.asarray(pixel_data, dtype=np.float32),
            None if mask is None else np.asarray(mask, dtype=bool),
            morphology,
        )


def _native_exact_level_set_convex_hull_smoothing(
    image: np.ndarray,
    mask: np.ndarray | None,
    morphology: object,
) -> np.ndarray:
    if image.ndim != 2:
        raise NotImplementedError(
            "Native exact convex-hull smoothing currently supports 2-D NumPy "
            f"planes, got shape {image.shape!r}."
        )
    valid_mask = (
        np.ones(image.shape, dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool)
    )
    if valid_mask.shape != image.shape:
        raise ValueError(
            "Convex-hull smoothing requires a mask matching the 2-D image "
            f"plane, got mask {valid_mask.shape!r} for image {image.shape!r}."
        )
    if not np.any(valid_mask):
        return np.zeros(image.shape, dtype=np.float32)

    footprint = _convex_hull_smoothing_footprint(morphology)
    eroded = _cellprofiler_masked_grey_erosion(image, valid_mask, footprint)
    valid_values = eroded[valid_mask]
    minimum = float(np.min(valid_values))
    maximum = float(np.max(valid_values))
    output = np.full(image.shape, minimum, dtype=np.float32)
    output[~valid_mask] = 0
    if maximum <= minimum:
        return _cellprofiler_masked_grey_dilation(output, valid_mask, footprint)

    for threshold in np.linspace(minimum, maximum, 256, dtype=np.float32)[1:]:
        level_mask = valid_mask & (eroded >= float(threshold))
        if not np.any(level_mask):
            continue
        output[morphology.convex_hull_image(level_mask) & valid_mask] = threshold
    return _cellprofiler_masked_grey_dilation(output, valid_mask, footprint)


def _convex_hull_smoothing_footprint(morphology: object) -> np.ndarray:
    """Return CP's radius-2 disk footprint for convex-hull smoothing."""
    return np.asarray(morphology.disk_footprint(2), dtype=bool)


def _cellprofiler_masked_grey_erosion(
    image: np.ndarray,
    mask: np.ndarray,
    footprint: np.ndarray,
) -> np.ndarray:
    """Match centrosome.cpmorphology.grey_erosion masking semantics."""
    from scipy import ndimage as ndi

    radius = max(1, int(np.ceil(np.max(np.asarray(footprint.shape)) / 2 - 0.5)))
    padded = np.ones(np.asarray(image.shape) + radius * 2, dtype=image.dtype)
    core = tuple(slice(radius, -radius) for _axis in image.shape)
    padded[core] = image
    padded_core = padded[core]
    padded_core[~mask] = 1
    eroded = ndi.grey_erosion(padded, footprint=footprint)[core]
    result = np.asarray(eroded, dtype=np.float32)
    result[~mask] = image[~mask]
    return result


def _cellprofiler_masked_grey_dilation(
    image: np.ndarray,
    mask: np.ndarray,
    footprint: np.ndarray,
) -> np.ndarray:
    """Match centrosome.cpmorphology.grey_dilation masking semantics."""
    from scipy import ndimage as ndi

    radius = max(1, int(np.ceil(np.max(np.asarray(footprint.shape)) / 2 - 0.5)))
    padded = np.zeros(np.asarray(image.shape) + radius * 2, dtype=image.dtype)
    core = tuple(slice(radius, -radius) for _axis in image.shape)
    padded[core] = image
    padded_core = padded[core]
    padded_core[~mask] = 0
    dilated = ndi.grey_dilation(padded, footprint=footprint)[core]
    result = np.asarray(dilated, dtype=np.float32)
    result[~mask] = image[~mask]
    return result


@njit(cache=True)
def _exact_level_set_convex_hull_smoothing_numba(
    image: np.ndarray,
    valid_mask: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    height, width = image.shape
    minimum = np.float32(0.0)
    maximum = np.float32(0.0)
    found_valid = False
    for y in range(height):
        for x in range(width):
            if not valid_mask[y, x]:
                continue
            value = image[y, x]
            if not found_valid:
                minimum = value
                maximum = value
                found_valid = True
            else:
                if value < minimum:
                    minimum = value
                if value > maximum:
                    maximum = value

    output = np.empty((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            output[y, x] = minimum if valid_mask[y, x] else np.float32(0.0)
    if (not found_valid) or maximum <= minimum:
        return output

    row_count2 = height * 2 + 1
    min_col_by_row = np.empty(row_count2, dtype=np.int64)
    max_col_by_row = np.empty(row_count2, dtype=np.int64)
    point_capacity = max(2, row_count2 * 2)
    point_x = np.empty(point_capacity, dtype=np.int64)
    point_y = np.empty(point_capacity, dtype=np.int64)
    hull_x = np.empty(point_capacity * 2, dtype=np.int64)
    hull_y = np.empty(point_capacity * 2, dtype=np.int64)
    for level_index in range(thresholds.size):
        threshold = thresholds[level_index]
        point_count = _collect_diamond_extreme_points(
            image,
            valid_mask,
            threshold,
            min_col_by_row,
            max_col_by_row,
            point_x,
            point_y,
        )
        if point_count == 0:
            continue
        hull_count = _monotone_chain_hull(
            point_x,
            point_y,
            point_count,
            hull_x,
            hull_y,
        )
        _paint_convex_hull(
            output,
            valid_mask,
            threshold,
            hull_x,
            hull_y,
            hull_count,
        )
    return output


@njit(cache=True)
def _collect_diamond_extreme_points(
    image: np.ndarray,
    valid_mask: np.ndarray,
    threshold: np.float32,
    min_col_by_row: np.ndarray,
    max_col_by_row: np.ndarray,
    point_x: np.ndarray,
    point_y: np.ndarray,
) -> int:
    height, width = image.shape
    row_count2 = height * 2 + 1
    for row_index in range(row_count2):
        min_col_by_row[row_index] = 9223372036854775807
        max_col_by_row[row_index] = -9223372036854775807

    for y in range(height):
        for x in range(width):
            if valid_mask[y, x] and image[y, x] >= threshold:
                _add_diamond_vertex(min_col_by_row, max_col_by_row, 2 * y - 1, 2 * x)
                _add_diamond_vertex(min_col_by_row, max_col_by_row, 2 * y + 1, 2 * x)
                _add_diamond_vertex(min_col_by_row, max_col_by_row, 2 * y, 2 * x - 1)
                _add_diamond_vertex(min_col_by_row, max_col_by_row, 2 * y, 2 * x + 1)

    point_count = 0
    for row_index in range(row_count2):
        max_col = max_col_by_row[row_index]
        if max_col < -9223372036854775800:
            continue
        row2 = row_index - 1
        min_col = min_col_by_row[row_index]
        point_x[point_count] = row2
        point_y[point_count] = min_col
        point_count += 1
        if max_col != min_col:
            point_x[point_count] = row2
            point_y[point_count] = max_col
            point_count += 1
    return point_count


@njit(cache=True)
def _add_diamond_vertex(
    min_col_by_row: np.ndarray,
    max_col_by_row: np.ndarray,
    row2: int,
    col2: int,
) -> None:
    row_index = row2 + 1
    if col2 < min_col_by_row[row_index]:
        min_col_by_row[row_index] = col2
    if col2 > max_col_by_row[row_index]:
        max_col_by_row[row_index] = col2


@njit(cache=True)
def _cross_points(
    ax: int,
    ay: int,
    bx: int,
    by: int,
    cx: int,
    cy: int,
) -> int:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


@njit(cache=True)
def _monotone_chain_hull(
    point_x: np.ndarray,
    point_y: np.ndarray,
    point_count: int,
    hull_x: np.ndarray,
    hull_y: np.ndarray,
) -> int:
    if point_count <= 1:
        if point_count == 1:
            hull_x[0] = point_x[0]
            hull_y[0] = point_y[0]
        return point_count

    hull_count = 0
    for index in range(point_count):
        px = point_x[index]
        py = point_y[index]
        while hull_count >= 2 and _cross_points(
            hull_x[hull_count - 2],
            hull_y[hull_count - 2],
            hull_x[hull_count - 1],
            hull_y[hull_count - 1],
            px,
            py,
        ) <= 0:
            hull_count -= 1
        hull_x[hull_count] = px
        hull_y[hull_count] = py
        hull_count += 1

    lower_count = hull_count
    for index in range(point_count - 2, -1, -1):
        px = point_x[index]
        py = point_y[index]
        while hull_count > lower_count and _cross_points(
            hull_x[hull_count - 2],
            hull_y[hull_count - 2],
            hull_x[hull_count - 1],
            hull_y[hull_count - 1],
            px,
            py,
        ) <= 0:
            hull_count -= 1
        hull_x[hull_count] = px
        hull_y[hull_count] = py
        hull_count += 1

    if hull_count > 1:
        hull_count -= 1
    return hull_count


@njit(cache=True)
def _paint_convex_hull(
    output: np.ndarray,
    valid_mask: np.ndarray,
    threshold: np.float32,
    hull_x: np.ndarray,
    hull_y: np.ndarray,
    hull_count: int,
) -> None:
    if hull_count <= 0:
        return
    if hull_count == 1:
        if hull_x[0] % 2 != 0 or hull_y[0] % 2 != 0:
            return
        y = hull_x[0] // 2
        x = hull_y[0] // 2
        if (
            y >= 0
            and y < valid_mask.shape[0]
            and x >= 0
            and x < valid_mask.shape[1]
            and valid_mask[y, x]
        ):
            output[y, x] = threshold
        return

    min_row2 = hull_x[0]
    max_row2 = hull_x[0]
    min_col2 = hull_y[0]
    max_col2 = hull_y[0]
    for index in range(1, hull_count):
        row2 = hull_x[index]
        col2 = hull_y[index]
        if row2 < min_row2:
            min_row2 = row2
        if row2 > max_row2:
            max_row2 = row2
        if col2 < min_col2:
            min_col2 = col2
        if col2 > max_col2:
            max_col2 = col2

    if hull_count == 2:
        _paint_line_hull(
            output,
            valid_mask,
            threshold,
            hull_x[0],
            hull_y[0],
            hull_x[1],
            hull_y[1],
            min_row2,
            max_row2,
            min_col2,
            max_col2,
        )
        return

    area2 = 0
    for index in range(hull_count):
        next_index = 0 if index == hull_count - 1 else index + 1
        area2 += hull_x[index] * hull_y[next_index]
        area2 -= hull_x[next_index] * hull_y[index]
    positive_orientation = area2 >= 0

    image_height, image_width = output.shape
    min_y = max(0, _ceil_div2(min_row2))
    max_y = min(image_height - 1, _floor_div2(max_row2))
    min_x = max(0, _ceil_div2(min_col2))
    max_x = min(image_width - 1, _floor_div2(max_col2))

    for y in range(min_y, max_y + 1):
        query_row2 = y * 2
        for x in range(min_x, max_x + 1):
            if not valid_mask[y, x]:
                continue
            query_col2 = x * 2
            inside = True
            for index in range(hull_count):
                next_index = 0 if index == hull_count - 1 else index + 1
                cross = _cross_points(
                    hull_x[index],
                    hull_y[index],
                    hull_x[next_index],
                    hull_y[next_index],
                    query_row2,
                    query_col2,
                )
                if positive_orientation:
                    if cross < 0:
                        inside = False
                        break
                elif cross > 0:
                    inside = False
                    break
            if inside:
                output[y, x] = threshold


@njit(cache=True)
def _ceil_div2(value: int) -> int:
    if value >= 0:
        return (value + 1) // 2
    return value // 2


@njit(cache=True)
def _floor_div2(value: int) -> int:
    if value >= 0:
        return value // 2
    return -((-value + 1) // 2)


@njit(cache=True)
def _paint_line_hull(
    output: np.ndarray,
    valid_mask: np.ndarray,
    threshold: np.float32,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    min_row2: int,
    max_row2: int,
    min_col2: int,
    max_col2: int,
) -> None:
    dx = x1 - x0
    dy = y1 - y0
    length2 = dx * dx + dy * dy
    if length2 == 0:
        if valid_mask[y0, x0]:
            output[y0, x0] = threshold
        return
    image_height, image_width = output.shape
    min_y = max(0, _ceil_div2(min_row2))
    max_y = min(image_height - 1, _floor_div2(max_row2))
    min_x = max(0, _ceil_div2(min_col2))
    max_x = min(image_width - 1, _floor_div2(max_col2))
    for y in range(min_y, max_y + 1):
        query_row2 = y * 2
        for x in range(min_x, max_x + 1):
            if not valid_mask[y, x]:
                continue
            query_col2 = x * 2
            dot = (query_row2 - x0) * dx + (query_col2 - y0) * dy
            if dot < 0 or dot > length2:
                continue
            cross = dx * (query_col2 - y0) - dy * (query_row2 - x0)
            if cross == 0:
                output[y, x] = threshold


def _rank_median_footprint_offsets(
    footprint: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    center_y = footprint.shape[0] // 2
    center_x = footprint.shape[1] // 2
    y, x = np.nonzero(footprint)
    return (
        (y - center_y).astype(np.int64, copy=False),
        (x - center_x).astype(np.int64, copy=False),
    )


def _rank_median_disk_rows(
    footprint: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse a dense disk footprint into per-row horizontal radii."""
    center_y = footprint.shape[0] // 2
    center_x = footprint.shape[1] // 2
    rows: list[int] = []
    radii: list[int] = []
    for y in range(footprint.shape[0]):
        xs = np.flatnonzero(footprint[y])
        if xs.size == 0:
            continue
        rows.append(y - center_y)
        radii.append(int(np.max(np.abs(xs - center_x))))
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(radii, dtype=np.int64),
    )


def _rank_median_native_reference(
    scaled: np.ndarray,
    footprint: np.ndarray,
) -> np.ndarray:
    import skimage.filters

    result = skimage.filters.median(
        scaled,
        footprint,
        behavior="rank",
    )
    return result.astype(np.float32) / 65535.0


@njit(cache=True)
def _rank_median_global_minimum_is_majority_everywhere_numba(
    image: np.ndarray,
    row_offsets_y: np.ndarray,
    row_radii_x: np.ndarray,
    minimum_value: np.uint16,
) -> bool:
    height, width = image.shape
    for y in range(height):
        total_count = 0
        minimum_count = 0

        for row_index in range(row_offsets_y.shape[0]):
            yy = y + row_offsets_y[row_index]
            if yy < 0 or yy >= height:
                continue
            radius_x = row_radii_x[row_index]
            right = radius_x
            if right >= width:
                right = width - 1
            for xx in range(0, right + 1):
                total_count += 1
                if image[yy, xx] == minimum_value:
                    minimum_count += 1

        if minimum_count <= total_count // 2:
            return False

        for x in range(1, width):
            for row_index in range(row_offsets_y.shape[0]):
                yy = y + row_offsets_y[row_index]
                if yy < 0 or yy >= height:
                    continue
                radius_x = row_radii_x[row_index]

                remove_x = x - 1 - radius_x
                if remove_x >= 0 and remove_x < width:
                    total_count -= 1
                    if image[yy, remove_x] == minimum_value:
                        minimum_count -= 1

                add_x = x + radius_x
                if add_x >= 0 and add_x < width:
                    total_count += 1
                    if image[yy, add_x] == minimum_value:
                        minimum_count += 1

            if minimum_count <= total_count // 2:
                return False
    return True


@njit(cache=True, parallel=True)
def _rank_median_codes_2d_sliding_histogram_numba(
    code_image: np.ndarray,
    row_offsets_y: np.ndarray,
    row_radii_x: np.ndarray,
    value_count: int,
) -> np.ndarray:
    height, width = code_image.shape
    output = np.empty((height, width), dtype=np.int32)
    for y in prange(height):
        tree = np.zeros(value_count + 1, dtype=np.int64)
        count = 0

        for row_index in range(row_offsets_y.shape[0]):
            yy = y + row_offsets_y[row_index]
            if yy < 0 or yy >= height:
                continue
            radius_x = row_radii_x[row_index]
            right = radius_x
            if right >= width:
                right = width - 1
            for xx in range(0, right + 1):
                _fenwick_add_code(tree, int(code_image[yy, xx]), 1)
                count += 1

        if count == 0:
            output[y, 0] = 0
        else:
            output[y, 0] = _fenwick_select_code(tree, count // 2)

        for x in range(1, width):
            for row_index in range(row_offsets_y.shape[0]):
                yy = y + row_offsets_y[row_index]
                if yy < 0 or yy >= height:
                    continue
                radius_x = row_radii_x[row_index]

                remove_x = x - 1 - radius_x
                if remove_x >= 0 and remove_x < width:
                    _fenwick_add_code(tree, int(code_image[yy, remove_x]), -1)
                    count -= 1

                add_x = x + radius_x
                if add_x >= 0 and add_x < width:
                    _fenwick_add_code(tree, int(code_image[yy, add_x]), 1)
                    count += 1

            if count == 0:
                output[y, x] = 0
            else:
                output[y, x] = _fenwick_select_code(tree, count // 2)
    return output


@njit(cache=True)
def _fenwick_add_code(tree: np.ndarray, code: int, delta: int) -> None:
    index = code + 1
    while index < tree.shape[0]:
        tree[index] += delta
        index += index & -index


@njit(cache=True)
def _fenwick_select_code(tree: np.ndarray, kth: int) -> int:
    index = 0
    bit = 1
    while bit < tree.shape[0]:
        bit <<= 1
    bit >>= 1
    target = kth + 1
    while bit != 0:
        next_index = index + bit
        if next_index < tree.shape[0] and tree[next_index] < target:
            index = next_index
            target -= tree[next_index]
        bit >>= 1
    return index


@njit(cache=True, parallel=True)
def _rank_median_uint16_2d_sliding_histogram_numba(
    image: np.ndarray,
    mask: np.ndarray,
    row_offsets_y: np.ndarray,
    row_radii_x: np.ndarray,
) -> np.ndarray:
    height, width = image.shape
    output = np.empty((height, width), dtype=np.uint16)
    histogram_size = 65536
    for y in prange(height):
        tree = np.zeros(histogram_size + 1, dtype=np.int64)
        count = 0

        for row_index in range(row_offsets_y.shape[0]):
            yy = y + row_offsets_y[row_index]
            if yy < 0 or yy >= height:
                continue
            radius_x = row_radii_x[row_index]
            right = radius_x
            if right >= width:
                right = width - 1
            for xx in range(0, right + 1):
                value = image[yy, xx] if mask[yy, xx] else np.uint16(0)
                _fenwick_add_uint16(tree, value, 1)
                count += 1

        if count == 0:
            output[y, 0] = np.uint16(0)
        else:
            output[y, 0] = _fenwick_select_uint16(tree, count // 2)

        for x in range(1, width):
            for row_index in range(row_offsets_y.shape[0]):
                yy = y + row_offsets_y[row_index]
                if yy < 0 or yy >= height:
                    continue
                radius_x = row_radii_x[row_index]

                remove_x = x - 1 - radius_x
                if remove_x >= 0 and remove_x < width:
                    value = (
                        image[yy, remove_x]
                        if mask[yy, remove_x]
                        else np.uint16(0)
                    )
                    _fenwick_add_uint16(tree, value, -1)
                    count -= 1

                add_x = x + radius_x
                if add_x >= 0 and add_x < width:
                    value = image[yy, add_x] if mask[yy, add_x] else np.uint16(0)
                    _fenwick_add_uint16(tree, value, 1)
                    count += 1

            if count == 0:
                output[y, x] = np.uint16(0)
            else:
                output[y, x] = _fenwick_select_uint16(tree, count // 2)
    return output


@njit(cache=True)
def _fenwick_add_uint16(tree: np.ndarray, value: np.uint16, delta: int) -> None:
    index = int(value) + 1
    while index < tree.shape[0]:
        tree[index] += delta
        index += index & -index


@njit(cache=True)
def _fenwick_select_uint16(tree: np.ndarray, kth: int) -> np.uint16:
    index = 0
    bit = 32768
    target = kth + 1
    while bit != 0:
        next_index = index + bit
        if next_index < tree.shape[0] and tree[next_index] < target:
            index = next_index
            target -= tree[next_index]
        bit >>= 1
    return np.uint16(index)


@njit(cache=True, parallel=True)
def _rank_median_uint16_2d_numba(
    image: np.ndarray,
    mask: np.ndarray,
    offsets_y: np.ndarray,
    offsets_x: np.ndarray,
) -> np.ndarray:
    height, width = image.shape
    output = np.empty((height, width), dtype=np.uint16)
    footprint_size = offsets_y.shape[0]
    for y in prange(height):
        values = np.empty(footprint_size, dtype=np.uint16)
        for x in range(width):
            count = 0
            for offset_index in range(footprint_size):
                yy = y + offsets_y[offset_index]
                xx = x + offsets_x[offset_index]
                if 0 <= yy < height and 0 <= xx < width:
                    values[count] = image[yy, xx] if mask[yy, xx] else 0
                    count += 1
            output[y, x] = _select_uint16(values, count, count // 2)
    return output


@njit(cache=True)
def _select_uint16(values: np.ndarray, count: int, kth: int) -> np.uint16:
    left = 0
    right = count - 1
    while True:
        if left == right:
            return values[left]
        pivot_index = (left + right) // 2
        pivot_index = _partition_uint16(values, left, right, pivot_index)
        if kth == pivot_index:
            return values[kth]
        if kth < pivot_index:
            right = pivot_index - 1
        else:
            left = pivot_index + 1


@njit(cache=True)
def _partition_uint16(
    values: np.ndarray,
    left: int,
    right: int,
    pivot_index: int,
) -> int:
    pivot_value = values[pivot_index]
    values[pivot_index] = values[right]
    values[right] = pivot_value
    store_index = left
    for index in range(left, right):
        if values[index] < pivot_value:
            current = values[store_index]
            values[store_index] = values[index]
            values[index] = current
            store_index += 1
    current = values[right]
    values[right] = values[store_index]
    values[store_index] = current
    return store_index


__all__ = [
    "ConvexHullSmoothingBackendStrategy",
    "ExactLevelSetNumpyConvexHullSmoothingBackendStrategy",
    "LegacyFastNumpyConvexHullSmoothingBackendStrategy",
    "NativeExactLevelSetNumpyConvexHullSmoothingBackendStrategy",
    "NativeNumpyRankMedianSmoothingBackendStrategy",
    "NumbaExactLevelSetNumpyConvexHullSmoothingBackendStrategy",
    "NumbaNumpyRankMedianSmoothingBackendStrategy",
    "RankMedianSmoothingBackendStrategy",
]
