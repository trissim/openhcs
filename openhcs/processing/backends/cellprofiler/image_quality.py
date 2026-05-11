"""Image-quality backends for CellProfiler-compatible processing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit

from openhcs.constants.constants import MemoryType
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    cellprofiler_backend_key,
)
from openhcs.processing.backends.cellprofiler.thresholding import threshold_primitives


class ThresholdMethod(Enum):
    OTSU = "otsu"
    LI = "li"
    TRIANGLE = "triangle"
    ISODATA = "isodata"
    MINIMUM = "minimum"
    MEAN = "mean"
    YEN = "yen"


@dataclass(frozen=True)
class _RadialSpectrumGeometry:
    radii: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True, slots=True)
class ImageQualityIntensityMetrics:
    total_area: int
    total_intensity: float
    mean_intensity: float
    median_intensity: float
    std_intensity: float
    mad_intensity: float
    min_intensity: float
    max_intensity: float


_RADIAL_SPECTRUM_GEOMETRY_CACHE: OrderedDict[
    tuple[int, int],
    _RadialSpectrumGeometry,
] = OrderedDict()
_RADIAL_SPECTRUM_GEOMETRY_CACHE_MAX_ENTRIES = 16


class ImageQualityBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Image-quality primitives keyed by OpenHCS memory type/provider."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def haralick_h3(self, image: np.ndarray, *, scale: int) -> float:
        """Return CP-style Haralick H3 correlation for one image plane."""

    @abstractmethod
    def radial_power_spectrum(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return CP-style radial Fourier spectrum bins."""


class ImageQualityThresholdStrategy(ABC, metaclass=AutoRegisterMeta):
    """Registered threshold primitive for MeasureImageQuality threshold metrics."""

    __registry_key__ = "method"
    __skip_if_no_key__ = True
    method: ClassVar[ThresholdMethod | None] = None
    primitive: ClassVar[Callable[[object, np.ndarray], float] | None] = None

    @classmethod
    def for_method(cls, method: ThresholdMethod) -> "ImageQualityThresholdStrategy":
        strategy_type = cls.__registry__.get(method)
        if strategy_type is None:
            raise NotImplementedError(f"Threshold method {method} not supported.")
        return strategy_type()

    def threshold(self, values: np.ndarray) -> float:
        """Return the requested threshold for a non-constant image."""
        primitive = type(self).primitive
        if primitive is None:
            raise NotImplementedError(f"{type(self).__name__} has no primitive.")
        return primitive(threshold_primitives(), values)


class OtsuImageQualityThresholdStrategy(ImageQualityThresholdStrategy):
    method = ThresholdMethod.OTSU
    primitive = lambda primitives, values: primitives.weighted_otsu_threshold(values)


class LiImageQualityThresholdStrategy(ImageQualityThresholdStrategy):
    method = ThresholdMethod.LI
    primitive = lambda primitives, values: primitives.li_threshold(values)


class TriangleImageQualityThresholdStrategy(ImageQualityThresholdStrategy):
    method = ThresholdMethod.TRIANGLE
    primitive = lambda primitives, values: primitives.triangle_threshold(values)


class IsodataImageQualityThresholdStrategy(ImageQualityThresholdStrategy):
    method = ThresholdMethod.ISODATA
    primitive = lambda primitives, values: primitives.isodata_threshold(values)


class MinimumImageQualityThresholdStrategy(ImageQualityThresholdStrategy):
    method = ThresholdMethod.MINIMUM
    primitive = lambda primitives, values: primitives.minimum_threshold(values)


class MeanImageQualityThresholdStrategy(ImageQualityThresholdStrategy):
    method = ThresholdMethod.MEAN
    primitive = lambda primitives, values: primitives.mean_threshold(values)


class YenImageQualityThresholdStrategy(ImageQualityThresholdStrategy):
    method = ThresholdMethod.YEN
    primitive = lambda primitives, values: primitives.yen_threshold(values)


class NumpyImageQualityBackendStrategy(ImageQualityBackendStrategy):
    """Independent NumPy implementation of image-quality primitives."""

    backend_key = cellprofiler_backend_key(MemoryType.NUMPY)
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NATIVE
    is_default_backend = False

    def haralick_h3(self, image: np.ndarray, *, scale: int) -> float:
        image_array = np.asarray(image, dtype=np.float32)
        if image_array.ndim != 2:
            raise NotImplementedError(
                "Image-quality Haralick correlation currently supports 2-D "
                f"NumPy planes, got shape {image_array.shape!r}."
            )
        return _haralick_h3_numpy(image_array, int(scale))

    def radial_power_spectrum(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image_array = np.asarray(image, dtype=np.float64)
        if image_array.ndim != 2:
            raise NotImplementedError(
                "Image-quality radial power spectrum currently supports 2-D "
                f"NumPy planes, got shape {image_array.shape!r}."
            )
        return _radial_power_spectrum_numpy(image_array)


class NumbaNumpyImageQualityBackendStrategy(NumpyImageQualityBackendStrategy):
    """Numba-accelerated NumPy image-quality backend."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def haralick_h3(self, image: np.ndarray, *, scale: int) -> float:
        image_array = np.asarray(image, dtype=np.float32)
        if image_array.ndim != 2:
            raise NotImplementedError(
                "Numba image-quality Haralick correlation currently supports "
                f"2-D NumPy planes, got shape {image_array.shape!r}."
            )
        return float(
            _haralick_h3_numba(
                np.ascontiguousarray(image_array, dtype=np.float32),
                int(scale),
            )
        )


class CentrosomeNumpyImageQualityBackendStrategy(ImageQualityBackendStrategy):
    """Explicit centrosome provider for image-quality primitives."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.CENTROSOME,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.CENTROSOME
    is_default_backend = False

    def haralick_h3(self, image: np.ndarray, *, scale: int) -> float:
        import centrosome.haralick

        image_array = np.asarray(image, dtype=np.float32)
        value = centrosome.haralick.Haralick(
            image_array,
            np.ones(image_array.shape, dtype=int),
            0,
            int(scale),
        ).H3()
        return _finite_scalar(value)

    def radial_power_spectrum(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import centrosome.radial_power_spectrum

        radii, magnitude, power = centrosome.radial_power_spectrum.rps(
            np.asarray(image)
        )
        return (
            np.asarray(radii),
            np.asarray(magnitude),
            np.asarray(power),
        )


def image_quality_backend(
    *,
    backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> ImageQualityBackendStrategy:
    """Return the selected image-quality backend."""
    return ImageQualityBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=backend_provider,
    )


def image_quality_focus_score(pixel_data: np.ndarray) -> float:
    """Calculate CP normalized-variance focus score."""
    if pixel_data.size == 0:
        return 0.0
    return float(
        _focus_score_numba(
            np.ascontiguousarray(pixel_data, dtype=np.float64),
        )
    )


def image_quality_local_focus_score(pixel_data: np.ndarray, scale: int) -> float:
    """Calculate CP local focus score using grid-based normalized variance."""
    if pixel_data.size == 0 or scale <= 0:
        return 0.0
    return float(
        _local_focus_score_numba(
            np.ascontiguousarray(pixel_data, dtype=np.float64),
            int(scale),
        )
    )


def image_quality_haralick_correlation(
    pixel_data: np.ndarray,
    scale: int,
    *,
    backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> float:
    """Calculate CellProfiler's Haralick H3 image-quality correlation."""
    if pixel_data.size == 0:
        return 0.0
    return image_quality_backend(
        backend_provider=backend_provider,
    ).haralick_h3(pixel_data, scale=scale)


def image_quality_power_spectrum_slope(
    pixel_data: np.ndarray,
    *,
    backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> float:
    """Calculate CellProfiler's log-log radial power spectrum slope."""
    if pixel_data.size == 0 or not image_quality_has_multiple_unique_values(pixel_data):
        return 0.0

    radii, magnitude, power = image_quality_backend(
        backend_provider=backend_provider,
    ).radial_power_spectrum(pixel_data)
    if np.sum(magnitude) <= 0:
        return 0.0

    valid = magnitude > 0
    radii = radii[valid].reshape((-1, 1))
    power = power[valid].reshape((-1, 1))
    if radii.shape[0] <= 1:
        return 0.0

    slope_value = _least_squares_log_log_slope_numba(
        np.ascontiguousarray(radii.ravel(), dtype=np.float64),
        np.ascontiguousarray(power.ravel(), dtype=np.float64),
    )
    return float(slope_value) if np.isfinite(slope_value) else 0.0


def image_quality_saturation(pixel_data: np.ndarray) -> tuple[float, float]:
    """Calculate percent of pixels at max and min values."""
    if pixel_data.size == 0:
        return 0.0, 0.0

    pixel_count = pixel_data.size
    max_val = np.max(pixel_data)
    min_val = np.min(pixel_data)
    num_maximal = np.sum(pixel_data == max_val)
    num_minimal = np.sum(pixel_data == min_val)
    return (
        100.0 * float(num_maximal) / float(pixel_count),
        100.0 * float(num_minimal) / float(pixel_count),
    )


def image_quality_intensity_metrics(
    pixel_data: np.ndarray,
) -> ImageQualityIntensityMetrics:
    """Calculate intensity-based image quality metrics."""
    if pixel_data.size == 0:
        return ImageQualityIntensityMetrics(
            total_area=0,
            total_intensity=0.0,
            mean_intensity=0.0,
            median_intensity=0.0,
            std_intensity=0.0,
            mad_intensity=0.0,
            min_intensity=0.0,
            max_intensity=0.0,
        )

    pixel_median = np.median(pixel_data)
    return ImageQualityIntensityMetrics(
        total_area=int(pixel_data.size),
        total_intensity=float(np.sum(pixel_data)),
        mean_intensity=float(np.mean(pixel_data)),
        median_intensity=float(pixel_median),
        std_intensity=float(np.std(pixel_data)),
        mad_intensity=float(np.median(np.abs(pixel_data - pixel_median))),
        min_intensity=float(np.min(pixel_data)),
        max_intensity=float(np.max(pixel_data)),
    )


def image_quality_threshold(pixel_data: np.ndarray, method: ThresholdMethod) -> float:
    """Calculate an automatic threshold using a MeasureImageQuality method."""
    if pixel_data.size == 0 or not image_quality_has_multiple_unique_values(pixel_data):
        return 0.0
    values = pixel_data.astype(np.float32, copy=False)
    return ImageQualityThresholdStrategy.for_method(method).threshold(values)


def image_quality_has_multiple_unique_values(pixel_data: np.ndarray) -> bool:
    """Return whether ``np.unique(pixel_data)`` would contain more than one value."""
    return bool(
        _has_multiple_unique_values_numba(
            np.ascontiguousarray(pixel_data, dtype=np.float32),
        )
    )


def _haralick_h3_numpy(image: np.ndarray, scale: int) -> float:
    if image.size == 0 or scale < 1 or image.shape[1] <= scale:
        return 0.0

    minimum = float(np.min(image))
    maximum = float(np.max(image))
    divisor = maximum - minimum if maximum > minimum else 1.0
    quantized = np.floor(((image - minimum) / divisor) * 8.0).astype(np.int16)
    quantized = np.clip(quantized, 0, 7)
    level_count = int(np.max(quantized)) + 1
    if level_count <= 0:
        return 0.0

    left = quantized[:, :-scale].ravel()
    right = quantized[:, scale:].ravel()
    pair_count = left.size
    if pair_count == 0:
        return 0.0
    flat_indexes = level_count * left + right
    matrix = np.bincount(
        flat_indexes,
        minlength=level_count * level_count,
    ).reshape(level_count, level_count).astype(float)
    return _haralick_h3_from_matrix(matrix / float(pair_count))


def _haralick_h3_from_matrix(matrix: np.ndarray) -> float:
    total = float(np.sum(matrix))
    if total <= 0.0:
        return 0.0
    matrix = matrix / total
    px = matrix.sum(axis=1)
    py = matrix.sum(axis=0)
    px_total = float(np.sum(px))
    py_total = float(np.sum(py))
    if px_total <= 0.0 or py_total <= 0.0:
        return 0.0
    px = px / px_total
    py = py / py_total
    levels = np.arange(matrix.shape[0], dtype=float) + 1.0
    mux = float(np.sum(levels * px))
    muy = float(np.sum(levels * py))
    sigmax = float(np.sqrt(np.sum(((levels - mux) ** 2) * px)))
    sigmay = float(np.sqrt(np.sum(((levels - muy) ** 2) * py)))
    if sigmax <= 0.0 or sigmay <= 0.0:
        return 0.0
    summed = float(np.sum(np.outer(levels, levels) * matrix))
    value = (summed - mux * muy) / (sigmax * sigmay)
    return value if np.isfinite(value) else 0.0


def _radial_power_spectrum_numpy(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.fftpack import fft2

    working = image.astype(np.float64, copy=False)
    if np.ptp(working) > 0.0:
        mean_value = float(np.mean(working))
        mad_value = float(np.median(np.abs(working - mean_value)))
        with np.errstate(divide="ignore", invalid="ignore"):
            working = working / mad_value

    centered = working - np.mean(working)
    magnitude = np.abs(fft2(centered))
    power = magnitude**2
    geometry = _radial_spectrum_geometry(image.shape)
    labels = geometry.labels
    if labels.size == 0:
        return (
            np.array([2], dtype=int),
            np.array([0], dtype=int),
            np.array([0], dtype=int),
        )
    radii_flat = geometry.radii.ravel()
    return (
        labels,
        np.bincount(
            radii_flat,
            weights=magnitude.ravel(),
            minlength=int(labels[-1]) + 1,
        )[labels],
        np.bincount(
            radii_flat,
            weights=power.ravel(),
            minlength=int(labels[-1]) + 1,
        )[labels],
    )


def _radial_spectrum_geometry(shape: tuple[int, int]) -> _RadialSpectrumGeometry:
    key = (int(shape[0]), int(shape[1]))
    geometry = _RADIAL_SPECTRUM_GEOMETRY_CACHE.get(key)
    if geometry is not None:
        _RADIAL_SPECTRUM_GEOMETRY_CACHE.move_to_end(key)
        return geometry

    height, width = key
    row2 = np.arange(height).reshape((height, 1)) ** 2
    col2 = np.arange(width) ** 2
    radii2 = row2 + col2
    radii2 = np.minimum(radii2, np.flipud(radii2))
    radii2 = np.minimum(radii2, np.fliplr(radii2))
    max_width = min(height, width) / 8.0
    geometry = _RadialSpectrumGeometry(
        radii=(np.floor(np.sqrt(radii2)).astype(int) + 1),
        labels=np.arange(2, int(np.floor(max_width)), dtype=int),
    )
    _RADIAL_SPECTRUM_GEOMETRY_CACHE[key] = geometry
    _RADIAL_SPECTRUM_GEOMETRY_CACHE.move_to_end(key)
    while len(_RADIAL_SPECTRUM_GEOMETRY_CACHE) > _RADIAL_SPECTRUM_GEOMETRY_CACHE_MAX_ENTRIES:
        _RADIAL_SPECTRUM_GEOMETRY_CACHE.popitem(last=False)
    return geometry


def _finite_scalar(value: object) -> float:
    array = np.asarray(value, dtype=float)
    if array.size != 1:
        return 0.0
    scalar = float(array.ravel()[0])
    return scalar if np.isfinite(scalar) else 0.0


@njit(cache=True)
def _focus_score_numba(pixel_data: np.ndarray) -> float:
    flat = pixel_data.ravel()
    count = flat.size
    if count == 0:
        return 0.0

    total = 0.0
    for index in range(count):
        total += flat[index]
    mean_value = total / float(count)
    if mean_value <= 0.0:
        return 0.0

    squared_sum = 0.0
    for index in range(count):
        diff = flat[index] - mean_value
        squared_sum += diff * diff
    return squared_sum / (float(count) * mean_value)


@njit(cache=True)
def _local_focus_score_numba(pixel_data: np.ndarray, scale: int) -> float:
    height, width = pixel_data.shape
    if height == 0 or width == 0 or scale <= 0:
        return 0.0

    grid_rows = (height + scale - 1) // scale
    grid_cols = (width + scale - 1) // scale
    grid_count = grid_rows * grid_cols

    sums = np.zeros(grid_count, dtype=np.float64)
    counts = np.zeros(grid_count, dtype=np.int64)
    for row in range(height):
        grid_row = int(row * float(grid_rows) / float(height))
        if grid_row >= grid_rows:
            grid_row = grid_rows - 1
        for col in range(width):
            grid_col = int(col * float(grid_cols) / float(width))
            if grid_col >= grid_cols:
                grid_col = grid_cols - 1
            grid_index = grid_row * grid_cols + grid_col
            sums[grid_index] += pixel_data[row, col]
            counts[grid_index] += 1

    means = np.zeros(grid_count, dtype=np.float64)
    valid_count = 0
    for grid_index in range(grid_count):
        count = counts[grid_index]
        if count <= 0:
            continue
        mean_value = sums[grid_index] / count
        if mean_value != 0.0 and np.isfinite(mean_value):
            means[grid_index] = mean_value
            valid_count += 1

    if valid_count == 0:
        return 0.0

    squared_sums = np.zeros(grid_count, dtype=np.float64)
    for row in range(height):
        grid_row = int(row * float(grid_rows) / float(height))
        if grid_row >= grid_rows:
            grid_row = grid_rows - 1
        for col in range(width):
            grid_col = int(col * float(grid_cols) / float(width))
            if grid_col >= grid_cols:
                grid_col = grid_cols - 1
            grid_index = grid_row * grid_cols + grid_col
            mean_value = means[grid_index]
            diff = pixel_data[row, col] - mean_value
            squared_sums[grid_index] += diff * diff

    local_norm_var = np.empty(valid_count, dtype=np.float64)
    output_index = 0
    for grid_index in range(grid_count):
        mean_value = means[grid_index]
        if mean_value == 0.0 or not np.isfinite(mean_value):
            continue
        value = squared_sums[grid_index] / (counts[grid_index] * mean_value)
        if np.isfinite(value):
            local_norm_var[output_index] = value
            output_index += 1

    if output_index == 0:
        return 0.0

    values = local_norm_var[:output_index]
    median_value = np.median(values)
    if (not np.isfinite(median_value)) or median_value <= 0.0:
        return 0.0

    mean_value = 0.0
    for index in range(output_index):
        mean_value += values[index]
    mean_value /= output_index

    variance = 0.0
    for index in range(output_index):
        diff = values[index] - mean_value
        variance += diff * diff
    variance /= output_index
    return variance / median_value


@njit(cache=True)
def _least_squares_log_log_slope_numba(
    radii: np.ndarray,
    power: np.ndarray,
) -> float:
    count = 0
    sum_x = 0.0
    sum_y = 0.0
    sum_xx = 0.0
    sum_xy = 0.0
    for index in range(radii.size):
        radius = radii[index]
        power_value = power[index]
        if radius <= 0.0 or power_value <= 0.0:
            continue
        x_value = np.log(radius)
        y_value = np.log(power_value)
        if not (np.isfinite(x_value) and np.isfinite(y_value)):
            continue
        count += 1
        sum_x += x_value
        sum_y += y_value
        sum_xx += x_value * x_value
        sum_xy += x_value * y_value
    if count <= 1:
        return 0.0
    denominator = float(count) * sum_xx - sum_x * sum_x
    if denominator == 0.0:
        return 0.0
    return (float(count) * sum_xy - sum_x * sum_y) / denominator


@njit(cache=True)
def _has_multiple_unique_values_numba(pixel_data: np.ndarray) -> bool:
    flat_size = pixel_data.size
    if flat_size <= 1:
        return False
    flat = pixel_data.ravel()
    first = flat[0]
    first_is_nan = np.isnan(first)
    for index in range(1, flat_size):
        value = flat[index]
        value_is_nan = np.isnan(value)
        if first_is_nan:
            if not value_is_nan:
                return True
        elif value_is_nan or value != first:
            return True
    return False


@njit(cache=True)
def _haralick_h3_numba(image: np.ndarray, scale: int) -> float:
    height, width = image.shape
    if height == 0 or width == 0 or scale < 1 or width <= scale:
        return 0.0

    minimum = image[0, 0]
    maximum = image[0, 0]
    for y in range(height):
        for x in range(width):
            value = image[y, x]
            if value < minimum:
                minimum = value
            if value > maximum:
                maximum = value
    divisor = maximum - minimum
    if divisor <= 0.0:
        divisor = 1.0

    level_count = 1
    for y in range(height):
        for x in range(width):
            level = int(((image[y, x] - minimum) / divisor) * 8.0)
            if level < 0:
                level = 0
            elif level > 7:
                level = 7
            if level + 1 > level_count:
                level_count = level + 1

    matrix = np.zeros((level_count, level_count), dtype=np.float64)
    pair_count = 0
    for y in range(height):
        for x in range(width - scale):
            left = int(((image[y, x] - minimum) / divisor) * 8.0)
            if left < 0:
                left = 0
            elif left > 7:
                left = 7
            right = int(((image[y, x + scale] - minimum) / divisor) * 8.0)
            if right < 0:
                right = 0
            elif right > 7:
                right = 7
            matrix[left, right] += 1.0
            pair_count += 1

    if pair_count == 0:
        return 0.0
    for y in range(level_count):
        for x in range(level_count):
            matrix[y, x] /= pair_count

    px = np.zeros(level_count, dtype=np.float64)
    py = np.zeros(level_count, dtype=np.float64)
    for y in range(level_count):
        for x in range(level_count):
            px[y] += matrix[y, x]
            py[x] += matrix[y, x]

    px_total = 0.0
    py_total = 0.0
    for index in range(level_count):
        px_total += px[index]
        py_total += py[index]
    if px_total <= 0.0 or py_total <= 0.0:
        return 0.0
    for index in range(level_count):
        px[index] /= px_total
        py[index] /= py_total

    mux = 0.0
    muy = 0.0
    for index in range(level_count):
        level_value = index + 1.0
        mux += level_value * px[index]
        muy += level_value * py[index]

    sigmax2 = 0.0
    sigmay2 = 0.0
    for index in range(level_count):
        level_value = index + 1.0
        dx = level_value - mux
        dy = level_value - muy
        sigmax2 += dx * dx * px[index]
        sigmay2 += dy * dy * py[index]
    if sigmax2 <= 0.0 or sigmay2 <= 0.0:
        return 0.0
    sigmax = np.sqrt(sigmax2)
    sigmay = np.sqrt(sigmay2)

    summed = 0.0
    for y in range(level_count):
        for x in range(level_count):
            summed += (y + 1.0) * (x + 1.0) * matrix[y, x]
    value = (summed - mux * muy) / (sigmax * sigmay)
    if np.isfinite(value):
        return value
    return 0.0


__all__ = [
    "CentrosomeNumpyImageQualityBackendStrategy",
    "ImageQualityBackendStrategy",
    "ImageQualityIntensityMetrics",
    "ImageQualityThresholdStrategy",
    "NumbaNumpyImageQualityBackendStrategy",
    "NumpyImageQualityBackendStrategy",
    "ThresholdMethod",
    "image_quality_backend",
    "image_quality_focus_score",
    "image_quality_haralick_correlation",
    "image_quality_has_multiple_unique_values",
    "image_quality_intensity_metrics",
    "image_quality_local_focus_score",
    "image_quality_power_spectrum_slope",
    "image_quality_saturation",
    "image_quality_threshold",
]
