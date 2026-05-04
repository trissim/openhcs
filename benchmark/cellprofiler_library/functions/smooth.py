"""
Converted from CellProfiler: Smooth
Original: Smooth.run

Smooths (blurs) images using various filtering methods.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta
from numba import njit, prange

from openhcs.core.memory.decorators import numpy
from openhcs.core.runtime_values import (
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    image_payload_with_context,
)
from openhcs.processing.backends.cellprofiler._backend import (
    CellProfilerBackendProvider,
    normalize_cellprofiler_backend_provider,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from benchmark.cellprofiler_library.functions._enum import _coerce_function_enum


class SmoothingMethod(Enum):
    FIT_POLYNOMIAL = "fit_polynomial"
    GAUSSIAN_FILTER = "gaussian_filter"
    MEDIAN_FILTER = "median_filter"
    SMOOTH_KEEPING_EDGES = "smooth_keeping_edges"
    CIRCULAR_AVERAGE_FILTER = "circular_average_filter"
    SMOOTH_TO_AVERAGE = "smooth_to_average"


@dataclass(frozen=True, slots=True)
class SmoothingStrategyKey:
    backend_provider: CellProfilerBackendProvider
    method: SmoothingMethod


@dataclass(frozen=True, slots=True)
class SmoothingRequest:
    pixel_data: np.ndarray
    mask: np.ndarray | None
    backend_provider: CellProfilerBackendProvider
    method: SmoothingMethod
    object_size: float
    sigma: float
    edge_intensity_difference: float
    clip_polynomial: bool


def _smoothing_strategy_label(
    backend_provider: CellProfilerBackendProvider,
    method: SmoothingMethod,
) -> str:
    return f"{backend_provider.value}:{method.value}"


def _default_smoothing_backend_provider(
    method: SmoothingMethod,
    backend_provider: CellProfilerBackendProvider | None,
) -> CellProfilerBackendProvider:
    if backend_provider is not None:
        return normalize_cellprofiler_backend_provider(backend_provider)
    if method is SmoothingMethod.GAUSSIAN_FILTER:
        return CellProfilerBackendProvider.NUMBA
    return CellProfilerBackendProvider.NATIVE


class SmoothingStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal dispatch for CellProfiler smoothing algorithms."""

    __registry_key__ = "strategy_label"
    __skip_if_no_key__ = True
    strategy_label: ClassVar[str | None] = None
    strategy_key: ClassVar[SmoothingStrategyKey | None] = None

    @classmethod
    def for_request(cls, request: SmoothingRequest) -> "SmoothingStrategy":
        strategy_type = cls.__registry__.get(
            _smoothing_strategy_label(request.backend_provider, request.method)
        )
        if strategy_type is None:
            raise NotImplementedError(
                "No CellProfiler smoothing backend is registered for provider "
                f"{request.backend_provider.value!r} and method "
                f"{request.method.value!r}."
            )
        return strategy_type()

    @abstractmethod
    def smooth(self, request: SmoothingRequest) -> np.ndarray:
        """Return smoothed pixels for this strategy."""


class NumbaGaussianSmoothingStrategy(SmoothingStrategy):
    strategy_key = SmoothingStrategyKey(
        CellProfilerBackendProvider.NUMBA,
        SmoothingMethod.GAUSSIAN_FILTER,
    )
    strategy_label = _smoothing_strategy_label(
        strategy_key.backend_provider,
        strategy_key.method,
    )

    def smooth(self, request: SmoothingRequest) -> np.ndarray:
        return _gaussian_filter_numba(request.pixel_data, request.mask, request.sigma)


class NumpyGaussianSmoothingStrategy(SmoothingStrategy):
    strategy_key = SmoothingStrategyKey(
        CellProfilerBackendProvider.NATIVE,
        SmoothingMethod.GAUSSIAN_FILTER,
    )
    strategy_label = _smoothing_strategy_label(
        strategy_key.backend_provider,
        strategy_key.method,
    )

    def smooth(self, request: SmoothingRequest) -> np.ndarray:
        from scipy.ndimage import gaussian_filter

        return _masked_linear_filter(
            request.pixel_data,
            request.mask,
            lambda image: gaussian_filter(
                image,
                request.sigma,
                mode="constant",
                cval=0,
            ),
        )


class MedianSmoothingStrategy(SmoothingStrategy):
    strategy_key = SmoothingStrategyKey(
        CellProfilerBackendProvider.NATIVE,
        SmoothingMethod.MEDIAN_FILTER,
    )
    strategy_label = _smoothing_strategy_label(
        strategy_key.backend_provider,
        strategy_key.method,
    )

    def smooth(self, request: SmoothingRequest) -> np.ndarray:
        import centrosome.filter

        return centrosome.filter.median_filter(
            request.pixel_data,
            request.mask,
            request.object_size / 2 + 1,
        )


class EdgePreservingSmoothingStrategy(SmoothingStrategy):
    strategy_key = SmoothingStrategyKey(
        CellProfilerBackendProvider.NATIVE,
        SmoothingMethod.SMOOTH_KEEPING_EDGES,
    )
    strategy_label = _smoothing_strategy_label(
        strategy_key.backend_provider,
        strategy_key.method,
    )

    def smooth(self, request: SmoothingRequest) -> np.ndarray:
        from skimage.restoration import denoise_bilateral

        return denoise_bilateral(
            image=request.pixel_data.astype(float),
            channel_axis=None,
            sigma_color=request.edge_intensity_difference,
            sigma_spatial=request.sigma,
        )


class PolynomialSmoothingStrategy(SmoothingStrategy):
    strategy_key = SmoothingStrategyKey(
        CellProfilerBackendProvider.NATIVE,
        SmoothingMethod.FIT_POLYNOMIAL,
    )
    strategy_label = _smoothing_strategy_label(
        strategy_key.backend_provider,
        strategy_key.method,
    )

    def smooth(self, request: SmoothingRequest) -> np.ndarray:
        return _fit_polynomial(
            request.pixel_data,
            request.mask,
            request.clip_polynomial,
        )


class CircularAverageSmoothingStrategy(SmoothingStrategy):
    strategy_key = SmoothingStrategyKey(
        CellProfilerBackendProvider.NATIVE,
        SmoothingMethod.CIRCULAR_AVERAGE_FILTER,
    )
    strategy_label = _smoothing_strategy_label(
        strategy_key.backend_provider,
        strategy_key.method,
    )

    def smooth(self, request: SmoothingRequest) -> np.ndarray:
        import centrosome.filter

        return centrosome.filter.circular_average_filter(
            request.pixel_data,
            request.object_size / 2 + 1,
            request.mask,
        )


class SmoothToAverageStrategy(SmoothingStrategy):
    strategy_key = SmoothingStrategyKey(
        CellProfilerBackendProvider.NATIVE,
        SmoothingMethod.SMOOTH_TO_AVERAGE,
    )
    strategy_label = _smoothing_strategy_label(
        strategy_key.backend_provider,
        strategy_key.method,
    )

    def smooth(self, request: SmoothingRequest) -> np.ndarray:
        if request.mask is None:
            mean_value = np.mean(request.pixel_data)
        else:
            mean_value = np.mean(request.pixel_data[request.mask])
        return np.full(request.pixel_data.shape, mean_value, dtype=np.float32)


def _masked_linear_filter(
    image: np.ndarray,
    mask: np.ndarray | None,
    operation,
) -> np.ndarray:
    if mask is None:
        mask = np.ones(image.shape, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    masked_image = np.zeros(image.shape, dtype=image.dtype)
    masked_image[mask] = image[mask]
    weights = operation(mask.astype(float))
    filtered = operation(masked_image)
    return filtered / (weights + np.finfo(float).eps)


def _gaussian_filter_numba(
    image: np.ndarray,
    mask: np.ndarray | None,
    sigma: float,
) -> np.ndarray:
    image_array = np.asarray(image, dtype=np.float32)
    if image_array.ndim != 2:
        raise NotImplementedError(
            "Numba smoothing backend currently supports 2-D Gaussian planes."
        )
    kernel = _gaussian_kernel_1d(sigma)
    contiguous_image = np.ascontiguousarray(image_array)
    if mask is None:
        mask_array = np.ones(image_array.shape, dtype=np.bool_)
        return _masked_separable_gaussian_constant_2d_numba(
            contiguous_image,
            np.ascontiguousarray(mask_array),
            kernel,
        )

    mask_array = np.asarray(mask, dtype=np.bool_)
    if mask_array.shape != image_array.shape:
        raise ValueError(
            "Smoothing mask must match image shape; got "
            f"{mask_array.shape!r} for image {image_array.shape!r}."
        )
    return _masked_separable_gaussian_constant_2d_numba(
        contiguous_image,
        np.ascontiguousarray(mask_array),
        kernel,
    )


def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    sigma = float(sigma)
    if sigma <= 0:
        return np.ones((1,), dtype=np.float64)
    radius = max(1, int(round(4.0 * sigma)))
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (coordinates / sigma) ** 2)
    kernel /= np.sum(kernel)
    return kernel.astype(np.float64, copy=False)


@njit(cache=True, parallel=True)
def _separable_gaussian_constant_2d_numba(
    image: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    height, width = image.shape
    radius = kernel.size // 2
    temp = np.zeros((height, width), dtype=np.float64)
    output = np.zeros((height, width), dtype=np.float32)

    for row in prange(height):
        for col in range(width):
            value = 0.0
            for offset in range(kernel.size):
                source_col = col + offset - radius
                if 0 <= source_col < width:
                    value += float(image[row, source_col]) * kernel[offset]
            temp[row, col] = value

    for row in prange(height):
        for col in range(width):
            value = 0.0
            for offset in range(kernel.size):
                source_row = row + offset - radius
                if 0 <= source_row < height:
                    value += temp[source_row, col] * kernel[offset]
            output[row, col] = value
    return output


@njit(cache=True, parallel=True)
def _masked_separable_gaussian_constant_2d_numba(
    image: np.ndarray,
    mask: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    height, width = image.shape
    radius = kernel.size // 2
    temp_values = np.zeros((height, width), dtype=np.float64)
    temp_weights = np.zeros((height, width), dtype=np.float64)
    output = np.zeros((height, width), dtype=np.float32)
    eps = np.finfo(np.float64).eps

    for row in prange(height):
        for col in range(width):
            weighted_value = 0.0
            weight = 0.0
            for offset in range(kernel.size):
                source_col = col + offset - radius
                if 0 <= source_col < width and mask[row, source_col]:
                    kernel_value = kernel[offset]
                    weighted_value += float(image[row, source_col]) * kernel_value
                    weight += kernel_value
            temp_values[row, col] = weighted_value
            temp_weights[row, col] = weight

    for row in prange(height):
        for col in range(width):
            weighted_value = 0.0
            weight = 0.0
            for offset in range(kernel.size):
                source_row = row + offset - radius
                if 0 <= source_row < height:
                    kernel_value = kernel[offset]
                    weighted_value += temp_values[source_row, col] * kernel_value
                    weight += temp_weights[source_row, col] * kernel_value
            output[row, col] = weighted_value / (weight + eps)
    return output


def _fit_polynomial(
    image: np.ndarray,
    mask: np.ndarray | None,
    clip: bool,
) -> np.ndarray:
    if mask is None:
        mask = np.ones(image.shape, dtype=bool)
    valid = np.asarray(mask, dtype=bool) & (image > 0)
    if not np.any(valid):
        return image

    x, y = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    terms = (x, y, x * x, y * y, x * y, np.ones(image.shape))
    design = np.column_stack([term[valid].ravel() for term in terms])
    coeffs, *_ = np.linalg.lstsq(design, image[valid].ravel(), rcond=None)
    output = np.sum([coeff * term for coeff, term in zip(coeffs, terms)], axis=0)
    if clip:
        output = np.clip(output, 0, 1)
    return output


@numpy(contract=ProcessingContract.PURE_2D)
def smooth(
    image: np.ndarray,
    smoothing_method: SmoothingMethod = SmoothingMethod.GAUSSIAN_FILTER,
    auto_object_size: bool = True,
    object_size: float = 16.0,
    edge_intensity_difference: float = 0.1,
    clip_polynomial: bool = True,
    smoothing_backend_provider: CellProfilerBackendProvider | None = None,
) -> np.ndarray:
    """
    Smooth (blur) an image using various filtering methods.
    
    Args:
        image: Input grayscale image (H, W)
        smoothing_method: Method to use for smoothing
        auto_object_size: If True, calculate artifact diameter automatically
        object_size: Typical artifact diameter in pixels (used if auto_object_size=False)
        edge_intensity_difference: Edge intensity threshold for smooth_keeping_edges method
        clip_polynomial: Whether to clip polynomial fit results to [0, 1]
    
    Returns:
        Smoothed image (H, W)
    """
    smoothing_method = _coerce_function_enum(SmoothingMethod, smoothing_method)
    backend_provider = _default_smoothing_backend_provider(
        smoothing_method,
        smoothing_backend_provider,
    )
    pixel_data = np.asarray(image_payload_data(image), dtype=np.float32)
    mask = image_payload_mask(image)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
    
    # Determine object size
    if auto_object_size:
        calculated_size = max(1, np.mean(pixel_data.shape) / 40)
        calculated_size = min(30, calculated_size)
    else:
        calculated_size = object_size
    
    # Convert object size to sigma (FWHM to sigma conversion)
    sigma = calculated_size / 2.35
    
    request = SmoothingRequest(
        pixel_data=pixel_data,
        mask=mask,
        backend_provider=backend_provider,
        method=smoothing_method,
        object_size=float(calculated_size),
        sigma=float(sigma),
        edge_intensity_difference=float(edge_intensity_difference),
        clip_polynomial=bool(clip_polynomial),
    )
    output = SmoothingStrategy.for_request(request).smooth(request)

    return image_payload_with_context(
        np.asarray(output, dtype=np.float32),
        mask=mask,
        metadata=image_payload_metadata(image),
    )
