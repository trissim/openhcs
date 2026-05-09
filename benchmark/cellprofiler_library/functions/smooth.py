"""
Converted from CellProfiler: Smooth
Original: Smooth.run

Smooths (blurs) images using various filtering methods.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, ClassVar

from metaclass_registry import AutoRegisterMeta
from numba import njit

from openhcs.core.callable_contract import processing_prepare
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import (
    RuntimePure2DSliceBatchRequest,
    pure_2d_batch_executor,
)
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
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


@dataclass(frozen=True, slots=True)
class SmoothingBackendSelectionRequest:
    """CellProfiler Smooth settings needed to select an equivalent backend."""

    method: SmoothingMethod
    auto_object_size: bool
    object_size: float
    image_shape: tuple[int, int]

    @property
    def effective_object_size(self) -> float:
        if not self.auto_object_size:
            return self.object_size
        calculated_size = max(1.0, float(np.mean(self.image_shape)) / 40.0)
        return min(30.0, calculated_size)

    @property
    def sigma(self) -> float:
        return self.effective_object_size / 2.35


class SmoothingBackendProviderPolicy(
    EnumKeyedStrategyMixin[SmoothingMethod],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal SSOT for CellProfiler Smooth default backend semantics."""

    __registry_key__ = "strategy_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "method"

    method: ClassVar[SmoothingMethod]
    strategy_label: ClassVar[str | None] = None

    @classmethod
    def resolve(
        cls,
        method: SmoothingMethod,
        backend_provider: CellProfilerBackendProvider | None,
        selection_request: SmoothingBackendSelectionRequest | None = None,
    ) -> CellProfilerBackendProvider:
        if backend_provider is not None:
            return normalize_cellprofiler_backend_provider(backend_provider)
        return cls.for_enum_member(method).default_provider(selection_request)

    @abstractmethod
    def default_provider(
        self,
        selection_request: SmoothingBackendSelectionRequest | None,
    ) -> CellProfilerBackendProvider:
        """Return CP-compatible default provider for this Smooth method."""


class NativeSmoothingBackendProviderPolicy(SmoothingBackendProviderPolicy):
    """Smooth methods whose CP reference implementation is the native Python path."""

    def default_provider(
        self,
        selection_request: SmoothingBackendSelectionRequest | None,
    ) -> CellProfilerBackendProvider:
        del selection_request
        return CellProfilerBackendProvider.NATIVE


class GaussianSmoothingBackendProviderPolicy(SmoothingBackendProviderPolicy):
    """Gaussian Smooth backend selection by numerical equivalence class."""

    method = SmoothingMethod.GAUSSIAN_FILTER
    opencv_equivalent_min_sigma: ClassVar[float] = 4.0

    def default_provider(
        self,
        selection_request: SmoothingBackendSelectionRequest | None,
    ) -> CellProfilerBackendProvider:
        if (
            selection_request is not None
            and selection_request.sigma >= self.opencv_equivalent_min_sigma
        ):
            return CellProfilerBackendProvider.OPENCV
        return CellProfilerBackendProvider.NATIVE


class MedianSmoothingBackendProviderPolicy(NativeSmoothingBackendProviderPolicy):
    method = SmoothingMethod.MEDIAN_FILTER


class EdgePreservingSmoothingBackendProviderPolicy(NativeSmoothingBackendProviderPolicy):
    method = SmoothingMethod.SMOOTH_KEEPING_EDGES


class PolynomialSmoothingBackendProviderPolicy(NativeSmoothingBackendProviderPolicy):
    method = SmoothingMethod.FIT_POLYNOMIAL


class CircularAverageSmoothingBackendProviderPolicy(NativeSmoothingBackendProviderPolicy):
    method = SmoothingMethod.CIRCULAR_AVERAGE_FILTER


class SmoothToAverageBackendProviderPolicy(NativeSmoothingBackendProviderPolicy):
    method = SmoothingMethod.SMOOTH_TO_AVERAGE


def _smoothing_strategy_label(
    backend_provider: CellProfilerBackendProvider,
    method: SmoothingMethod,
) -> str:
    return f"{backend_provider.value}:{method.value}"


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


class OpenCVGaussianSmoothingStrategy(SmoothingStrategy):
    strategy_key = SmoothingStrategyKey(
        CellProfilerBackendProvider.OPENCV,
        SmoothingMethod.GAUSSIAN_FILTER,
    )
    strategy_label = _smoothing_strategy_label(
        strategy_key.backend_provider,
        strategy_key.method,
    )

    def smooth(self, request: SmoothingRequest) -> np.ndarray:
        return _masked_gaussian_filter_opencv(
            request.pixel_data,
            request.mask,
            request.sigma,
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


def _masked_linear_filter_stack(
    image_stack: np.ndarray,
    mask_stack: np.ndarray | None,
    operation,
) -> np.ndarray:
    if mask_stack is None:
        mask_stack = np.ones(image_stack.shape, dtype=bool)
    else:
        mask_stack = np.asarray(mask_stack, dtype=bool)
    masked_image = np.zeros(image_stack.shape, dtype=image_stack.dtype)
    masked_image[mask_stack] = image_stack[mask_stack]
    weights = operation(mask_stack.astype(float))
    filtered = operation(masked_image)
    return filtered / (weights + np.finfo(float).eps)


def _masked_gaussian_filter_opencv(
    image: np.ndarray,
    mask: np.ndarray | None,
    sigma: float,
) -> np.ndarray:
    import cv2

    image_array = np.ascontiguousarray(image, dtype=np.float32)
    if mask is None:
        mask_array = np.ones(image_array.shape, dtype=np.float32)
    else:
        mask_bool = np.asarray(mask, dtype=bool)
        if mask_bool.shape != image_array.shape:
            raise ValueError(
                "Smoothing mask must match image shape; got "
                f"{mask_bool.shape!r} for image {image_array.shape!r}."
            )
        mask_array = np.ascontiguousarray(mask_bool.astype(np.float32))
    kernel = _gaussian_kernel_1d(sigma).astype(np.float32, copy=False)
    masked_image = np.zeros(image_array.shape, dtype=np.float32)
    np.copyto(masked_image, image_array, where=mask_array.astype(bool, copy=False))
    filtered = cv2.sepFilter2D(
        masked_image,
        cv2.CV_32F,
        kernel,
        kernel,
        borderType=cv2.BORDER_CONSTANT,
    )
    weights = cv2.sepFilter2D(
        mask_array,
        cv2.CV_32F,
        kernel,
        kernel,
        borderType=cv2.BORDER_CONSTANT,
    )
    return filtered / (weights + np.finfo(np.float32).eps)


def _masked_gaussian_filter_stack_opencv(
    image_stack: np.ndarray,
    mask_stack: np.ndarray | None,
    sigma: float,
) -> np.ndarray:
    return np.stack(
        [
            _masked_gaussian_filter_opencv(
                image_stack[index],
                None if mask_stack is None else mask_stack[index],
                sigma,
            )
            for index in range(image_stack.shape[0])
        ],
        axis=0,
    )


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
        return _separable_gaussian_normalized_constant_2d_numba(
            contiguous_image,
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


@njit(cache=True)
def _separable_gaussian_normalized_constant_2d_numba(
    image: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    height, width = image.shape
    radius = kernel.size // 2
    temp = np.zeros((height, width), dtype=np.float64)
    x_weights = np.zeros(width, dtype=np.float64)
    y_weights = np.zeros(height, dtype=np.float64)
    output = np.zeros((height, width), dtype=np.float32)

    for row in range(height):
        for col in range(width):
            value = 0.0
            weight = 0.0
            for offset in range(kernel.size):
                source_col = col + offset - radius
                if 0 <= source_col < width:
                    kernel_value = kernel[offset]
                    value += float(image[row, source_col]) * kernel_value
                    weight += kernel_value
            temp[row, col] = value
            if row == 0:
                x_weights[col] = weight

    for row in range(height):
        y_weight = 0.0
        for offset in range(kernel.size):
            source_row = row + offset - radius
            if 0 <= source_row < height:
                y_weight += kernel[offset]
        y_weights[row] = y_weight
        for col in range(width):
            value = 0.0
            for offset in range(kernel.size):
                source_row = row + offset - radius
                if 0 <= source_row < height:
                    value += temp[source_row, col] * kernel[offset]
            output[row, col] = value / (x_weights[col] * y_weights[row] + np.finfo(np.float64).eps)
    return output


@njit(cache=True)
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

    for row in range(height):
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

    for row in range(height):
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
    pixel_data = np.asarray(image_payload_data(image), dtype=np.float32)
    backend_provider = SmoothingBackendProviderPolicy.resolve(
        smoothing_method,
        smoothing_backend_provider,
        SmoothingBackendSelectionRequest(
            method=smoothing_method,
            auto_object_size=auto_object_size,
            object_size=float(object_size),
            image_shape=tuple(int(axis) for axis in pixel_data.shape),
        ),
    )
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
        metadata=image_payload_metadata(image).without_unit_interval_intensity_scale(),
    )


def _smooth_batch(request: RuntimePure2DSliceBatchRequest) -> list[Any]:
    slices_2d = request.slices_2d
    kwargs = request.kwargs
    smoothing_method = _coerce_function_enum(
        SmoothingMethod,
        kwargs.get("smoothing_method", SmoothingMethod.GAUSSIAN_FILTER),
    )
    pixel_stack = np.ascontiguousarray(
        np.stack(
            [
                np.asarray(image_payload_data(slice_2d), dtype=np.float32)
                for slice_2d in slices_2d
            ],
            axis=0,
        ),
    )
    backend_provider = SmoothingBackendProviderPolicy.resolve(
        smoothing_method,
        kwargs.get("smoothing_backend_provider"),
        SmoothingBackendSelectionRequest(
            method=smoothing_method,
            auto_object_size=bool(kwargs.get("auto_object_size", True)),
            object_size=float(kwargs.get("object_size", 16.0)),
            image_shape=tuple(int(axis) for axis in pixel_stack.shape[1:]),
        ),
    )
    if (
        smoothing_method is not SmoothingMethod.GAUSSIAN_FILTER
        or backend_provider
        not in {CellProfilerBackendProvider.NATIVE, CellProfilerBackendProvider.OPENCV}
    ):
        return [
            request.execute_one(slice_index)
            for slice_index in range(request.slice_count)
        ]

    masks = tuple(image_payload_mask(slice_2d) for slice_2d in slices_2d)
    mask_stack = None
    if any(mask is not None for mask in masks):
        mask_stack = np.stack(
            [
                np.ones(pixel_stack.shape[1:], dtype=bool)
                if mask is None
                else np.asarray(mask, dtype=bool)
                for mask in masks
            ],
            axis=0,
        )

    if bool(kwargs.get("auto_object_size", True)):
        calculated_size = max(1, np.mean(pixel_stack.shape[1:]) / 40)
        calculated_size = min(30, calculated_size)
    else:
        calculated_size = kwargs.get("object_size", 16.0)
    sigma = float(calculated_size) / 2.35

    if backend_provider is CellProfilerBackendProvider.OPENCV:
        output_stack = _masked_gaussian_filter_stack_opencv(
            pixel_stack,
            mask_stack,
            sigma,
        ).astype(np.float32, copy=False)
    else:
        from scipy.ndimage import gaussian_filter

        output_stack = _masked_linear_filter_stack(
            pixel_stack,
            mask_stack,
            lambda image: gaussian_filter(
                image,
                (0.0, sigma, sigma),
                mode="constant",
                cval=0,
            ),
        ).astype(np.float32, copy=False)

    return [
        image_payload_with_context(
            output_stack[slice_index],
            mask=masks[slice_index],
            metadata=image_payload_metadata(slice_2d).without_unit_interval_intensity_scale(),
        )
        for slice_index, slice_2d in enumerate(slices_2d)
    ]


@processing_prepare(smooth)
def _prepare_smooth() -> None:
    """Compile default Gaussian smoothing before timed execution."""
    image = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape((64, 64))
    smooth.__wrapped__(image)


pure_2d_batch_executor(_smooth_batch)(smooth)
