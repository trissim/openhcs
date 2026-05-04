"""
Converted from CellProfiler: CorrectIlluminationCalculate
Calculates an illumination correction function to correct uneven illumination/lighting/shading.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit

from openhcs.core.memory.decorators import numpy
from openhcs.core.runtime_values import (
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    image_payload_with_context,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.backends.cellprofiler._backend import CellProfilerBackendProvider
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.processing.materialization import csv_materializer

from benchmark.cellprofiler_library.functions._enum import _coerce_function_enum


class IntensityChoice(Enum):
    REGULAR = "regular"
    BACKGROUND = "background"


class SmoothingMethod(Enum):
    NONE = "none"
    CONVEX_HULL = "convex_hull"
    FIT_POLYNOMIAL = "fit_polynomial"
    MEDIAN_FILTER = "median_filter"
    GAUSSIAN_FILTER = "gaussian_filter"
    TO_AVERAGE = "to_average"
    SPLINES = "splines"


class FilterSizeMethod(Enum):
    AUTOMATIC = "automatic"
    OBJECT_SIZE = "object_size"
    MANUALLY = "manually"


class RescaleOption(Enum):
    YES = "yes"
    NO = "no"
    MEDIAN = "median"


class SplineBgMode(Enum):
    AUTO = "auto"
    DARK = "dark"
    BRIGHT = "bright"
    GRAY = "gray"


class CalculationScope(Enum):
    EACH = "each"
    ALL_FIRST_CYCLE = "all_first_cycle"
    ALL_ACROSS_CYCLES = "all_across_cycles"


@dataclass
class IlluminationStats:
    slice_index: int
    min_value: float
    max_value: float
    mean_value: float
    calculation_type: str
    smoothing_method: str


ROBUST_FACTOR = 0.02
NDIMAGE_CONSTANT_MODE = "constant"


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


@dataclass(frozen=True, slots=True)
class SmoothingPlaneRequest:
    """Authoritative smoothing context for illumination background estimation."""

    pixel_data: np.ndarray
    mask: np.ndarray | None
    smoothing_method: SmoothingMethod
    filter_size: float
    spline_bg_mode: SplineBgMode
    spline_points: int
    spline_threshold: float
    spline_rescale: float
    spline_max_iterations: int
    spline_convergence: float
    automatic_splines: bool
    morphology: MorphologyBackendStrategy
    convex_hull_backend_provider: CellProfilerBackendProvider | None
    rank_median_backend_provider: CellProfilerBackendProvider | None

    @property
    def sigma(self) -> float:
        return self.filter_size / 2.35


class SmoothingPlaneStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal smoothing implementation for one closed CellProfiler mode."""

    __registry_key__ = "method_label"
    __skip_if_no_key__ = True
    method_label: ClassVar[str | None] = None
    method: ClassVar[SmoothingMethod | None] = None

    @classmethod
    def for_method(cls, method: SmoothingMethod) -> "SmoothingPlaneStrategy":
        return cls.__registry__[method.value]()

    @abstractmethod
    def smooth(self, request: SmoothingPlaneRequest) -> np.ndarray:
        """Return the smoothed illumination plane."""


class NoSmoothingPlaneStrategy(SmoothingPlaneStrategy):
    method = SmoothingMethod.NONE
    method_label = method.value

    def smooth(self, request: SmoothingPlaneRequest) -> np.ndarray:
        return request.pixel_data


class FitPolynomialSmoothingPlaneStrategy(SmoothingPlaneStrategy):
    method = SmoothingMethod.FIT_POLYNOMIAL
    method_label = method.value

    def smooth(self, request: SmoothingPlaneRequest) -> np.ndarray:
        return _fit_polynomial_surface(
            request.pixel_data,
            request.mask,
        )


class GaussianFilterSmoothingPlaneStrategy(SmoothingPlaneStrategy):
    method = SmoothingMethod.GAUSSIAN_FILTER
    method_label = method.value

    def smooth(self, request: SmoothingPlaneRequest) -> np.ndarray:
        return _masked_gaussian_filter(
            request.pixel_data,
            request.mask,
            request.sigma,
        )


class MedianFilterSmoothingPlaneStrategy(SmoothingPlaneStrategy):
    method = SmoothingMethod.MEDIAN_FILTER
    method_label = method.value

    def smooth(self, request: SmoothingPlaneRequest) -> np.ndarray:
        from openhcs.processing.backends.cellprofiler.illumination import (
            RankMedianSmoothingBackendStrategy,
        )

        filter_sigma = max(1, int(request.sigma + 0.5))
        return RankMedianSmoothingBackendStrategy.for_memory_type(
            backend_provider=request.rank_median_backend_provider,
        ).smooth_background_plane(
            request.pixel_data,
            mask=request.mask,
            radius=filter_sigma,
            morphology=request.morphology,
        )


class AverageSmoothingPlaneStrategy(SmoothingPlaneStrategy):
    method = SmoothingMethod.TO_AVERAGE
    method_label = method.value

    def smooth(self, request: SmoothingPlaneRequest) -> np.ndarray:
        if request.mask is not None:
            mean_val = np.mean(request.pixel_data[request.mask])
        else:
            mean_val = np.mean(request.pixel_data)
        return np.full(
            request.pixel_data.shape,
            mean_val,
            dtype=request.pixel_data.dtype,
        )


class ConvexHullSmoothingPlaneStrategy(SmoothingPlaneStrategy):
    method = SmoothingMethod.CONVEX_HULL
    method_label = method.value

    def smooth(self, request: SmoothingPlaneRequest) -> np.ndarray:
        from openhcs.processing.backends.cellprofiler.illumination import (
            ConvexHullSmoothingBackendStrategy,
        )

        return ConvexHullSmoothingBackendStrategy.for_memory_type(
            backend_provider=request.convex_hull_backend_provider,
        ).smooth_background_plane(
            request.pixel_data,
            mask=request.mask,
            filter_size=request.filter_size,
            morphology=request.morphology,
        )


class SplinesSmoothingPlaneStrategy(SmoothingPlaneStrategy):
    method = SmoothingMethod.SPLINES
    method_label = method.value

    def smooth(self, request: SmoothingPlaneRequest) -> np.ndarray:
        from scipy.interpolate import RectBivariateSpline

        pixel_data = request.pixel_data
        h, w = pixel_data.shape
        if request.automatic_splines:
            shortest_side = min(h, w)
            scale = max(1, shortest_side // 200)
            n_points = 5
        else:
            scale = int(request.spline_rescale)
            n_points = request.spline_points
        downsampled = pixel_data[::scale, ::scale]
        dh, dw = downsampled.shape
        y_points = np.linspace(0, dh - 1, n_points)
        x_points = np.linspace(0, dw - 1, n_points)
        yi = np.clip(np.round(y_points).astype(int), 0, dh - 1)
        xi = np.clip(np.round(x_points).astype(int), 0, dw - 1)
        spline = RectBivariateSpline(
            y_points,
            x_points,
            downsampled[np.ix_(yi, xi)],
            kx=3,
            ky=3,
        )
        result = spline(
            np.linspace(0, dh - 1, h),
            np.linspace(0, dw - 1, w),
        )
        if request.mask is not None:
            result[request.mask] -= np.mean(result[request.mask])
        else:
            result -= np.mean(result)
        return result


def _masked_gaussian_filter(
    pixel_data: np.ndarray,
    mask: np.ndarray | None,
    sigma: float,
) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    if mask is None:
        return gaussian_filter(
            pixel_data,
            sigma,
            mode=NDIMAGE_CONSTANT_MODE,
            cval=0,
        )

    masked_data = pixel_data.copy()
    masked_data[~mask] = 0
    smoothed = gaussian_filter(
        masked_data,
        sigma,
        mode=NDIMAGE_CONSTANT_MODE,
        cval=0,
    )
    mask_smoothed = gaussian_filter(
        mask.astype(float),
        sigma,
        mode=NDIMAGE_CONSTANT_MODE,
        cval=0,
    )
    return smoothed / np.maximum(mask_smoothed, 1e-10)


def _fit_polynomial_surface(
    pixel_data: np.ndarray,
    mask: np.ndarray | None,
) -> np.ndarray:
    """Fit CP's quadratic illumination surface without dense design matrices."""
    image = np.ascontiguousarray(pixel_data, dtype=np.float64)
    if image.ndim != 2:
        raise NotImplementedError(
            "Fit-polynomial illumination smoothing currently supports 2-D "
            f"NumPy planes, got shape {image.shape!r}."
        )
    mask_array = (
        np.empty((0, 0), dtype=np.bool_)
        if mask is None
        else np.ascontiguousarray(mask, dtype=np.bool_)
    )
    if mask is not None and mask_array.shape != image.shape:
        raise ValueError(
            "Fit-polynomial illumination mask must match the image shape; got "
            f"mask {mask_array.shape!r} for image {image.shape!r}."
        )
    gram, rhs = _fit_polynomial_normal_equations_numba(
        image,
        mask_array,
        mask is not None,
    )
    coeffs = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    return _evaluate_polynomial_surface_numba(
        image.shape[0],
        image.shape[1],
        np.ascontiguousarray(coeffs, dtype=np.float64),
    )


@njit(cache=True)
def _fit_polynomial_normal_equations_numba(
    pixel_data: np.ndarray,
    mask: np.ndarray,
    has_mask: bool,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = pixel_data.shape
    gram = np.zeros((6, 6), dtype=np.float64)
    rhs = np.zeros(6, dtype=np.float64)
    features = np.empty(6, dtype=np.float64)
    for row in range(height):
        y_value = row / height - 0.5
        y2 = y_value * y_value
        for col in range(width):
            if has_mask and not mask[row, col]:
                continue
            x_value = col / width - 0.5
            features[0] = x_value * x_value
            features[1] = y2
            features[2] = x_value * y_value
            features[3] = x_value
            features[4] = y_value
            features[5] = 1.0
            value = pixel_data[row, col]
            for i in range(6):
                rhs[i] += features[i] * value
                for j in range(6):
                    gram[i, j] += features[i] * features[j]
    return gram, rhs


@njit(cache=True)
def _evaluate_polynomial_surface_numba(
    height: int,
    width: int,
    coeffs: np.ndarray,
) -> np.ndarray:
    output = np.empty((height, width), dtype=np.float64)
    for row in range(height):
        y_value = row / height - 0.5
        y2 = y_value * y_value
        for col in range(width):
            x_value = col / width - 0.5
            output[row, col] = (
                coeffs[0] * x_value * x_value
                + coeffs[1] * y2
                + coeffs[2] * x_value * y_value
                + coeffs[3] * x_value
                + coeffs[4] * y_value
                + coeffs[5]
            )
    return output


def _blockwise_background_minimum(
    pixel_data: np.ndarray,
    mask: np.ndarray | None,
    block_size: int,
    morphology: MorphologyBackendStrategy,
) -> np.ndarray:
    from scipy.ndimage import minimum

    labels, indexes = morphology.block_labels(pixel_data.shape[:2], block_size)
    labels = labels.copy()
    if mask is not None:
        labels[~mask] = -1

    valid = labels != -1
    result = np.zeros(pixel_data.shape, dtype=pixel_data.dtype)
    if not np.any(valid):
        return result

    if pixel_data.ndim == 2:
        minima = morphology.fix_labeled_result(minimum(pixel_data, labels, indexes))
        result[valid] = minima[labels[valid]]
        return result

    for channel in range(pixel_data.shape[2]):
        minima = morphology.fix_labeled_result(
            minimum(pixel_data[:, :, channel], labels, indexes)
        )
        result[valid, channel] = minima[labels[valid]]
    return result


def _preprocess_for_averaging(
    pixel_data: np.ndarray,
    mask: np.ndarray | None,
    intensity_choice: IntensityChoice,
    smoothing_method: SmoothingMethod,
    block_size: int,
    morphology: MorphologyBackendStrategy,
) -> np.ndarray:
    """Create a version of the image appropriate for averaging."""
    if (
        intensity_choice == IntensityChoice.REGULAR
        or smoothing_method == SmoothingMethod.SPLINES
    ):
        result = pixel_data.copy()
        if mask is not None:
            result[~mask] = 0
        return result
    else:  # BACKGROUND
        return _blockwise_background_minimum(
            pixel_data,
            mask,
            block_size,
            morphology,
        )


def _calculation_scope_uses_all_images(calculation_scope: CalculationScope) -> bool:
    return calculation_scope in {
        CalculationScope.ALL_FIRST_CYCLE,
        CalculationScope.ALL_ACROSS_CYCLES,
    }


def _is_multi_image_stack(
    pixel_data: np.ndarray,
    calculation_scope: CalculationScope | None = None,
) -> bool:
    del calculation_scope
    return pixel_data.ndim >= 3


def _spatial_image_shape(
    pixel_data: np.ndarray,
    calculation_scope: CalculationScope,
) -> tuple[int, ...]:
    if (
        _calculation_scope_uses_all_images(calculation_scope)
        and _is_multi_image_stack(pixel_data, calculation_scope)
    ):
        return tuple(pixel_data.shape[1:])
    return tuple(pixel_data.shape)


def _illumination_average_image(
    pixel_data: np.ndarray,
    mask: np.ndarray | None,
    intensity_choice: IntensityChoice,
    smoothing_method: SmoothingMethod,
    block_size: int,
    morphology: MorphologyBackendStrategy,
    calculation_scope: CalculationScope,
) -> np.ndarray:
    if not (
        _calculation_scope_uses_all_images(calculation_scope)
        and _is_multi_image_stack(pixel_data, calculation_scope)
    ):
        return _preprocess_for_averaging(
            pixel_data,
            mask,
            intensity_choice,
            smoothing_method,
            block_size,
            morphology,
        )

    if pixel_data.shape[0] == 0:
        raise ValueError("All-image illumination calculation requires at least one image.")

    averaged_inputs = [
        _preprocess_for_averaging(
            np.asarray(slice_data),
            _mask_for_stack_slice(mask, slice_index),
            intensity_choice,
            smoothing_method,
            block_size,
            morphology,
        )
        for slice_index, slice_data in enumerate(pixel_data)
    ]
    return np.mean(np.stack(averaged_inputs, axis=0), axis=0)


def _normalized_illumination_mask(
    mask: object | None,
    pixel_data: np.ndarray,
) -> np.ndarray | None:
    """Return a mask aligned with the illumination input data."""
    if mask is None:
        return None
    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.shape == pixel_data.shape:
        return mask_array
    if mask_array.shape == pixel_data.shape[-mask_array.ndim :]:
        return mask_array
    if pixel_data.ndim == 2 and mask_array.ndim == 3 and mask_array.shape[0] == 1:
        return mask_array[0]
    return mask_array


def _mask_for_stack_slice(
    mask: np.ndarray | None,
    slice_index: int,
) -> np.ndarray | None:
    if mask is None:
        return None
    if mask.ndim >= 3 and slice_index < mask.shape[0]:
        return np.asarray(mask[slice_index], dtype=bool)
    return mask


def _output_mask_for_illumination(
    mask: np.ndarray | None,
    illumination: np.ndarray,
) -> np.ndarray | None:
    if mask is None:
        return None
    if mask.shape == illumination.shape:
        return mask
    if illumination.ndim == 2 and mask.ndim >= 3:
        return np.any(mask, axis=0)
    return mask


def _apply_dilation(
    pixel_data: np.ndarray,
    mask: np.ndarray | None,
    dilate: bool,
    dilation_radius: int,
) -> np.ndarray:
    """Apply dilation using Gaussian convolution."""
    if not dilate:
        return pixel_data

    result = _masked_gaussian_filter(pixel_data, mask, dilation_radius)
    if mask is not None:
        result[~mask] = 0
    return result


def _apply_scaling(
    pixel_data: np.ndarray,
    mask: np.ndarray | None,
    rescale_option: RescaleOption,
) -> np.ndarray:
    """Rescale the illumination function."""
    if rescale_option == RescaleOption.NO:
        return pixel_data
    
    if mask is not None:
        sorted_data = pixel_data[(pixel_data > 0) & mask]
    else:
        sorted_data = pixel_data[pixel_data > 0]
    
    if sorted_data.size == 0:
        return pixel_data
    
    sorted_data = np.sort(sorted_data)
    
    if rescale_option == RescaleOption.YES:
        idx = int(len(sorted_data) * ROBUST_FACTOR)
        robust_minimum = sorted_data[idx]
        result = pixel_data.copy()
        result[result < robust_minimum] = robust_minimum
    else:  # MEDIAN
        idx = len(sorted_data) // 2
        robust_minimum = sorted_data[idx]
        result = pixel_data.copy()
    
    if robust_minimum == 0:
        return result
    
    return result / robust_minimum


def _calculate_illumination_from_pixels(
    pixel_data: np.ndarray,
    *,
    mask: np.ndarray | None,
    intensity_choice: IntensityChoice,
    dilate_objects: bool,
    object_dilation_radius: int,
    block_size: int,
    rescale_option: RescaleOption,
    smoothing_method: SmoothingMethod,
    filter_size_method: FilterSizeMethod,
    object_width: int,
    manual_filter_size: int,
    automatic_splines: bool,
    spline_bg_mode: SplineBgMode,
    spline_points: int,
    spline_threshold: float,
    spline_rescale: float,
    spline_max_iterations: int,
    spline_convergence: float,
    calculation_scope: CalculationScope,
    morphology: MorphologyBackendStrategy,
    convex_hull_backend_provider: CellProfilerBackendProvider | None,
    rank_median_backend_provider: CellProfilerBackendProvider | None,
    slice_index: int = 0,
) -> tuple[np.ndarray, IlluminationStats]:
    filter_size = SmoothingFilterSizeStrategy.for_method(
        filter_size_method,
    ).calculate(
        SmoothingFilterSizeRequest(
            image_shape=_spatial_image_shape(pixel_data, calculation_scope),
            object_width=object_width,
            manual_filter_size=manual_filter_size,
        )
    )

    avg_image = _illumination_average_image(
        pixel_data,
        mask,
        intensity_choice,
        smoothing_method,
        block_size,
        morphology,
        calculation_scope,
    )
    dilated_image = _apply_dilation(avg_image, mask, dilate_objects, object_dilation_radius)
    smoothing_request = SmoothingPlaneRequest(
        pixel_data=dilated_image,
        mask=mask,
        smoothing_method=smoothing_method,
        filter_size=filter_size,
        spline_bg_mode=spline_bg_mode,
        spline_points=spline_points,
        spline_threshold=spline_threshold,
        spline_rescale=spline_rescale,
        spline_max_iterations=spline_max_iterations,
        spline_convergence=spline_convergence,
        automatic_splines=automatic_splines,
        morphology=morphology,
        convex_hull_backend_provider=convex_hull_backend_provider,
        rank_median_backend_provider=rank_median_backend_provider,
    )
    smoothed_image = SmoothingPlaneStrategy.for_method(smoothing_method).smooth(
        smoothing_request
    )

    output_image = _apply_scaling(smoothed_image, mask, rescale_option).astype(np.float32)
    stats = IlluminationStats(
        slice_index=slice_index,
        min_value=float(np.min(output_image)),
        max_value=float(np.max(output_image)),
        mean_value=float(np.mean(output_image)),
        calculation_type=intensity_choice.value,
        smoothing_method=smoothing_method.value
    )
    return output_image, stats


@numpy(contract=ProcessingContract.FLEXIBLE)
@special_outputs(("illumination_stats", csv_materializer(
    fields=["slice_index", "min_value", "max_value", "mean_value", "calculation_type", "smoothing_method"],
    analysis_type="illumination_correction"
)))
def correct_illumination_calculate(
    image: np.ndarray,
    intensity_choice: IntensityChoice | str = IntensityChoice.REGULAR,
    dilate_objects: bool = False,
    object_dilation_radius: int = 1,
    block_size: int = 60,
    rescale_option: RescaleOption | str = RescaleOption.YES,
    smoothing_method: SmoothingMethod | str = SmoothingMethod.FIT_POLYNOMIAL,
    filter_size_method: FilterSizeMethod | str = FilterSizeMethod.AUTOMATIC,
    object_width: int = 10,
    manual_filter_size: int = 10,
    automatic_splines: bool = True,
    spline_bg_mode: SplineBgMode | str = SplineBgMode.AUTO,
    spline_points: int = 5,
    spline_threshold: float = 2.0,
    spline_rescale: float = 2.0,
    spline_max_iterations: int = 40,
    spline_convergence: float = 0.001,
    calculation_scope: CalculationScope | str = CalculationScope.EACH,
    convex_hull_backend_provider: CellProfilerBackendProvider | None = None,
    rank_median_backend_provider: CellProfilerBackendProvider | None = None,
) -> tuple[np.ndarray, IlluminationStats]:
    """
    Calculate an illumination correction function.
    
    This function calculates an illumination function that can be used to correct
    uneven illumination/lighting/shading in images.
    
    Args:
        image: Input image (H, W)
        intensity_choice: Method for calculating illumination function (REGULAR or BACKGROUND)
        dilate_objects: Whether to dilate objects in the averaged image
        object_dilation_radius: Radius for object dilation
        block_size: Block size for background method
        rescale_option: How to rescale the illumination function
        smoothing_method: Method for smoothing the illumination function
        filter_size_method: How to calculate smoothing filter size
        object_width: Approximate object diameter for filter size calculation
        manual_filter_size: Manual smoothing filter size
        automatic_splines: Whether to automatically calculate spline parameters
        spline_bg_mode: Background mode for spline fitting
        spline_points: Number of spline control points
        spline_threshold: Background threshold for splines
        spline_rescale: Image resampling factor for splines
        spline_max_iterations: Maximum iterations for spline fitting
        spline_convergence: Convergence criterion for splines
        calculation_scope: Calculate each image independently or from all images
        convex_hull_backend_provider: Optional OpenHCS CellProfiler illumination
            backend provider. ``None`` uses the registered default provider.
        rank_median_backend_provider: Optional OpenHCS CellProfiler rank-median
            smoothing backend provider. ``None`` uses the registered default
            provider.
    
    Returns:
        Tuple of (illumination_function, stats)
    """
    intensity_choice = _coerce_function_enum(IntensityChoice, intensity_choice)
    rescale_option = _coerce_function_enum(RescaleOption, rescale_option)
    smoothing_method = _coerce_function_enum(SmoothingMethod, smoothing_method)
    filter_size_method = _coerce_function_enum(FilterSizeMethod, filter_size_method)
    spline_bg_mode = _coerce_function_enum(SplineBgMode, spline_bg_mode)
    calculation_scope = _coerce_function_enum(CalculationScope, calculation_scope)
    from openhcs.processing.backends.cellprofiler.morphology import (
        MorphologyBackendStrategy,
    )

    morphology = MorphologyBackendStrategy.for_callable(
        correct_illumination_calculate,
    )

    pixel_data = np.asarray(image_payload_data(image))
    mask = _normalized_illumination_mask(image_payload_mask(image), pixel_data)
    metadata = image_payload_metadata(image)
    common_kwargs = {
        "intensity_choice": intensity_choice,
        "dilate_objects": dilate_objects,
        "object_dilation_radius": object_dilation_radius,
        "block_size": block_size,
        "rescale_option": rescale_option,
        "smoothing_method": smoothing_method,
        "filter_size_method": filter_size_method,
        "object_width": object_width,
        "manual_filter_size": manual_filter_size,
        "automatic_splines": automatic_splines,
        "spline_bg_mode": spline_bg_mode,
        "spline_points": spline_points,
        "spline_threshold": spline_threshold,
        "spline_rescale": spline_rescale,
        "spline_max_iterations": spline_max_iterations,
        "spline_convergence": spline_convergence,
        "morphology": morphology,
        "convex_hull_backend_provider": convex_hull_backend_provider,
        "rank_median_backend_provider": rank_median_backend_provider,
    }

    if (
        _is_multi_image_stack(pixel_data, calculation_scope)
        and not _calculation_scope_uses_all_images(calculation_scope)
    ):
        slice_results = [
            _calculate_illumination_from_pixels(
                np.asarray(slice_data),
                mask=_mask_for_stack_slice(mask, slice_index),
                calculation_scope=CalculationScope.EACH,
                slice_index=slice_index,
                **common_kwargs,
            )
            for slice_index, slice_data in enumerate(pixel_data)
        ]
        illumination_stack = np.stack(
            [result[0] for result in slice_results],
            axis=0,
        ).astype(np.float32)
        return (
            image_payload_with_context(
                illumination_stack,
                mask=_output_mask_for_illumination(mask, illumination_stack),
                metadata=metadata,
            ),
            [result[1] for result in slice_results],
        )

    illumination, stats = _calculate_illumination_from_pixels(
        pixel_data,
        mask=mask,
        calculation_scope=calculation_scope,
        **common_kwargs,
    )
    return (
        image_payload_with_context(
            illumination,
            mask=_output_mask_for_illumination(mask, illumination),
            metadata=metadata,
        ),
        stats,
    )
