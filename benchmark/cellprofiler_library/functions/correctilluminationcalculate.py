"""
Converted from CellProfiler: CorrectIlluminationCalculate
Calculates an illumination correction function to correct uneven illumination/lighting/shading.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

import numpy as np
from metaclass_registry import AutoRegisterMeta

from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
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

    __registry_key__ = "method"
    __skip_if_no_key__ = True
    method: ClassVar[FilterSizeMethod | None] = None

    @classmethod
    def for_method(
        cls,
        method: FilterSizeMethod,
    ) -> "SmoothingFilterSizeStrategy":
        return cls.__registry__[method]()

    @abstractmethod
    def calculate(self, request: SmoothingFilterSizeRequest) -> float:
        """Return the smoothing filter size."""


class ManualSmoothingFilterSizeStrategy(SmoothingFilterSizeStrategy):
    method = FilterSizeMethod.MANUALLY

    def calculate(self, request: SmoothingFilterSizeRequest) -> float:
        return float(request.manual_filter_size)


class ObjectWidthSmoothingFilterSizeStrategy(SmoothingFilterSizeStrategy):
    method = FilterSizeMethod.OBJECT_SIZE

    def calculate(self, request: SmoothingFilterSizeRequest) -> float:
        return request.object_width * 2.35 / 3.5


class AutomaticSmoothingFilterSizeStrategy(SmoothingFilterSizeStrategy):
    method = FilterSizeMethod.AUTOMATIC

    def calculate(self, request: SmoothingFilterSizeRequest) -> float:
        return min(30.0, float(max(request.image_shape)) / 40.0)


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

    @property
    def sigma(self) -> float:
        return self.filter_size / 2.35


class SmoothingPlaneStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal smoothing implementation for one closed CellProfiler mode."""

    __registry_key__ = "method"
    __skip_if_no_key__ = True
    method: ClassVar[SmoothingMethod | None] = None

    @classmethod
    def for_method(cls, method: SmoothingMethod) -> "SmoothingPlaneStrategy":
        return cls.__registry__[method]()

    @abstractmethod
    def smooth(self, request: SmoothingPlaneRequest) -> np.ndarray:
        """Return the smoothed illumination plane."""


class NoSmoothingPlaneStrategy(SmoothingPlaneStrategy):
    method = SmoothingMethod.NONE

    def smooth(self, request: SmoothingPlaneRequest) -> np.ndarray:
        return request.pixel_data


class FitPolynomialSmoothingPlaneStrategy(SmoothingPlaneStrategy):
    method = SmoothingMethod.FIT_POLYNOMIAL

    def smooth(self, request: SmoothingPlaneRequest) -> np.ndarray:
        pixel_data = request.pixel_data
        h, w = pixel_data.shape
        y, x = np.mgrid[0:h, 0:w].astype(float)
        y = y / h - 0.5
        x = x / w - 0.5
        valid = (
            request.mask.flatten()
            if request.mask is not None
            else np.ones(h * w, dtype=bool)
        )
        design = np.column_stack(
            [
                (x**2).flatten()[valid],
                (y**2).flatten()[valid],
                (x * y).flatten()[valid],
                x.flatten()[valid],
                y.flatten()[valid],
                np.ones(valid.sum()),
            ]
        )
        coeffs, _, _, _ = np.linalg.lstsq(
            design,
            pixel_data.flatten()[valid],
            rcond=None,
        )
        full_design = np.column_stack(
            [
                (x**2).flatten(),
                (y**2).flatten(),
                (x * y).flatten(),
                x.flatten(),
                y.flatten(),
                np.ones(h * w),
            ]
        )
        return (full_design @ coeffs).reshape(h, w)


class GaussianFilterSmoothingPlaneStrategy(SmoothingPlaneStrategy):
    method = SmoothingMethod.GAUSSIAN_FILTER

    def smooth(self, request: SmoothingPlaneRequest) -> np.ndarray:
        return _masked_gaussian_filter(
            request.pixel_data,
            request.mask,
            request.sigma,
        )


class MedianFilterSmoothingPlaneStrategy(SmoothingPlaneStrategy):
    method = SmoothingMethod.MEDIAN_FILTER

    def smooth(self, request: SmoothingPlaneRequest) -> np.ndarray:
        from skimage.filters import median
        from skimage.morphology import disk

        filter_sigma = max(1, int(request.sigma + 0.5))
        scaled = (request.pixel_data * 65535).astype(np.uint16)
        if request.mask is not None:
            scaled = scaled * request.mask.astype(np.uint16)
        result = median(scaled, disk(filter_sigma))
        return result.astype(np.float32) / 65535.0


class AverageSmoothingPlaneStrategy(SmoothingPlaneStrategy):
    method = SmoothingMethod.TO_AVERAGE

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

    def smooth(self, request: SmoothingPlaneRequest) -> np.ndarray:
        from scipy.ndimage import grey_dilation, grey_erosion, maximum_filter

        eroded = grey_erosion(request.pixel_data, size=3)
        hull_approx = maximum_filter(eroded, size=int(request.filter_size))
        return grey_dilation(hull_approx, size=3)


class SplinesSmoothingPlaneStrategy(SmoothingPlaneStrategy):
    method = SmoothingMethod.SPLINES

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


def _preprocess_for_averaging(
    pixel_data: np.ndarray,
    mask: np.ndarray | None,
    intensity_choice: IntensityChoice,
    block_size: int,
) -> np.ndarray:
    """Create a version of the image appropriate for averaging."""
    if intensity_choice == IntensityChoice.REGULAR:
        result = pixel_data.copy()
        if mask is not None:
            result[~mask] = 0
        return result
    else:  # BACKGROUND
        from scipy.ndimage import minimum_filter
        # Find minimum in blocks
        result = minimum_filter(pixel_data, size=block_size)
        if mask is not None:
            result[~mask] = 0
        return result


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


@numpy(contract=ProcessingContract.PURE_2D)
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
    
    Returns:
        Tuple of (illumination_function, stats)
    """
    intensity_choice = _coerce_function_enum(IntensityChoice, intensity_choice)
    rescale_option = _coerce_function_enum(RescaleOption, rescale_option)
    smoothing_method = _coerce_function_enum(SmoothingMethod, smoothing_method)
    filter_size_method = _coerce_function_enum(FilterSizeMethod, filter_size_method)
    spline_bg_mode = _coerce_function_enum(SplineBgMode, spline_bg_mode)

    mask: np.ndarray | None = None

    filter_size = SmoothingFilterSizeStrategy.for_method(
        filter_size_method,
    ).calculate(
        SmoothingFilterSizeRequest(
            image_shape=image.shape,
            object_width=object_width,
            manual_filter_size=manual_filter_size,
        )
    )

    avg_image = _preprocess_for_averaging(image, mask, intensity_choice, block_size)
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
    )
    smoothed_image = SmoothingPlaneStrategy.for_method(smoothing_method).smooth(
        smoothing_request
    )

    output_image = _apply_scaling(smoothed_image, mask, rescale_option)
    
    # Ensure output is float32
    output_image = output_image.astype(np.float32)
    
    # Calculate statistics
    stats = IlluminationStats(
        slice_index=0,
        min_value=float(np.min(output_image)),
        max_value=float(np.max(output_image)),
        mean_value=float(np.mean(output_image)),
        calculation_type=intensity_choice.value,
        smoothing_method=smoothing_method.value
    )
    
    return output_image, stats
