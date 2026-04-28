"""
Converted from CellProfiler: CorrectIlluminationCalculate
Calculates an illumination correction function to correct uneven illumination/lighting/shading.
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.processing.materialization import csv_materializer


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


def _calculate_smoothing_filter_size(
    image_shape: tuple,
    filter_size_method: FilterSizeMethod,
    object_width: int,
    manual_filter_size: int
) -> float:
    """Calculate the smoothing filter size based on settings and image size."""
    if filter_size_method == FilterSizeMethod.MANUALLY:
        return float(manual_filter_size)
    elif filter_size_method == FilterSizeMethod.OBJECT_SIZE:
        return object_width * 2.35 / 3.5
    else:  # AUTOMATIC
        return min(30.0, float(max(image_shape)) / 40.0)


def _preprocess_for_averaging(
    pixel_data: np.ndarray,
    mask: np.ndarray,
    intensity_choice: IntensityChoice,
    block_size: int
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
    mask: np.ndarray,
    dilate: bool,
    dilation_radius: int
) -> np.ndarray:
    """Apply dilation using Gaussian convolution."""
    if not dilate:
        return pixel_data
    
    from scipy.ndimage import gaussian_filter
    
    sigma = dilation_radius
    if mask is not None:
        # Smooth with mask handling
        masked_data = pixel_data.copy()
        masked_data[~mask] = 0
        smoothed = gaussian_filter(masked_data, sigma, mode='constant', cval=0)
        mask_smoothed = gaussian_filter(mask.astype(float), sigma, mode='constant', cval=0)
        mask_smoothed = np.maximum(mask_smoothed, 1e-10)
        result = smoothed / mask_smoothed
        result[~mask] = 0
        return result
    else:
        return gaussian_filter(pixel_data, sigma, mode='constant', cval=0)


def _smooth_plane(
    pixel_data: np.ndarray,
    mask: np.ndarray,
    smoothing_method: SmoothingMethod,
    filter_size: float,
    spline_bg_mode: SplineBgMode,
    spline_points: int,
    spline_threshold: float,
    spline_rescale: float,
    spline_max_iterations: int,
    spline_convergence: float,
    automatic_splines: bool
) -> np.ndarray:
    """Smooth one 2D plane of an image."""
    from scipy.ndimage import gaussian_filter, median_filter
    
    sigma = filter_size / 2.35
    
    if smoothing_method == SmoothingMethod.NONE:
        return pixel_data
    
    elif smoothing_method == SmoothingMethod.FIT_POLYNOMIAL:
        # Fit polynomial: A*x^2 + B*y^2 + C*xy + D*x + E*y + F
        h, w = pixel_data.shape
        y, x = np.mgrid[0:h, 0:w].astype(float)
        y = y / h - 0.5
        x = x / w - 0.5
        
        if mask is not None:
            valid = mask.flatten()
        else:
            valid = np.ones(h * w, dtype=bool)
        
        # Build design matrix
        A = np.column_stack([
            (x**2).flatten()[valid],
            (y**2).flatten()[valid],
            (x*y).flatten()[valid],
            x.flatten()[valid],
            y.flatten()[valid],
            np.ones(valid.sum())
        ])
        b = pixel_data.flatten()[valid]
        
        # Solve least squares
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        # Reconstruct
        A_full = np.column_stack([
            (x**2).flatten(),
            (y**2).flatten(),
            (x*y).flatten(),
            x.flatten(),
            y.flatten(),
            np.ones(h * w)
        ])
        result = (A_full @ coeffs).reshape(h, w)
        return result
    
    elif smoothing_method == SmoothingMethod.GAUSSIAN_FILTER:
        if mask is not None:
            masked_data = pixel_data.copy()
            masked_data[~mask] = 0
            smoothed = gaussian_filter(masked_data, sigma, mode='constant', cval=0)
            mask_smoothed = gaussian_filter(mask.astype(float), sigma, mode='constant', cval=0)
            mask_smoothed = np.maximum(mask_smoothed, 1e-10)
            result = smoothed / mask_smoothed
            return result
        else:
            return gaussian_filter(pixel_data, sigma, mode='constant', cval=0)
    
    elif smoothing_method == SmoothingMethod.MEDIAN_FILTER:
        from skimage.filters import median
        from skimage.morphology import disk
        
        filter_sigma = max(1, int(sigma + 0.5))
        selem = disk(filter_sigma)
        # Scale to uint16 for median filter
        scaled = (pixel_data * 65535).astype(np.uint16)
        if mask is not None:
            scaled = scaled * mask.astype(np.uint16)
        result = median(scaled, selem)
        return result.astype(np.float32) / 65535.0
    
    elif smoothing_method == SmoothingMethod.TO_AVERAGE:
        if mask is not None:
            mean_val = np.mean(pixel_data[mask])
        else:
            mean_val = np.mean(pixel_data)
        return np.full(pixel_data.shape, mean_val, dtype=pixel_data.dtype)
    
    elif smoothing_method == SmoothingMethod.CONVEX_HULL:
        # Simplified convex hull transform using morphological operations
        from scipy.ndimage import grey_erosion, grey_dilation
        
        eroded = grey_erosion(pixel_data, size=3)
        # Simple approximation: use maximum filter as proxy for convex hull
        from scipy.ndimage import maximum_filter
        hull_approx = maximum_filter(eroded, size=int(filter_size))
        dilated = grey_dilation(hull_approx, size=3)
        return dilated
    
    elif smoothing_method == SmoothingMethod.SPLINES:
        # Simplified spline background estimation
        from scipy.interpolate import RectBivariateSpline
        
        h, w = pixel_data.shape
        
        if automatic_splines:
            shortest_side = min(h, w)
            scale = max(1, shortest_side // 200)
            n_points = 5
        else:
            scale = int(spline_rescale)
            n_points = spline_points
        
        # Downsample
        downsampled = pixel_data[::scale, ::scale]
        dh, dw = downsampled.shape
        
        # Create grid points
        y_points = np.linspace(0, dh-1, n_points)
        x_points = np.linspace(0, dw-1, n_points)
        
        # Sample values at grid points
        yi = np.round(y_points).astype(int)
        xi = np.round(x_points).astype(int)
        yi = np.clip(yi, 0, dh-1)
        xi = np.clip(xi, 0, dw-1)
        
        z_values = downsampled[np.ix_(yi, xi)]
        
        # Fit spline
        spline = RectBivariateSpline(y_points, x_points, z_values, kx=3, ky=3)
        
        # Evaluate on full grid
        y_full = np.linspace(0, dh-1, h)
        x_full = np.linspace(0, dw-1, w)
        result = spline(y_full, x_full)
        
        # Normalize to preserve mean
        if mask is not None:
            mean_intensity = np.mean(result[mask])
            result[mask] -= mean_intensity
        else:
            mean_intensity = np.mean(result)
            result -= mean_intensity
        
        return result
    
    return pixel_data


def _apply_scaling(
    pixel_data: np.ndarray,
    mask: np.ndarray,
    rescale_option: RescaleOption
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
    intensity_choice: IntensityChoice = IntensityChoice.REGULAR,
    dilate_objects: bool = False,
    object_dilation_radius: int = 1,
    block_size: int = 60,
    rescale_option: RescaleOption = RescaleOption.YES,
    smoothing_method: SmoothingMethod = SmoothingMethod.FIT_POLYNOMIAL,
    filter_size_method: FilterSizeMethod = FilterSizeMethod.AUTOMATIC,
    object_width: int = 10,
    manual_filter_size: int = 10,
    automatic_splines: bool = True,
    spline_bg_mode: SplineBgMode = SplineBgMode.AUTO,
    spline_points: int = 5,
    spline_threshold: float = 2.0,
    spline_rescale: float = 2.0,
    spline_max_iterations: int = 40,
    spline_convergence: float = 0.001,
) -> Tuple[np.ndarray, IlluminationStats]:
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
    # Assume no mask for single image processing
    mask = None
    
    # Calculate filter size
    filter_size = _calculate_smoothing_filter_size(
        image.shape, filter_size_method, object_width, manual_filter_size
    )
    
    # Preprocess for averaging
    avg_image = _preprocess_for_averaging(image, mask, intensity_choice, block_size)
    
    # Apply dilation
    dilated_image = _apply_dilation(avg_image, mask, dilate_objects, object_dilation_radius)
    
    # Apply smoothing
    smoothed_image = _smooth_plane(
        dilated_image, mask, smoothing_method, filter_size,
        spline_bg_mode, spline_points, spline_threshold,
        spline_rescale, spline_max_iterations, spline_convergence,
        automatic_splines
    )
    
    # Apply scaling
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