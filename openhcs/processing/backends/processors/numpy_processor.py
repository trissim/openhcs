"""
NumPy Image Processor Implementation

This module implements the ImageProcessorInterface using NumPy as the backend.
It serves as the reference implementation and works on CPU.

Doctrinal Clauses:
- Clause 3 — Declarative Primacy: All functions are pure and stateless
- Clause 88 — No Inferred Capabilities: Explicit NumPy dependency
- Clause 106-A — Declared Memory Types: All methods specify NumPy arrays
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from enum import Enum
from typing import Annotated, Any, List, Optional, Tuple

from metaclass_registry import AutoRegisterMeta
import numpy as np
from skimage import exposure, filters
from skimage import morphology as morph
from skimage import transform as trans

# Use direct import from core memory decorators to avoid circular imports
from openhcs.core.memory import numpy as numpy_func
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.runtime_array_values import RuntimeArrayPayload
from openhcs.processing.backends.processors.method_axes import (
    OrthogonalProjectionPlane,
    SpatialBinMethod,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract

logger = logging.getLogger(__name__)


PercentileLowerEndpointInput = Annotated[
    float,
    "Percentile used as the input-range lower endpoint (0 to 100).",
]
PercentileUpperEndpointInput = Annotated[
    float,
    "Percentile used as the input-range upper endpoint (0 to 100).",
]
NormalizationTargetMinimumInput = Annotated[
    float,
    "Output intensity assigned to the normalized range's lower endpoint.",
]
NormalizationTargetMaximumInput = Annotated[
    float,
    "Output intensity assigned to the normalized range's upper endpoint.",
]
class NumpySpatialBinStrategy(
    EnumKeyedStrategyMixin[SpatialBinMethod], metaclass=AutoRegisterMeta
):
    __enum_member_attr__ = "method"

    @abstractmethod
    def apply(self, array: np.ndarray, axis: tuple[int, ...]) -> np.ndarray:
        raise NotImplementedError


class NumpyMeanSpatialBinStrategy(NumpySpatialBinStrategy):
    method = SpatialBinMethod.MEAN

    def apply(self, array: np.ndarray, axis: tuple[int, ...]) -> np.ndarray:
        return np.mean(array, axis=axis)


class NumpySumSpatialBinStrategy(NumpySpatialBinStrategy):
    method = SpatialBinMethod.SUM

    def apply(self, array: np.ndarray, axis: tuple[int, ...]) -> np.ndarray:
        return np.sum(array, axis=axis)


class NumpyMaxSpatialBinStrategy(NumpySpatialBinStrategy):
    method = SpatialBinMethod.MAX

    def apply(self, array: np.ndarray, axis: tuple[int, ...]) -> np.ndarray:
        return np.max(array, axis=axis)


class NumpyMinSpatialBinStrategy(NumpySpatialBinStrategy):
    method = SpatialBinMethod.MIN

    def apply(self, array: np.ndarray, axis: tuple[int, ...]) -> np.ndarray:
        return np.min(array, axis=axis)


@numpy_func
def create_linear_weight_mask(
    height: int, width: int, margin_ratio: float = 0.1
) -> np.ndarray:
    """
    Create a 2D weight mask that linearly ramps from 0 at the edges to 1 in the center.

    Args:
        height: Height of the mask
        width: Width of the mask
        margin_ratio: Ratio of the margin to the image size

    Returns:
        2D weight mask of shape (height, width)
    """
    margin_y = int(np.floor(height * margin_ratio))
    margin_x = int(np.floor(width * margin_ratio))

    weight_y = np.ones(height, dtype=np.float32)
    if margin_y > 0:
        ramp_top = np.linspace(0, 1, margin_y, endpoint=False)
        ramp_bottom = np.linspace(1, 0, margin_y, endpoint=False)
        weight_y[:margin_y] = ramp_top
        weight_y[-margin_y:] = ramp_bottom

    weight_x = np.ones(width, dtype=np.float32)
    if margin_x > 0:
        ramp_left = np.linspace(0, 1, margin_x, endpoint=False)
        ramp_right = np.linspace(1, 0, margin_x, endpoint=False)
        weight_x[:margin_x] = ramp_left
        weight_x[-margin_x:] = ramp_right

    # Create 2D weight mask
    weight_mask = np.outer(weight_y, weight_x)

    return weight_mask


def _validate_3d_array(array: Any, name: str = "input") -> None:
    """
    Validate that the input is a 3D NumPy array.

    Args:
        array: Array to validate
        name: Name of the array for error messages

    Raises:
        TypeError: If the array is not a NumPy array
        ValueError: If the array is not 3D
    """
    if not isinstance(array, (np.ndarray, RuntimeArrayPayload)):
        raise TypeError(f"{name} must be a NumPy array, got {type(array)}")

    if array.ndim != 3:
        raise ValueError(f"{name} must be a 3D array, got {array.ndim}D")


@numpy_func
def sharpen(image: np.ndarray, radius: float = 1.0, amount: float = 1.0) -> np.ndarray:
    """
    Sharpen a 3D image using unsharp masking.

    This applies sharpening to each Z-slice independently.

    Args:
        radius: Radius of Gaussian blur
        amount: Sharpening strength

    Returns:
        Sharpened 3D NumPy array of shape (Z, Y, X)
    """
    _validate_3d_array(image)

    # Store original dtype
    dtype = image.dtype

    # Process each Z-slice independently
    result = np.zeros_like(image, dtype=np.float32)

    for z in range(image.shape[0]):
        # Convert to float for processing
        slice_float = image[z].astype(np.float32) / np.max(image[z])

        # Create blurred version for unsharp mask
        blurred = filters.gaussian(slice_float, sigma=radius)

        # Apply unsharp mask: original + amount * (original - blurred)
        sharpened = slice_float + amount * (slice_float - blurred)

        # Clip to valid range
        sharpened = np.clip(sharpened, 0, 1.0)

        # Scale back to original range
        sharpened = exposure.rescale_intensity(
            sharpened, in_range="image", out_range=(0, 65535)
        )
        result[z] = sharpened

    # Convert back to original dtype
    return result.astype(dtype)


@numpy_func
def percentile_normalize(
    image: np.ndarray,
    low_percentile: PercentileLowerEndpointInput = 1.0,
    high_percentile: PercentileUpperEndpointInput = 99.0,
    target_min: NormalizationTargetMinimumInput = 0.0,
    target_max: NormalizationTargetMaximumInput = 65535.0,
) -> np.ndarray:
    """
    Normalize each plane independently using percentile-based contrast stretching.

    The first transported array axis is the declared plane axis, which may be
    Z, channel, time, or another ``variable_components`` axis. Each plane gets
    its own percentile endpoints.

    Use when:
        Per-plane contrast must be made comparable for visualization or a
        downstream segmentation operation.
    Avoid when:
        Absolute intensity or intensity differences between planes are the
        measurement of interest; independent scaling removes those differences.
    Validate:
        Inspect representative raw/normalized pairs for clipping, amplified
        noise, and loss of dim structures before using the result downstream.

    Returns:
        Normalized 3D NumPy array of shape (N, Y, X).
    """
    _validate_3d_array(image)

    # Import shared utilities
    from .percentile_utils import resolve_target_range, slice_percentile_normalize_core

    # Auto-detect target range based on input dtype if not specified
    target_min, target_max = resolve_target_range(image.dtype, target_min, target_max)

    # Use shared core logic with NumPy-specific functions
    return slice_percentile_normalize_core(
        image=image,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
        target_min=target_min,
        target_max=target_max,
        percentile_func=np.percentile,
        clip_func=np.clip,
        ones_like_func=np.ones_like,
        zeros_like_func=lambda arr, dtype=None: np.zeros_like(
            arr, dtype=dtype or np.float32
        ),
    )


@numpy_func
def stack_percentile_normalize(
    stack: np.ndarray,
    low_percentile: PercentileLowerEndpointInput = 1.0,
    high_percentile: PercentileUpperEndpointInput = 99.0,
    target_min: NormalizationTargetMinimumInput = 0.0,
    target_max: NormalizationTargetMaximumInput = 65535.0,
) -> np.ndarray:
    """
    Normalize a stack using global percentile-based contrast stretching.

    One percentile pair is computed over every plane on the declared leading
    array axis, so relative intensities between unclipped pixels are preserved
    within this input stack. That axis may represent Z, channel, time, or another
    ``variable_components`` selection.

    Use when:
        A shared robust display or segmentation scale is needed across the
        planes assembled for one invocation.
    Avoid when:
        Independently fitted stacks must retain absolute intensity differences
        across wells, sites, or experimental conditions. Global percentile
        scaling still clips endpoint values and changes quantitative intensity.
    Validate:
        Compare raw and normalized plane histograms, the fraction clipped at
        each endpoint, and representative dim and bright structures.

    Returns:
        Normalized 3D NumPy array of shape (N, Y, X).
    """
    _validate_3d_array(stack)

    # Import shared utilities
    from .percentile_utils import resolve_target_range, percentile_normalize_core

    # Auto-detect target range based on input dtype if not specified
    target_min, target_max = resolve_target_range(stack.dtype, target_min, target_max)

    # Use shared core logic with NumPy-specific functions
    return percentile_normalize_core(
        stack=stack,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
        target_min=target_min,
        target_max=target_max,
        percentile_func=lambda arr, pct: np.percentile(arr, pct, axis=None),
        clip_func=np.clip,
        ones_like_func=np.ones_like,
    )


@numpy_func(contract=ProcessingContract.VOLUMETRIC_TO_SLICE)
def create_composite(
    stack: np.ndarray, weights: Optional[List[float]] = None
) -> np.ndarray:
    """
    Create a composite image from a 3D stack where each slice is a channel.

    Args:
        stack: 3D NumPy array of shape (N, Y, X) where N is number of channel slices
        weights: List of weights for each slice. If None, equal weights are used.

    Returns:
        Composite 2D NumPy array of shape (Y, X). OpenHCS contract execution
        restores the singleton runtime stack axis.
    """
    # Validate input is 3D array
    _validate_3d_array(stack)

    n_slices, height, width = stack.shape

    # Default weights if none provided
    if weights is None:
        # Equal weights forangeit so that it can only all slices
        weights = [1.0 / n_slices] * n_slices
    elif isinstance(weights, (list, tuple)):
        # Convert tuple to list if needed
        weights = list(weights)
        if len(weights) != n_slices:
            raise ValueError(
                f"Number of weights ({len(weights)}) must match number of slices ({n_slices})"
            )
    else:
        raise TypeError(
            f"weights must be a list of values or None, got {type(weights)}: {weights}"
        )

    # Normalize weights to sum to 1
    weight_sum = sum(weights)
    if weight_sum == 0:
        raise ValueError("Sum of weights cannot be zero")
    normalized_weights = [w / weight_sum for w in weights]

    # Convert weights to NumPy array for efficient computation
    # CRITICAL: Use float32 for weights to preserve fractional values, not stack.dtype
    weights_array = np.array(normalized_weights, dtype=np.float32)

    # Reshape weights for broadcasting: (N, 1, 1) to multiply with (N, Y, X)
    weights_array = weights_array.reshape(n_slices, 1, 1)

    # Create composite by weighted sum along the first axis
    # Convert stack to float32 for computation to avoid precision loss
    stack_float = stack.astype(np.float32)
    weighted_stack = stack_float * weights_array
    composite_slice = np.sum(weighted_stack, axis=0, keepdims=True)  # Keep as (1, Y, X)

    # Convert back to original dtype
    composite_slice = composite_slice.astype(stack.dtype)

    return composite_slice[0]


@numpy_func
def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a mask to a 3D image.

    This applies the mask to each Z-slice independently if mask is 2D,
    or applies the 3D mask directly if mask is 3D.

    Args:
        mask: 3D NumPy array of shape (Z, Y, X) or 2D NumPy array of shape (Y, X)

    Returns:
        Masked 3D NumPy array of shape (Z, Y, X)
    """
    _validate_3d_array(image)

    # Handle 2D mask (apply to each Z-slice)
    if isinstance(mask, np.ndarray) and mask.ndim == 2:
        if mask.shape != image.shape[1:]:
            raise ValueError(
                f"2D mask shape {mask.shape} doesn't match image slice shape {image.shape[1:]}"
            )

        # Apply 2D mask to each Z-slice
        result = np.zeros_like(image)
        for z in range(image.shape[0]):
            result[z] = image[z].astype(np.float32) * mask.astype(np.float32)

        return result.astype(image.dtype)

    # Handle 3D mask
    if isinstance(mask, np.ndarray) and mask.ndim == 3:
        if mask.shape != image.shape:
            raise ValueError(
                f"3D mask shape {mask.shape} doesn't match image shape {image.shape}"
            )

        # Apply 3D mask directly
        masked = image.astype(np.float32) * mask.astype(np.float32)
        return masked.astype(image.dtype)

    # If we get here, the mask is neither 2D nor 3D NumPy array
    raise TypeError(f"mask must be a 2D or 3D NumPy array, got {type(mask)}")


@numpy_func
def create_weight_mask(shape: Tuple[int, int], margin_ratio: float = 0.1) -> np.ndarray:
    """
    Create a weight mask for blending images.

    Args:
        shape: Shape of the mask (height, width)
        margin_ratio: Ratio of image size to use as margin

    Returns:
        2D NumPy weight mask of shape (Y, X)
    """
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise TypeError("shape must be a tuple of (height, width)")

    height, width = shape
    return create_linear_weight_mask(height, width, margin_ratio)


@numpy_func
def max_projection(stack: np.ndarray) -> np.ndarray:
    """
    Create a maximum intensity projection from a Z-stack.

    Returns:
        3D NumPy array of shape (1, Y, X)
    """
    _validate_3d_array(stack)

    # Create max projection
    projection_2d = np.max(stack, axis=0)
    return projection_2d.reshape(1, projection_2d.shape[0], projection_2d.shape[1])


@numpy_func
def mean_projection(stack: np.ndarray) -> np.ndarray:
    """
    Create a mean intensity projection from a Z-stack.

    Returns:
        3D NumPy array of shape (1, Y, X)
    """
    _validate_3d_array(stack)

    # Create mean projection
    projection_2d = np.mean(stack, axis=0).astype(stack.dtype)
    return projection_2d.reshape(1, projection_2d.shape[0], projection_2d.shape[1])


def min_projection(stack: np.ndarray) -> np.ndarray:
    """Create a minimum intensity projection from a Z-stack."""

    _validate_3d_array(stack)
    projection_2d = np.min(stack, axis=0)
    return projection_2d.reshape(1, *projection_2d.shape)


class NumpyStackProjectionMethod(Enum):
    """Stack reductions implemented by the NumPy projection dispatcher."""

    MAX = "max_projection"
    MEAN = "mean_projection"
    MIN = "min_projection"


class NumpyStackProjectionStrategy(
    EnumKeyedStrategyMixin[NumpyStackProjectionMethod],
    metaclass=AutoRegisterMeta,
):
    __enum_member_attr__ = "method"

    @abstractmethod
    def apply(self, stack: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class NumpyMaxStackProjectionStrategy(NumpyStackProjectionStrategy):
    method = NumpyStackProjectionMethod.MAX

    def apply(self, stack: np.ndarray) -> np.ndarray:
        return max_projection(stack)


class NumpyMeanStackProjectionStrategy(NumpyStackProjectionStrategy):
    method = NumpyStackProjectionMethod.MEAN

    def apply(self, stack: np.ndarray) -> np.ndarray:
        return mean_projection(stack)


class NumpyMinStackProjectionStrategy(NumpyStackProjectionStrategy):
    method = NumpyStackProjectionMethod.MIN

    def apply(self, stack: np.ndarray) -> np.ndarray:
        return min_projection(stack)


@numpy_func
def create_orthogonal_projections(
    stack: np.ndarray,
    projections: Tuple[OrthogonalProjectionPlane, ...] = (
        OrthogonalProjectionPlane.XY,
        OrthogonalProjectionPlane.XZ,
        OrthogonalProjectionPlane.YZ,
    ),
) -> dict:
    """
    Create orthogonal max projections from a Z-stack.

    Args:
        projections: Tuple of projection types to create. Options: "xy", "xz", "yz"

    Returns:
        Dict of 2D NumPy arrays, keyed by projection type:
        - "xy": (Y, X) - max along Z axis (top-down view)
        - "xz": (Z, X) - max along Y axis (side view)
        - "yz": (Z, Y) - max along X axis (side view)

    Invariants:
        - Pure function: same input → same output
        - No external dependencies beyond numpy
        - Returns data, never performs I/O
        - Testable in isolation
    """
    _validate_3d_array(stack)

    result = {}
    if OrthogonalProjectionPlane.XY in projections:
        result[OrthogonalProjectionPlane.XY.value] = stack.max(axis=0)
    if OrthogonalProjectionPlane.XZ in projections:
        result[OrthogonalProjectionPlane.XZ.value] = stack.max(axis=1)
    if OrthogonalProjectionPlane.YZ in projections:
        result[OrthogonalProjectionPlane.YZ.value] = stack.max(axis=2)
    return result


@numpy_func
def gaussian_blur(stack: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian blur to reduce noise in image stack.

    Args:
        sigma: Standard deviation for Gaussian kernel (higher = more blur)

    Returns:
        Blurred 3D NumPy array of shape (Z, Y, X)
    """
    _validate_3d_array(stack)

    # Apply Gaussian blur slice-by-slice
    blurred = np.zeros_like(stack, dtype=np.float64)
    for z in range(stack.shape[0]):
        blurred[z] = filters.gaussian(stack[z], sigma=sigma, preserve_range=True)

    return blurred.astype(stack.dtype)


@numpy_func
def spatial_bin_2d(
    stack: np.ndarray,
    bin_size: int = 2,
    method: SpatialBinMethod = SpatialBinMethod.MEAN,
) -> np.ndarray:
    """
    Apply 2D spatial binning to each slice in the stack.

    Reduces spatial resolution by combining neighboring pixels in 2D blocks.
    Each slice is processed independently.

    Args:
        bin_size: Size of the square binning kernel (e.g., 2 = 2x2 binning)

    Returns:
        Binned 3D NumPy array of shape (Z, Y//bin_size, X//bin_size)
    """
    _validate_3d_array(stack)

    if bin_size <= 0:
        raise ValueError("bin_size must be positive")

    z_slices, height, width = stack.shape

    # Calculate output dimensions
    new_height = height // bin_size
    new_width = width // bin_size

    if new_height == 0 or new_width == 0:
        raise ValueError(
            f"bin_size {bin_size} is too large for image dimensions {height}x{width}"
        )

    # Crop to make dimensions divisible by bin_size
    crop_height = new_height * bin_size
    crop_width = new_width * bin_size
    cropped_stack = stack[:, :crop_height, :crop_width]

    # Reshape for binning: (Z, new_height, bin_size, new_width, bin_size)
    reshaped = cropped_stack.reshape(
        z_slices, new_height, bin_size, new_width, bin_size
    )

    result = NumpySpatialBinStrategy.for_enum_member(method).apply(
        reshaped, axis=(2, 4)
    )

    return result.astype(stack.dtype)


@numpy_func
def spatial_bin_3d(
    stack: np.ndarray,
    bin_size: int = 2,
    method: SpatialBinMethod = SpatialBinMethod.MEAN,
) -> np.ndarray:
    """
    Apply 3D spatial binning to the entire stack.

    Reduces spatial resolution by combining neighboring voxels in 3D blocks.

    Args:
        bin_size: Size of the cubic binning kernel (e.g., 2 = 2x2x2 binning)

    Returns:
        Binned 3D NumPy array of shape (Z//bin_size, Y//bin_size, X//bin_size)
    """
    _validate_3d_array(stack)

    if bin_size <= 0:
        raise ValueError("bin_size must be positive")

    depth, height, width = stack.shape

    # Calculate output dimensions
    new_depth = depth // bin_size
    new_height = height // bin_size
    new_width = width // bin_size

    if new_depth == 0 or new_height == 0 or new_width == 0:
        raise ValueError(
            f"bin_size {bin_size} is too large for stack dimensions {depth}x{height}x{width}"
        )

    # Crop to make dimensions divisible by bin_size
    crop_depth = new_depth * bin_size
    crop_height = new_height * bin_size
    crop_width = new_width * bin_size
    cropped_stack = stack[:crop_depth, :crop_height, :crop_width]

    # Reshape for 3D binning: (new_depth, bin_size, new_height, bin_size, new_width, bin_size)
    reshaped = cropped_stack.reshape(
        new_depth, bin_size, new_height, bin_size, new_width, bin_size
    )

    result = NumpySpatialBinStrategy.for_enum_member(method).apply(
        reshaped, axis=(1, 3, 5)
    )

    return result.astype(stack.dtype)


@numpy_func
def stack_equalize_histogram(
    stack: np.ndarray,
    bins: int = 65536,
    range_min: float = 0.0,
    range_max: float = 65535.0,
) -> np.ndarray:
    """
    Apply histogram equalization to an entire stack.

    This ensures consistent contrast enhancement across all Z-slices by
    computing a global histogram across the entire stack.

    Args:
        bins: Number of bins for histogram computation
        range_min: Minimum value for histogram range
        range_max: Maximum value for histogram range

    Returns:
        Equalized 3D NumPy array of shape (Z, Y, X)
    """
    _validate_3d_array(stack)

    # Remember input dtype to preserve it
    input_dtype = stack.dtype

    # Flatten the entire stack to compute the global histogram
    flat_stack = stack.flatten()

    # Calculate the histogram and cumulative distribution function (CDF)
    hist, bin_edges = np.histogram(flat_stack, bins=bins, range=(range_min, range_max))
    cdf = hist.cumsum()

    # Normalize the CDF to the input dtype range
    # Avoid division by zero
    if cdf[-1] > 0:
        if np.issubdtype(input_dtype, np.integer):
            dtype_info = np.iinfo(input_dtype)
            cdf = dtype_info.max * cdf / cdf[-1]
        else:
            # For float dtypes, normalize to [0, 1]
            cdf = cdf / cdf[-1]

    # Use linear interpolation to map input values to equalized values
    equalized_stack = np.interp(stack.flatten(), bin_edges[:-1], cdf).reshape(
        stack.shape
    )

    # Convert back to input dtype
    if np.issubdtype(input_dtype, np.integer):
        dtype_info = np.iinfo(input_dtype)
        return np.clip(equalized_stack, dtype_info.min, dtype_info.max).astype(
            input_dtype
        )
    else:
        return equalized_stack.astype(input_dtype)


@numpy_func(contract=ProcessingContract.VOLUMETRIC_TO_SLICE)
def create_projection(
    stack: np.ndarray,
    method: NumpyStackProjectionMethod = NumpyStackProjectionMethod.MAX,
) -> np.ndarray:
    """
    Create a projection from a stack using the specified method.

    Args:
        method: Projection method owned by ``NumpyStackProjectionStrategy``.

    Returns:
        2D NumPy array of shape (Y, X)
    """
    stack_array = np.asarray(stack)
    _validate_3d_array(stack_array)

    return NumpyStackProjectionStrategy.for_enum_member(method).apply(stack_array)[0]


@numpy_func
def crop(
    input_image: np.ndarray,
    start_x: int = 0,
    start_y: int = 0,
    start_z: int = 0,
    width: int = 1,
    height: int = 1,
    depth: int = 1,
) -> np.ndarray:
    """
    Crop a given substack out of a given image stack.

    Equivalent to pyclesperanto.crop() but using NumPy operations.

    Parameters
    ----------
    input_image: np.ndarray
        Input 3D image to process of shape (Z, Y, X)
    start_x: int (= 0)
        Starting index coordinate x
    start_y: int (= 0)
        Starting index coordinate y
    start_z: int (= 0)
        Starting index coordinate z
    width: int (= 1)
        Width size of the region to crop
    height: int (= 1)
        Height size of the region to crop
    depth: int (= 1)
        Depth size of the region to crop

    Returns
    -------
    np.ndarray
        Cropped 3D array of shape (depth, height, width)
    """
    _validate_3d_array(input_image)

    # Validate crop parameters
    if width <= 0 or height <= 0 or depth <= 0:
        raise ValueError(
            f"Crop dimensions must be positive: width={width}, height={height}, depth={depth}"
        )

    if start_x < 0 or start_y < 0 or start_z < 0:
        raise ValueError(
            f"Start coordinates must be non-negative: start_x={start_x}, start_y={start_y}, start_z={start_z}"
        )

    # Get input dimensions
    input_depth, input_height, input_width = input_image.shape

    # Calculate end coordinates
    end_x = start_x + width
    end_y = start_y + height
    end_z = start_z + depth

    # Validate bounds
    if end_x > input_width or end_y > input_height or end_z > input_depth:
        raise ValueError(
            f"Crop region extends beyond image bounds. "
            f"Image shape: {input_image.shape}, "
            f"Crop region: ({start_z}:{end_z}, {start_y}:{end_y}, {start_x}:{end_x})"
        )

    # Perform the crop using NumPy slicing
    cropped = input_image[start_z:end_z, start_y:end_y, start_x:end_x]

    return cropped


@numpy_func
def tophat(
    image: np.ndarray,
    selem_radius: int = 50,
    downsample_factor: int = 4,
    downsample_anti_aliasing: bool = True,
    upsample_order: int = 0,
) -> np.ndarray:
    """
    Apply white top-hat background subtraction to each image plane.

    Each plane on the declared leading array axis is processed independently.
    The structuring-element radius sets the foreground/background size boundary
    in full-resolution pixels; downsampling approximates that operation faster.

    Use when:
        Bright foreground structures are smaller than the chosen radius and
        sit on uneven, slowly varying background.
    Avoid when:
        Desired structures approach or exceed the radius, background contains
        sharp spatial changes, or corrected pixels will be treated as
        unaltered quantitative intensity measurements.
    Validate:
        Inspect raw/corrected overlays across plate positions for halos, erased
        broad structures, residual background, and changed object intensities.

    Args:
        selem_radius: Approximate foreground/background size boundary in
            full-resolution pixels.
        downsample_factor: Spatial reduction factor used while estimating the
            background; larger values are faster but less spatially precise.
        downsample_anti_aliasing: Apply anti-aliasing before the reduced-scale
            background estimate.
        upsample_order: Interpolation order for restoring the estimated
            background (0=nearest, 1=linear, and so on).

    Returns:
        Background-subtracted 3D NumPy array of shape (N, Y, X).
    """
    _validate_3d_array(image)

    # Process each Z-slice independently
    result = np.zeros_like(image)

    for z in range(image.shape[0]):
        # Store original data type
        input_dtype = image[z].dtype

        # 1) Downsample
        image_small = trans.resize(
            image[z],
            (
                image[z].shape[0] // downsample_factor,
                image[z].shape[1] // downsample_factor,
            ),
            anti_aliasing=downsample_anti_aliasing,
            preserve_range=True,
        )

        # 2) Build structuring element for the smaller image
        selem_small = morph.disk(selem_radius // downsample_factor)

        # 3) White top-hat on the smaller image
        tophat_small = morph.white_tophat(image_small, selem_small)

        # 4) Upscale background to original size
        background_small = image_small - tophat_small
        background_large = trans.resize(
            background_small, image[z].shape, order=upsample_order, preserve_range=True
        )

        # 5) Subtract background and clip negative values
        slice_result = np.maximum(image[z] - background_large, 0)

        # 6) Convert back to original data type
        result[z] = slice_result.astype(input_dtype)

    return result
