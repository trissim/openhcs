"""
Converted from CellProfiler: FindMaxima
Original: FindMaxima.run

Isolates local peaks of high intensity from an image.
Returns an image with single pixels (or labeled regions) at each position
where a peak of intensity was found in the input image.
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.processing.materialization import csv_materializer


class ExcludeMode(Enum):
    THRESHOLD = "threshold"
    MASK = "mask"
    OBJECTS = "objects"


@dataclass
class MaximaResult:
    slice_index: int
    maxima_count: int
    min_distance_used: int
    threshold_used: float


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(("maxima_results", csv_materializer(
    fields=["slice_index", "maxima_count", "min_distance_used", "threshold_used"],
    analysis_type="maxima_detection"
)))
def find_maxima(
    image: np.ndarray,
    min_distance: int = 5,
    exclude_mode: ExcludeMode = ExcludeMode.THRESHOLD,
    min_intensity: float = 0.0,
    label_maxima: bool = True,
) -> Tuple[np.ndarray, MaximaResult]:
    """
    Find local maxima (intensity peaks) in an image.
    
    Args:
        image: Input grayscale image (H, W)
        min_distance: Minimum distance between accepted local maxima
        exclude_mode: Method for excluding background
            - THRESHOLD: Use min_intensity as threshold
            - MASK: Requires mask to be stacked in dim 0 (use FLEXIBLE contract variant)
            - OBJECTS: Requires labels to be stacked in dim 0 (use FLEXIBLE contract variant)
        min_intensity: Minimum pixel intensity to be considered as a peak
            (only used when exclude_mode is THRESHOLD)
        label_maxima: If True, assign unique labels to each maxima.
            If False, return binary image.
    
    Returns:
        Tuple of:
            - Output image with maxima (labeled or binary)
            - MaximaResult dataclass with detection statistics
    """
    from skimage.feature import peak_local_max
    import scipy.ndimage
    
    x_data = image.copy()
    th_abs = None
    
    if exclude_mode == ExcludeMode.THRESHOLD:
        th_abs = min_intensity if min_intensity > 0 else None
    # Note: MASK and OBJECTS modes require multi-input variant
    # For single-image processing, only THRESHOLD mode is supported
    
    # Find local maxima coordinates
    maxima_coords = peak_local_max(
        x_data,
        min_distance=min_distance,
        threshold_abs=th_abs,
    )
    
    # Create output image
    y_data = np.zeros(x_data.shape, dtype=np.float32)
    if len(maxima_coords) > 0:
        y_data[tuple(maxima_coords.T)] = 1.0
    
    # Optionally label each maximum with unique ID
    if label_maxima:
        y_data = scipy.ndimage.label(y_data > 0)[0].astype(np.float32)
    
    maxima_count = len(maxima_coords)
    
    result = MaximaResult(
        slice_index=0,
        maxima_count=maxima_count,
        min_distance_used=min_distance,
        threshold_used=th_abs if th_abs is not None else 0.0
    )
    
    return y_data, result


@numpy
@special_outputs(("maxima_results", csv_materializer(
    fields=["slice_index", "maxima_count", "min_distance_used", "threshold_used"],
    analysis_type="maxima_detection"
)))
def find_maxima_with_mask(
    image: np.ndarray,
    min_distance: int = 5,
    min_intensity: float = 0.0,
    label_maxima: bool = True,
) -> Tuple[np.ndarray, MaximaResult]:
    """
    Find local maxima within a masked region.
    
    Args:
        image: Stacked array (2, H, W) where:
            - image[0] is the intensity image
            - image[1] is the binary mask (non-zero = valid region)
        min_distance: Minimum distance between accepted local maxima
        min_intensity: Minimum pixel intensity to be considered as a peak
        label_maxima: If True, assign unique labels to each maxima.
    
    Returns:
        Tuple of:
            - Output image with maxima (labeled or binary), shape (1, H, W)
            - MaximaResult dataclass with detection statistics
    """
    from skimage.feature import peak_local_max
    import scipy.ndimage
    
    # Unstack inputs
    intensity_image = image[0]
    mask = image[1].astype(bool)
    
    x_data = intensity_image.copy()
    x_data[~mask] = 0
    
    th_abs = min_intensity if min_intensity > 0 else None
    
    # Find local maxima coordinates
    maxima_coords = peak_local_max(
        x_data,
        min_distance=min_distance,
        threshold_abs=th_abs,
    )
    
    # Create output image
    y_data = np.zeros(x_data.shape, dtype=np.float32)
    if len(maxima_coords) > 0:
        y_data[tuple(maxima_coords.T)] = 1.0
    
    # Optionally label each maximum with unique ID
    if label_maxima:
        y_data = scipy.ndimage.label(y_data > 0)[0].astype(np.float32)
    
    maxima_count = len(maxima_coords)
    
    result = MaximaResult(
        slice_index=0,
        maxima_count=maxima_count,
        min_distance_used=min_distance,
        threshold_used=th_abs if th_abs is not None else 0.0
    )
    
    # Return with batch dimension
    return y_data[np.newaxis, ...], result