"""
Converted from CellProfiler: Crop
Original: crop, measure_area_retained_after_cropping, measure_original_image_area, get_measurements
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.processing.materialization import csv_materializer


class RemovalMethod(Enum):
    """Method for handling pixels outside the cropping region."""
    SET_TO_ZERO = "set_to_zero"
    SET_TO_MASK = "set_to_mask"
    REMOVE = "remove"


@dataclass
class CropMeasurement:
    """Measurements from cropping operation."""
    slice_index: int
    original_area: int
    area_retained: int
    fraction_retained: float


def _get_cropped_mask(
    cropping: np.ndarray,
    mask: np.ndarray,
    removal_method: RemovalMethod
) -> np.ndarray:
    """Apply cropping to an existing mask."""
    if removal_method == RemovalMethod.REMOVE:
        # For REMOVE, we need to extract the bounding box
        rows = np.any(cropping, axis=1)
        cols = np.any(cropping, axis=0)
        if not np.any(rows) or not np.any(cols):
            return np.zeros((1, 1), dtype=mask.dtype)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return mask[rmin:rmax+1, cmin:cmax+1] * cropping[rmin:rmax+1, cmin:cmax+1]
    else:
        # SET_TO_ZERO or SET_TO_MASK: keep same size, apply cropping
        return mask * cropping


def _get_cropped_image_mask(
    cropping: np.ndarray,
    mask: np.ndarray,
    orig_image_mask: np.ndarray,
    removal_method: RemovalMethod
) -> np.ndarray:
    """Get the combined mask after cropping."""
    combined = mask * orig_image_mask
    if removal_method == RemovalMethod.REMOVE:
        rows = np.any(cropping, axis=1)
        cols = np.any(cropping, axis=0)
        if not np.any(rows) or not np.any(cols):
            return np.zeros((1, 1), dtype=combined.dtype)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return combined[rmin:rmax+1, cmin:cmax+1]
    return combined


def _get_cropped_image_pixels(
    image: np.ndarray,
    cropping: np.ndarray,
    mask: np.ndarray,
    removal_method: RemovalMethod
) -> np.ndarray:
    """Crop image pixels according to the cropping mask and removal method."""
    if removal_method == RemovalMethod.REMOVE:
        # Extract bounding box of cropping region
        rows = np.any(cropping, axis=1)
        cols = np.any(cropping, axis=0)
        if not np.any(rows) or not np.any(cols):
            return np.zeros((1, 1), dtype=image.dtype)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        cropped = image[rmin:rmax+1, cmin:cmax+1].copy()
        # Apply mask within the cropped region
        crop_mask = cropping[rmin:rmax+1, cmin:cmax+1]
        cropped = cropped * crop_mask
        return cropped
    elif removal_method == RemovalMethod.SET_TO_ZERO:
        # Keep same size, set pixels outside cropping to zero
        return image * cropping
    else:  # SET_TO_MASK
        # Keep same size, apply mask
        return image * mask


@numpy
@special_outputs(
    ("crop_measurements", csv_materializer(
        fields=["slice_index", "original_area", "area_retained", "fraction_retained"],
        analysis_type="crop"
    ))
)
def crop(
    image: np.ndarray,
    removal_method: RemovalMethod = RemovalMethod.SET_TO_ZERO,
) -> Tuple[np.ndarray, CropMeasurement]:
    """
    Crop an image using a cropping mask.
    
    Args:
        image: Shape (3, H, W) or (4, H, W) - stacked arrays:
               [0]: Original image pixels
               [1]: Cropping mask (1 for pixels to keep, 0 to remove)
               [2]: Previous cropping mask (or ones if none)
               [3]: Original image mask (optional, defaults to ones)
        removal_method: How to handle pixels outside cropping region
        
    Returns:
        Tuple of cropped image and measurements
    """
    # Unstack inputs from dimension 0
    orig_image_pixels = image[0]
    cropping = image[1].astype(bool).astype(np.float32)
    
    # Handle optional inputs
    if image.shape[0] >= 3:
        mask = image[2].astype(bool).astype(np.float32)
    else:
        mask = np.ones_like(orig_image_pixels)
    
    if image.shape[0] >= 4:
        orig_image_mask = image[3].astype(bool).astype(np.float32)
    else:
        orig_image_mask = np.ones_like(orig_image_pixels)
    
    # Crop the mask
    cropped_mask = _get_cropped_mask(cropping, mask, removal_method)
    
    # Crop the image
    cropped_pixel_data = _get_cropped_image_pixels(
        orig_image_pixels, cropping, cropped_mask, removal_method
    )
    
    # Calculate measurements
    original_area = int(np.prod(orig_image_pixels.shape))
    area_retained = int(np.sum(cropping))
    fraction_retained = area_retained / original_area if original_area > 0 else 0.0
    
    measurements = CropMeasurement(
        slice_index=0,
        original_area=original_area,
        area_retained=area_retained,
        fraction_retained=fraction_retained
    )
    
    # Return with batch dimension
    result = cropped_pixel_data[np.newaxis, :, :]
    
    return result, measurements


@numpy(contract=ProcessingContract.PURE_2D)
def crop_simple(
    image: np.ndarray,
    crop_top: int = 0,
    crop_bottom: int = 0,
    crop_left: int = 0,
    crop_right: int = 0,
) -> np.ndarray:
    """
    Simple rectangular crop by specifying pixel amounts to remove from each edge.
    
    Args:
        image: Input image (H, W)
        crop_top: Pixels to remove from top
        crop_bottom: Pixels to remove from bottom
        crop_left: Pixels to remove from left
        crop_right: Pixels to remove from right
        
    Returns:
        Cropped image
    """
    h, w = image.shape
    
    # Calculate crop bounds
    y_start = crop_top
    y_end = h - crop_bottom if crop_bottom > 0 else h
    x_start = crop_left
    x_end = w - crop_right if crop_right > 0 else w
    
    # Ensure valid bounds
    y_start = max(0, min(y_start, h - 1))
    y_end = max(y_start + 1, min(y_end, h))
    x_start = max(0, min(x_start, w - 1))
    x_end = max(x_start + 1, min(x_end, w))
    
    return image[y_start:y_end, x_start:x_end].copy()