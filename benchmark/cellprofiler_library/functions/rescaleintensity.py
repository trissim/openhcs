"""
Converted from CellProfiler: RescaleIntensity
Original: RescaleIntensity module

Rescales the intensity range of an image using various methods.
"""

import numpy as np
from typing import Tuple, Optional
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


class RescaleMethod(Enum):
    STRETCH = "stretch"
    MANUAL_INPUT_RANGE = "manual_input_range"
    MANUAL_IO_RANGE = "manual_io_range"
    DIVIDE_BY_IMAGE_MINIMUM = "divide_by_image_minimum"
    DIVIDE_BY_IMAGE_MAXIMUM = "divide_by_image_maximum"
    DIVIDE_BY_VALUE = "divide_by_value"


class AutomaticLow(Enum):
    CUSTOM = "custom"
    EACH_IMAGE = "each_image"


class AutomaticHigh(Enum):
    CUSTOM = "custom"
    EACH_IMAGE = "each_image"


@numpy(contract=ProcessingContract.PURE_2D)
def rescale_intensity(
    image: np.ndarray,
    rescale_method: RescaleMethod = RescaleMethod.STRETCH,
    automatic_low: AutomaticLow = AutomaticLow.EACH_IMAGE,
    automatic_high: AutomaticHigh = AutomaticHigh.EACH_IMAGE,
    source_low: float = 0.0,
    source_high: float = 1.0,
    dest_low: float = 0.0,
    dest_high: float = 1.0,
    divisor_value: float = 1.0,
) -> np.ndarray:
    """
    Rescale the intensity of an image using various methods.
    
    Args:
        image: Input image array (H, W)
        rescale_method: Method to use for rescaling
        automatic_low: How to determine minimum intensity for manual range methods
        automatic_high: How to determine maximum intensity for manual range methods
        source_low: Custom lower intensity limit for input image
        source_high: Custom upper intensity limit for input image
        dest_low: Lower intensity limit for output image (manual_io_range only)
        dest_high: Upper intensity limit for output image (manual_io_range only)
        divisor_value: Value to divide by (divide_by_value method only)
    
    Returns:
        Rescaled image array (H, W)
    """
    from skimage.exposure import rescale_intensity as skimage_rescale
    
    data = image.astype(np.float64)
    
    if rescale_method == RescaleMethod.STRETCH:
        # Stretch to use full intensity range based on image min/max
        in_min = np.min(data)
        in_max = np.max(data)
        if in_min == in_max:
            # Avoid division by zero for constant images
            return np.zeros_like(data)
        in_range = (in_min, in_max)
        rescaled = skimage_rescale(data, in_range=in_range, out_range=(0.0, 1.0))
        
    elif rescale_method == RescaleMethod.MANUAL_INPUT_RANGE:
        # Rescale from specified input range to 0-1
        in_range = _get_source_range(data, automatic_low, automatic_high, source_low, source_high)
        rescaled = skimage_rescale(data, in_range=in_range, out_range=(0.0, 1.0))
        
    elif rescale_method == RescaleMethod.MANUAL_IO_RANGE:
        # Rescale from specified input range to specified output range
        in_range = _get_source_range(data, automatic_low, automatic_high, source_low, source_high)
        out_range = (dest_low, dest_high)
        rescaled = skimage_rescale(data, in_range=in_range, out_range=out_range)
        
    elif rescale_method == RescaleMethod.DIVIDE_BY_IMAGE_MINIMUM:
        # Divide by image minimum
        src_min = np.min(data)
        if src_min == 0.0:
            raise ZeroDivisionError("Cannot divide pixel intensity by 0.")
        rescaled = data / src_min
        
    elif rescale_method == RescaleMethod.DIVIDE_BY_IMAGE_MAXIMUM:
        # Divide by image maximum
        src_max = np.max(data)
        if src_max == 0.0:
            src_max = 1.0  # Avoid division by zero
        rescaled = data / src_max
        
    elif rescale_method == RescaleMethod.DIVIDE_BY_VALUE:
        # Divide by specified value
        if divisor_value == 0.0:
            raise ZeroDivisionError("Cannot divide pixel intensity by 0.")
        rescaled = data / divisor_value
        
    else:
        # Default to stretch
        in_min = np.min(data)
        in_max = np.max(data)
        if in_min == in_max:
            return np.zeros_like(data)
        in_range = (in_min, in_max)
        rescaled = skimage_rescale(data, in_range=in_range, out_range=(0.0, 1.0))
    
    return rescaled.astype(np.float32)


def _get_source_range(
    data: np.ndarray,
    automatic_low: AutomaticLow,
    automatic_high: AutomaticHigh,
    source_low: float,
    source_high: float,
) -> Tuple[float, float]:
    """
    Determine the source intensity range based on settings.
    
    Args:
        data: Input image data
        automatic_low: How to determine minimum
        automatic_high: How to determine maximum
        source_low: Custom low value
        source_high: Custom high value
    
    Returns:
        Tuple of (min, max) intensity values
    """
    if automatic_low == AutomaticLow.EACH_IMAGE:
        src_min = float(np.min(data))
    else:
        src_min = source_low
    
    if automatic_high == AutomaticHigh.EACH_IMAGE:
        src_max = float(np.max(data))
    else:
        src_max = source_high
    
    return src_min, src_max


@numpy
def rescale_intensity_match_maximum(
    image: np.ndarray,
) -> np.ndarray:
    """
    Scale an image so its maximum matches another image's maximum.
    
    This function expects two images stacked along dimension 0:
    - image[0]: The image to rescale
    - image[1]: The reference image whose maximum to match
    
    Args:
        image: Stacked images (2, H, W) - input image and reference image
    
    Returns:
        Rescaled image (1, H, W)
    """
    input_data = image[0].astype(np.float64)
    reference_data = image[1].astype(np.float64)
    
    image_max = np.max(input_data)
    reference_max = np.max(reference_data)
    
    if image_max == 0:
        # Cannot scale if input max is zero
        result = input_data
    else:
        result = (input_data * reference_max) / image_max
    
    return result.astype(np.float32)[np.newaxis, :, :]