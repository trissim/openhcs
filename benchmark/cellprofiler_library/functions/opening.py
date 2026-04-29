"""Converted from CellProfiler: Opening."""

import numpy as np
from openhcs.core.memory import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract

from .structuring_elements import StructuringElement, build_structuring_element


@numpy(contract=ProcessingContract.PURE_2D)
def opening(
    image: np.ndarray,
    structuring_element: StructuringElement = StructuringElement.DISK,
    size: int = 3,
) -> np.ndarray:
    """
    Apply morphological opening to an image.
    
    Opening is erosion followed by dilation. It removes small bright spots
    (noise) and smooths object boundaries while preserving object size.
    
    Args:
        image: Input image with shape (H, W)
        structuring_element: Shape of the structuring element.
            Options: "disk", "square", "diamond", "octagon", "star"
        size: Size of the structuring element (radius for disk, side length for square, etc.)
    
    Returns:
        Opened image with shape (H, W)
    """
    from skimage.morphology import opening as skimage_opening

    result = skimage_opening(
        image,
        build_structuring_element(structuring_element, size),
    )
    return result.astype(image.dtype)
