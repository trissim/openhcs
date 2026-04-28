"""
Converted from CellProfiler: Closing
Original: closing
"""

import numpy as np
from typing import Literal
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


@numpy(contract=ProcessingContract.PURE_2D)
def closing(
    image: np.ndarray,
    structuring_element: Literal["disk", "square", "diamond", "octagon", "star"] = "disk",
    size: int = 3,
) -> np.ndarray:
    """
    Apply morphological closing to an image.
    
    Closing is a dilation followed by an erosion. It is useful for closing
    small holes in foreground objects and connecting nearby objects.
    
    Args:
        image: Input image with shape (H, W)
        structuring_element: Shape of the structuring element.
            Options: "disk", "square", "diamond", "octagon", "star"
        size: Size of the structuring element (radius for disk, side length for square, etc.)
    
    Returns:
        Morphologically closed image with shape (H, W)
    """
    from skimage.morphology import (
        closing as skimage_closing,
        disk,
        square,
        diamond,
        octagon,
        star,
    )
    
    # Create structuring element based on type
    if structuring_element == "disk":
        selem = disk(size)
    elif structuring_element == "square":
        selem = square(size)
    elif structuring_element == "diamond":
        selem = diamond(size)
    elif structuring_element == "octagon":
        # octagon requires two parameters, use size for both
        selem = octagon(size, size)
    elif structuring_element == "star":
        selem = star(size)
    else:
        # Default to disk if unknown
        selem = disk(size)
    
    # Apply morphological closing
    result = skimage_closing(image, selem)
    
    return result.astype(image.dtype)