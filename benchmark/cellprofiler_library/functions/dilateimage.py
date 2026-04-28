"""
Converted from CellProfiler: DilateImage
Original: dilate_image
"""

import numpy as np
from typing import Tuple
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


class StructuringElementShape(Enum):
    DISK = "disk"
    SQUARE = "square"
    DIAMOND = "diamond"
    OCTAGON = "octagon"
    STAR = "star"


@numpy(contract=ProcessingContract.PURE_2D)
def dilate_image(
    image: np.ndarray,
    structuring_element_shape: StructuringElementShape = StructuringElementShape.DISK,
    structuring_element_size: int = 3,
) -> np.ndarray:
    """Apply morphological dilation to an image.
    
    Morphological dilation expands bright regions in an image. It is useful for
    filling small holes, connecting nearby objects, and expanding object boundaries.
    
    Args:
        image: Input image with shape (H, W). Can be grayscale or binary.
        structuring_element_shape: Shape of the structuring element.
            Options: DISK, SQUARE, DIAMOND, OCTAGON, STAR.
        structuring_element_size: Size (radius for disk/diamond, side for square)
            of the structuring element. Must be > 0.
            
    Returns:
        Dilated image with same shape (H, W) as input.
    """
    from skimage.morphology import (
        dilation,
        disk,
        square,
        diamond,
        octagon,
        star,
    )
    
    # Ensure size is at least 1
    size = max(1, structuring_element_size)
    
    # Create structuring element based on shape
    if structuring_element_shape == StructuringElementShape.DISK:
        selem = disk(size)
    elif structuring_element_shape == StructuringElementShape.SQUARE:
        selem = square(size)
    elif structuring_element_shape == StructuringElementShape.DIAMOND:
        selem = diamond(size)
    elif structuring_element_shape == StructuringElementShape.OCTAGON:
        # octagon takes two parameters: m and n
        # For simplicity, use size for both
        selem = octagon(size, size)
    elif structuring_element_shape == StructuringElementShape.STAR:
        # star takes a single parameter 'a'
        selem = star(size)
    else:
        # Default to disk
        selem = disk(size)
    
    # Apply dilation
    dilated = dilation(image, selem)
    
    return dilated.astype(image.dtype)