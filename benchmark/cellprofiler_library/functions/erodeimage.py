"""
Converted from CellProfiler: ErodeImage
Original: erode_image
"""

import numpy as np
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
def erode_image(
    image: np.ndarray,
    structuring_element_shape: StructuringElementShape = StructuringElementShape.DISK,
    structuring_element_size: int = 3,
) -> np.ndarray:
    """Apply morphological erosion to an image.
    
    Erosion shrinks bright regions and enlarges dark regions. It is useful for
    removing small bright spots (noise) and separating touching objects.
    
    Args:
        image: Input image (H, W) - grayscale or binary
        structuring_element_shape: Shape of the structuring element
        structuring_element_size: Size/radius of the structuring element (must be > 0)
        
    Returns:
        Eroded image with same dimensions as input
    """
    from skimage.morphology import erosion, disk, square, diamond, octagon, star
    
    # Create structuring element based on shape
    if structuring_element_shape == StructuringElementShape.DISK:
        selem = disk(structuring_element_size)
    elif structuring_element_shape == StructuringElementShape.SQUARE:
        # square() takes the side length, not radius
        side = 2 * structuring_element_size + 1
        selem = square(side)
    elif structuring_element_shape == StructuringElementShape.DIAMOND:
        selem = diamond(structuring_element_size)
    elif structuring_element_shape == StructuringElementShape.OCTAGON:
        # octagon takes m and n parameters, use size for both
        selem = octagon(structuring_element_size, structuring_element_size)
    elif structuring_element_shape == StructuringElementShape.STAR:
        selem = star(structuring_element_size)
    else:
        # Default to disk
        selem = disk(structuring_element_size)
    
    # Apply erosion
    eroded = erosion(image, selem)
    
    return eroded.astype(image.dtype)