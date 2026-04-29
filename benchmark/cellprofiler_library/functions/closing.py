"""Converted from CellProfiler: Closing."""

import numpy as np
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract

from .structuring_elements import StructuringElement, build_structuring_element


@numpy(contract=ProcessingContract.PURE_2D)
def closing(
    image: np.ndarray,
    structuring_element: StructuringElement = StructuringElement.DISK,
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
    from skimage.morphology import closing as skimage_closing

    result = skimage_closing(
        image,
        build_structuring_element(structuring_element, size),
    )
    return result.astype(image.dtype)
