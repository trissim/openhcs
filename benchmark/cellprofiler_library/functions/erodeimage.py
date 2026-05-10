"""Converted from CellProfiler: ErodeImage."""

import numpy as np
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract

from openhcs.processing.backends.cellprofiler.structuring_elements import (
    StructuringElement,
    apply_structuring_element,
    build_structuring_element,
)


@numpy(contract=ProcessingContract.PURE_2D)
def erode_image(
    image: np.ndarray,
    structuring_element: StructuringElement | str = StructuringElement.DISK,
    size: int = 3,
) -> np.ndarray:
    """Apply morphological erosion to an image.

    Erosion shrinks bright regions and enlarges dark regions. It is useful for
    removing small bright spots (noise) and separating touching objects.

    Args:
        image: Input image (H, W) - grayscale or binary
        structuring_element: Shape of the structuring element.
        size: Size of the structuring element.

    Returns:
        Eroded image with same dimensions as input
    """
    from skimage.morphology import erosion

    eroded = apply_structuring_element(
        image,
        build_structuring_element(structuring_element, size),
        lambda spatial_image, footprint: erosion(spatial_image, footprint),
    )
    return eroded.astype(image.dtype)
