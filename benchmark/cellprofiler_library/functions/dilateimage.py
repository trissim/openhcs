"""Converted from CellProfiler: DilateImage."""

import numpy as np
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract

from openhcs.processing.backends.cellprofiler.structuring_elements import (
    StructuringElement,
    apply_structuring_element,
    build_structuring_element,
)


@numpy(contract=ProcessingContract.PURE_2D)
def dilate_image(
    image: np.ndarray,
    structuring_element: StructuringElement = StructuringElement.DISK,
    size: int = 3,
) -> np.ndarray:
    """Apply morphological dilation to an image.

    Morphological dilation expands bright regions in an image. It is useful for
    filling small holes, connecting nearby objects, and expanding object boundaries.

    Args:
        image: Input image with shape (H, W). Can be grayscale or binary.
        structuring_element: Shape of the structuring element.
        size: Size of the structuring element.

    Returns:
        Dilated image with same shape (H, W) as input.
    """
    from skimage.morphology import dilation

    dilated = apply_structuring_element(
        image,
        build_structuring_element(structuring_element, size),
        lambda spatial_image, footprint: dilation(spatial_image, footprint),
    )
    return dilated.astype(image.dtype)
