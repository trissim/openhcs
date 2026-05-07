"""Converted from CellProfiler: ErodeImage."""

import numpy as np
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract

from benchmark.cellprofiler_library.functions.spatial_axes import (
    apply_over_trailing_spatial_axes,
)
from benchmark.cellprofiler_library.functions.structuring_elements import (
    StructuringElement,
    adapt_structuring_element_rank,
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

    footprint = adapt_structuring_element_rank(
        build_structuring_element(structuring_element, size),
        image.ndim,
    )
    eroded = apply_over_trailing_spatial_axes(
        image,
        footprint.ndim,
        lambda spatial_image: erosion(spatial_image, footprint),
    )
    return eroded.astype(image.dtype)
