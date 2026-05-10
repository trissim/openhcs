"""Converted from CellProfiler: Opening."""

import numpy as np
from openhcs.core.memory import numpy
from openhcs.core.runtime_values import (
    image_payload_data,
    image_payload_metadata,
    with_image_payload_data,
)
from openhcs.processing.backends.cellprofiler._backend import (
    CellProfilerBackendProvider,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract

from openhcs.processing.backends.cellprofiler.structuring_elements import (
    StructuringElement,
    apply_structuring_element,
    build_structuring_element,
)


@numpy(contract=ProcessingContract.PURE_2D)
def opening(
    image: np.ndarray,
    structuring_element: StructuringElement = StructuringElement.DISK,
    size: int = 3,
    morphology_backend_provider: CellProfilerBackendProvider | None = (
        CellProfilerBackendProvider.OPENCV
    ),
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
    from openhcs.processing.backends.cellprofiler.morphology import (
        MorphologyBackendStrategy,
    )

    pixel_data = image_payload_data(image)
    morphology = MorphologyBackendStrategy.for_callable(
        opening,
        backend_provider=morphology_backend_provider,
    )
    result = apply_structuring_element(
        pixel_data,
        build_structuring_element(structuring_element, size),
        morphology.grayscale_opening,
    )
    return with_image_payload_data(
        image,
        result.astype(pixel_data.dtype, copy=False),
        metadata=image_payload_metadata(image).without_unit_interval_intensity_scale(),
    )
