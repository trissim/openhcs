"""
Converted from CellProfiler: ConvertObjectsToImage
Original: convert_objects_to_image

Converts object labels to various image representations (binary, grayscale, color, uint16).
"""

import numpy as np

from openhcs.core.memory import numpy
from openhcs.core.pipeline.function_contracts import special_inputs
from openhcs.core.runtime_values import object_label_dense_array
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.processing.backends.cellprofiler.object_images import (
    ImageMode,
    ImageModeRenderer,
)


def _coerce_image_mode(image_mode: ImageMode | str) -> ImageMode:
    return coerce_cellprofiler_enum(ImageMode, image_mode)


@numpy
@special_inputs("labels")
def convert_objects_to_image(
    image: np.ndarray,
    labels: np.ndarray,
    image_mode: ImageMode = ImageMode.COLOR,
    colormap_value: str = "jet",
) -> np.ndarray:
    """
    Convert object labels to an image representation.
    
    Args:
        image: Input image (H, W) - used for shape reference
        labels: Object labels (H, W) - integer labels where 0 is background
        image_mode: Output image format (BINARY, GRAYSCALE, COLOR, UINT16)
        colormap_value: Matplotlib colormap name for COLOR mode
    
    Returns:
        Converted image:
        - BINARY: (H, W) boolean mask where objects are True
        - GRAYSCALE: (H, W) float with normalized label values
        - COLOR: (H, W, 3) RGB image with colored objects
        - UINT16: (H, W) integer labels
    """
    del image
    labels = object_label_dense_array(labels, dtype=np.int32)
    resolved_image_mode = _coerce_image_mode(image_mode)
    return ImageModeRenderer.for_image_mode(resolved_image_mode).render(
        labels,
        colormap_value=colormap_value,
    )
