"""Converted from CellProfiler: OverlayOutlines

Places outlines of objects over a desired image.
Supports both 2D and 3D images.
"""

import numpy as np
from typing import Tuple
from enum import Enum
from openhcs.core.memory import numpy
from openhcs.core.pipeline.function_contracts import special_inputs


class LineMode(Enum):
    INNER = "inner"
    OUTER = "outer"
    THICK = "thick"


class OutlineDisplayMode(Enum):
    COLOR = "color"
    GRAYSCALE = "grayscale"


class MaxType(Enum):
    MAX_IMAGE = "max_image"
    MAX_POSSIBLE = "max_possible"


@numpy
@special_inputs("labels")
def overlay_outlines(
    image: np.ndarray,
    labels: np.ndarray,
    blank_image: bool = False,
    display_mode: OutlineDisplayMode = OutlineDisplayMode.COLOR,
    line_mode: LineMode = LineMode.INNER,
    max_type: MaxType = MaxType.MAX_IMAGE,
    outline_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> np.ndarray:
    """
    Overlay outlines of segmented objects on an image.
    
    Args:
        image: Input image (H, W), grayscale or will be converted
        labels: Label image from segmentation (H, W)
        blank_image: If True, draw outlines on black background
        display_mode: COLOR for colored outlines, GRAYSCALE for intensity outlines
        line_mode: INNER, OUTER, or THICK boundary mode
        max_type: For grayscale mode, MAX_IMAGE uses image max, MAX_POSSIBLE uses 1.0
        outline_color: RGB tuple (0-1 range) for outline color in color mode
    
    Returns:
        Image with outlines overlaid (H, W, 3) for color or (H, W) for grayscale
    """
    import skimage.segmentation
    import skimage.color
    from skimage import img_as_float
    
    # Ensure image is float
    image = img_as_float(image)
    
    # Create base image
    if blank_image:
        # Black background
        if display_mode == OutlineDisplayMode.COLOR:
            base_image = np.zeros(image.shape + (3,), dtype=np.float32)
        else:
            base_image = np.zeros(image.shape, dtype=np.float32)
    else:
        # Use input image as background
        if display_mode == OutlineDisplayMode.COLOR:
            # Convert grayscale to RGB if needed
            if image.ndim == 2:
                base_image = skimage.color.gray2rgb(image).astype(np.float32)
            else:
                base_image = image.astype(np.float32)
        else:
            if image.ndim == 3:
                base_image = skimage.color.rgb2gray(image).astype(np.float32)
            else:
                base_image = image.astype(np.float32)
    
    # Ensure labels match image shape
    labels_2d = labels.astype(np.int32)
    if labels_2d.shape != base_image.shape[:2]:
        # Resize labels to match image if needed
        from skimage.transform import resize
        labels_2d = resize(
            labels_2d, 
            base_image.shape[:2], 
            order=0, 
            preserve_range=True,
            anti_aliasing=False
        ).astype(np.int32)
    
    # Determine outline color
    if display_mode == OutlineDisplayMode.COLOR:
        color = outline_color
    else:
        if blank_image or max_type == MaxType.MAX_POSSIBLE:
            color = 1.0
        else:
            color = float(np.max(base_image))
    
    # Get line mode string for skimage
    mode_str = line_mode.value
    
    # Draw outlines
    if display_mode == OutlineDisplayMode.COLOR:
        # Ensure base_image is RGB for mark_boundaries
        if base_image.ndim == 2:
            base_image = skimage.color.gray2rgb(base_image)
        
        result = skimage.segmentation.mark_boundaries(
            base_image,
            labels_2d,
            color=color,
            mode=mode_str,
        )
        return result.astype(np.float32)
    else:
        # For grayscale, we need to work with RGB then convert back
        if base_image.ndim == 2:
            rgb_image = skimage.color.gray2rgb(base_image)
        else:
            rgb_image = base_image
        
        # Use white color for marking, then convert to grayscale
        gray_color = (color, color, color) if isinstance(color, float) else color
        
        result = skimage.segmentation.mark_boundaries(
            rgb_image,
            labels_2d,
            color=gray_color,
            mode=mode_str,
        )
        
        # Convert back to grayscale
        result_gray = skimage.color.rgb2gray(result)
        return result_gray.astype(np.float32)
