"""
Converted from CellProfiler: OverlayObjects
Overlays labeled objects on an image with colored regions.
"""

import numpy as np
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_inputs


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
def overlay_objects(
    image: np.ndarray,
    labels: np.ndarray,
    opacity: float = 0.3,
    max_label: int = None,
    seed: int = None,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Overlay labeled objects on an image with colored regions.
    
    Args:
        image: Input grayscale or RGB image (H, W) or (H, W, 3)
        labels: Label image where each object has a unique integer ID
        opacity: Opacity of the overlay (0.0 = transparent, 1.0 = opaque)
        max_label: Maximum label value for colormap normalization. If None, uses max in labels.
        seed: Random seed for reproducible colors (if using random colormap)
        colormap: Name of colormap to use for coloring objects
    
    Returns:
        RGB image with colored object overlay (H, W, 3)
    """
    from skimage.color import label2rgb
    
    # Ensure image is 2D grayscale for overlay
    if image.ndim == 3:
        # If RGB, convert to grayscale for background
        img_gray = np.mean(image, axis=-1)
    else:
        img_gray = image.copy()
    
    # Normalize image to 0-1 range if needed
    if img_gray.max() > 1.0:
        img_gray = img_gray / img_gray.max()
    
    # Ensure labels are integer type
    labels_int = labels.astype(np.int32)
    
    # Determine max label for color normalization
    if max_label is None:
        max_label = labels_int.max()
    
    # Set random state if seed provided
    if seed is not None:
        np.random.seed(seed)
    
    # Generate colors for each label using colormap
    n_labels = max_label + 1
    
    # Create colormap colors
    try:
        from matplotlib import colormaps
        cmap = colormaps.get_cmap(colormap)
    except (ImportError, AttributeError):
        # Fallback for older matplotlib versions
        import matplotlib.pyplot as plt
        cmap = plt.cm.get_cmap(colormap)
    
    # Generate colors for each label (skip 0 which is background)
    colors = []
    for i in range(1, n_labels):
        color_val = (i / max(n_labels - 1, 1)) if n_labels > 1 else 0.5
        rgba = cmap(color_val)
        colors.append(rgba[:3])  # RGB only, no alpha
    
    # Use skimage's label2rgb for overlay
    if len(colors) > 0:
        overlay = label2rgb(
            labels_int,
            image=img_gray,
            colors=colors,
            alpha=opacity,
            bg_label=0,
            bg_color=None,
            kind='overlay'
        )
    else:
        # No objects, just convert grayscale to RGB
        overlay = np.stack([img_gray, img_gray, img_gray], axis=-1)
    
    # Ensure output is float32 in range 0-1
    overlay = np.clip(overlay, 0, 1).astype(np.float32)
    
    return overlay