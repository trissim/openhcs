"""
Converted from CellProfiler: ExpandOrShrinkObjects
Original: expand_or_shrink_objects
"""

import numpy as np
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.processing.materialization import materialize_segmentation_masks


class ExpandShrinkMode(Enum):
    EXPAND_DEFINED_PIXELS = "expand_defined_pixels"
    EXPAND_INFINITE = "expand_infinite"
    SHRINK_DEFINED_PIXELS = "shrink_defined_pixels"
    SHRINK_TO_POINT = "shrink_to_point"
    ADD_DIVIDING_LINES = "add_dividing_lines"
    DESPUR = "despur"
    SKELETONIZE = "skeletonize"


def _expand_defined_pixels(labels: np.ndarray, iterations: int) -> np.ndarray:
    """Expand labeled objects by a defined number of pixels."""
    from scipy.ndimage import distance_transform_edt, maximum_filter
    
    if iterations <= 0:
        return labels.copy()
    
    result = labels.copy()
    for _ in range(iterations):
        # Create a mask of the current labels
        mask = result > 0
        # Dilate by finding nearest label for each background pixel within 1 pixel
        distances, indices = distance_transform_edt(~mask, return_indices=True)
        # Only expand by 1 pixel at a time
        expand_mask = (distances > 0) & (distances <= 1)
        result[expand_mask] = result[indices[0][expand_mask], indices[1][expand_mask]]
    
    return result


def _expand_until_touching(labels: np.ndarray) -> np.ndarray:
    """Expand labeled objects until they touch (Voronoi-like expansion)."""
    from scipy.ndimage import distance_transform_edt
    
    if labels.max() == 0:
        return labels.copy()
    
    # Use distance transform to find nearest labeled pixel for each background pixel
    mask = labels > 0
    distances, indices = distance_transform_edt(~mask, return_indices=True)
    
    # Assign each pixel to its nearest labeled object
    result = labels[indices[0], indices[1]]
    
    return result


def _shrink_defined_pixels(labels: np.ndarray, iterations: int, fill: bool) -> np.ndarray:
    """Shrink labeled objects by a defined number of pixels."""
    from scipy.ndimage import binary_erosion, generate_binary_structure
    
    if iterations <= 0:
        return labels.copy()
    
    result = np.zeros_like(labels)
    struct = generate_binary_structure(2, 1)  # 4-connectivity
    
    for label_id in range(1, labels.max() + 1):
        obj_mask = labels == label_id
        eroded = binary_erosion(obj_mask, structure=struct, iterations=iterations)
        
        if fill and not eroded.any():
            # If object disappeared, keep a single pixel at centroid
            coords = np.where(obj_mask)
            if len(coords[0]) > 0:
                cy, cx = int(np.mean(coords[0])), int(np.mean(coords[1]))
                eroded[cy, cx] = True
        
        result[eroded] = label_id
    
    return result


def _shrink_to_point(labels: np.ndarray, fill: bool) -> np.ndarray:
    """Shrink each labeled object to a single point at its centroid."""
    from skimage.measure import regionprops
    
    result = np.zeros_like(labels)
    
    props = regionprops(labels.astype(np.int32))
    for prop in props:
        cy, cx = int(prop.centroid[0]), int(prop.centroid[1])
        # Ensure centroid is within image bounds
        cy = max(0, min(labels.shape[0] - 1, cy))
        cx = max(0, min(labels.shape[1] - 1, cx))
        result[cy, cx] = prop.label
    
    return result


def _add_dividing_lines(labels: np.ndarray) -> np.ndarray:
    """Add 1-pixel dividing lines between touching objects."""
    from scipy.ndimage import maximum_filter, minimum_filter
    
    if labels.max() == 0:
        return labels.copy()
    
    result = labels.copy()
    
    # Find pixels where neighboring labels differ (boundaries)
    max_filt = maximum_filter(labels, size=3)
    min_filt = minimum_filter(labels, size=3)
    
    # Boundary pixels are where max != min and both are > 0
    boundary = (max_filt != min_filt) & (min_filt > 0)
    
    result[boundary] = 0
    
    return result


def _despur(labels: np.ndarray, iterations: int) -> np.ndarray:
    """Remove spurs (small protrusions) from labeled objects."""
    from scipy.ndimage import binary_erosion, binary_dilation, generate_binary_structure
    
    if iterations <= 0:
        return labels.copy()
    
    result = np.zeros_like(labels)
    struct = generate_binary_structure(2, 1)
    
    for label_id in range(1, labels.max() + 1):
        obj_mask = labels == label_id
        # Opening operation removes small protrusions
        opened = binary_erosion(obj_mask, structure=struct, iterations=iterations)
        opened = binary_dilation(opened, structure=struct, iterations=iterations)
        result[opened] = label_id
    
    return result


def _skeletonize_labels(labels: np.ndarray) -> np.ndarray:
    """Reduce labeled objects to their skeletons."""
    from skimage.morphology import skeletonize
    
    result = np.zeros_like(labels)
    
    for label_id in range(1, labels.max() + 1):
        obj_mask = labels == label_id
        skeleton = skeletonize(obj_mask)
        result[skeleton] = label_id
    
    return result


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(("labels", materialize_segmentation_masks))
def expand_or_shrink_objects(
    image: np.ndarray,
    labels: np.ndarray,
    mode: ExpandShrinkMode = ExpandShrinkMode.EXPAND_DEFINED_PIXELS,
    iterations: int = 1,
    fill_holes: bool = True,
) -> tuple:
    """
    Expand or shrink labeled objects using various methods.
    
    Args:
        image: Input image (H, W) - passed through unchanged
        labels: Label image (H, W) - integer labels for each object
        mode: Operation mode - expand, shrink, skeletonize, etc.
        iterations: Number of pixels to expand/shrink (for applicable modes)
        fill_holes: Whether to preserve objects that would disappear (for shrink modes)
    
    Returns:
        Tuple of (image, modified_labels)
    """
    labels_int = labels.astype(np.int32)
    
    if mode == ExpandShrinkMode.EXPAND_DEFINED_PIXELS:
        result_labels = _expand_defined_pixels(labels_int, iterations)
    elif mode == ExpandShrinkMode.EXPAND_INFINITE:
        result_labels = _expand_until_touching(labels_int)
    elif mode == ExpandShrinkMode.SHRINK_DEFINED_PIXELS:
        result_labels = _shrink_defined_pixels(labels_int, iterations, fill_holes)
    elif mode == ExpandShrinkMode.SHRINK_TO_POINT:
        result_labels = _shrink_to_point(labels_int, fill_holes)
    elif mode == ExpandShrinkMode.ADD_DIVIDING_LINES:
        result_labels = _add_dividing_lines(labels_int)
    elif mode == ExpandShrinkMode.DESPUR:
        result_labels = _despur(labels_int, iterations)
    elif mode == ExpandShrinkMode.SKELETONIZE:
        result_labels = _skeletonize_labels(labels_int)
    else:
        result_labels = labels_int.copy()
    
    return image, result_labels.astype(np.float32)