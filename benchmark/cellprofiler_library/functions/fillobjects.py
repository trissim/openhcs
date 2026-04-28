"""
Converted from CellProfiler: FillObjects
Original: fillobjects
"""

import numpy as np
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.processing.materialization import materialize_segmentation_masks


class FillMode(Enum):
    HOLES = "holes"
    CONVEX_HULL = "convex_hull"


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(("labels", materialize_segmentation_masks))
def fill_objects(
    image: np.ndarray,
    labels: np.ndarray,
    mode: FillMode = FillMode.HOLES,
    diameter: float = 64.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fill holes in objects or convert objects to their convex hulls.
    
    Args:
        image: Input image (H, W) - passed through unchanged
        labels: Label image (H, W) where each object has a unique integer ID
        mode: Fill mode - 'holes' to fill holes smaller than diameter,
              'convex_hull' to replace objects with their convex hulls
        diameter: Maximum diameter of holes to fill (only used in 'holes' mode)
    
    Returns:
        Tuple of (original image, filled labels)
    """
    from scipy.ndimage import binary_fill_holes, label as nd_label
    from skimage.morphology import remove_small_holes, convex_hull_image
    from skimage.measure import regionprops
    
    if labels.max() == 0:
        # No objects, return as-is
        return image, labels.copy()
    
    filled_labels = np.zeros_like(labels)
    
    if mode == FillMode.HOLES:
        # Fill holes smaller than specified diameter
        # Convert diameter to area (assuming circular holes)
        max_hole_area = np.pi * (diameter / 2.0) ** 2
        
        for region in regionprops(labels.astype(np.int32)):
            obj_mask = labels == region.label
            
            # Fill small holes in this object
            filled_mask = remove_small_holes(
                obj_mask, 
                area_threshold=int(max_hole_area),
                connectivity=1
            )
            
            filled_labels[filled_mask] = region.label
            
    elif mode == FillMode.CONVEX_HULL:
        # Replace each object with its convex hull
        for region in regionprops(labels.astype(np.int32)):
            obj_mask = labels == region.label
            
            # Get bounding box for efficiency
            minr, minc, maxr, maxc = region.bbox
            
            # Extract object region
            obj_crop = obj_mask[minr:maxr, minc:maxc]
            
            # Compute convex hull
            if obj_crop.sum() > 2:  # Need at least 3 points for convex hull
                hull = convex_hull_image(obj_crop)
                # Place back into full image
                filled_labels[minr:maxr, minc:maxc][hull] = region.label
            else:
                # Too few points, keep original
                filled_labels[obj_mask] = region.label
    else:
        raise ValueError(
            f"Mode '{mode}' is not supported. "
            f"Available modes are: 'holes' and 'convex_hull'."
        )
    
    return image, filled_labels.astype(labels.dtype)