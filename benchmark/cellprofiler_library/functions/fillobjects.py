"""
Converted from CellProfiler: FillObjects
Original: fillobjects
"""

import numpy as np
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.core.runtime_values import object_label_dense_array
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.processing.materialization import segmentation_mask_rois
from openhcs.processing.backends.analysis.region_properties import (
    LabelRegionPropertiesBackendStrategy,
)


class FillMode(Enum):
    HOLES = "holes"
    CONVEX_HULL = "convex_hull"


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(("labels", segmentation_mask_rois()))
def fill_objects(
    image: np.ndarray,
    labels: np.ndarray,
    mode: FillMode = FillMode.HOLES,
    diameter: float = 64.0,
    morphology_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
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
    from skimage.morphology import remove_small_holes
    from openhcs.processing.backends.cellprofiler.morphology import (
        MorphologyBackendStrategy,
    )
    label_array = object_label_dense_array(labels, dtype=np.int32)
    
    if label_array.max() == 0:
        # No objects, return as-is
        return image, label_array.copy()
    
    filled_labels = np.zeros_like(label_array)
    region_props = LabelRegionPropertiesBackendStrategy.for_memory_type().measure_2d(
        label_array
    )
    
    if mode == FillMode.HOLES:
        # Fill holes smaller than specified diameter
        # Convert diameter to area (assuming circular holes)
        max_hole_area = np.pi * (diameter / 2.0) ** 2
        
        for label_id in region_props.label:
            obj_mask = label_array == int(label_id)
            
            # Fill small holes in this object
            filled_mask = remove_small_holes(
                obj_mask, 
                area_threshold=int(max_hole_area),
                connectivity=1
            )
            
            filled_labels[filled_mask] = int(label_id)
            
    elif mode == FillMode.CONVEX_HULL:
        morphology = MorphologyBackendStrategy.for_callable(
            fill_objects,
            backend_provider=morphology_backend_provider,
        )

        # Replace each object with its convex hull
        for index, label_id in enumerate(region_props.label):
            label_int = int(label_id)
            obj_mask = label_array == label_int
            
            # Get bounding box for efficiency
            minr = int(region_props.bbox_min_y[index])
            minc = int(region_props.bbox_min_x[index])
            maxr = int(region_props.bbox_max_y[index])
            maxc = int(region_props.bbox_max_x[index])
            
            # Extract object region
            obj_crop = obj_mask[minr:maxr, minc:maxc]
            
            # Compute convex hull
            if obj_crop.sum() > 2:  # Need at least 3 points for convex hull
                hull = morphology.convex_hull_image(obj_crop)
                # Place back into full image
                filled_labels[minr:maxr, minc:maxc][hull] = label_int
            else:
                # Too few points, keep original
                filled_labels[obj_mask] = label_int
    else:
        raise ValueError(
            f"Mode '{mode}' is not supported. "
            f"Available modes are: 'holes' and 'convex_hull'."
        )
    
    return image, filled_labels.astype(label_array.dtype)
