"""Converted from CellProfiler: ConvertImageToObjects"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.processing.materialization import csv_materializer
from openhcs.processing.backends.analysis.cell_counting_cpu import materialize_segmentation_masks


@dataclass
class ObjectConversionStats:
    slice_index: int
    object_count: int
    mean_area: float
    total_area: int


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(
    ("conversion_stats", csv_materializer(
        fields=["slice_index", "object_count", "mean_area", "total_area"],
        analysis_type="object_conversion"
    )),
    ("labels", materialize_segmentation_masks)
)
def convert_image_to_objects(
    image: np.ndarray,
    cast_to_bool: bool = False,
    preserve_label: bool = False,
    background: int = 0,
    connectivity: int = 1,
) -> Tuple[np.ndarray, ObjectConversionStats, np.ndarray]:
    """Convert an image to labeled objects.
    
    Takes a grayscale or binary image and converts it to a labeled object image.
    Can optionally preserve existing labels or create new labels via connected
    component analysis.
    
    Args:
        image: Input image (H, W) - grayscale or binary
        cast_to_bool: If True, convert grayscale to binary before labeling
        preserve_label: If True, preserve original pixel values as labels
        background: Pixel value to treat as background (not labeled)
        connectivity: Connectivity for connected component labeling (1 or 2)
    
    Returns:
        Tuple of (original image, conversion stats, label image)
    """
    from skimage.measure import label, regionprops
    
    # Work with a copy to avoid modifying input
    working_image = image.copy()
    
    # Cast to binary if requested
    if cast_to_bool:
        working_image = (working_image != background).astype(np.uint8)
    
    if preserve_label:
        # Use the image values directly as labels
        # Ensure background is set to 0
        labels = working_image.astype(np.int32)
        labels[labels == background] = 0
    else:
        # Create binary mask and run connected component labeling
        binary_mask = working_image != background
        labels = label(binary_mask, connectivity=connectivity)
    
    # Ensure labels are proper integer type
    labels = labels.astype(np.int32)
    
    # Calculate statistics
    props = regionprops(labels)
    object_count = len(props)
    
    if object_count > 0:
        areas = [p.area for p in props]
        mean_area = float(np.mean(areas))
        total_area = int(np.sum(areas))
    else:
        mean_area = 0.0
        total_area = 0
    
    stats = ObjectConversionStats(
        slice_index=0,
        object_count=object_count,
        mean_area=mean_area,
        total_area=total_area
    )
    
    return image, stats, labels