"""
Converted from CellProfiler: IdentifyObjectsManually
Original: IdentifyObjectsManually.run

Note: This module in CellProfiler requires interactive user input via a GUI dialog.
In OpenHCS, we provide a placeholder that returns empty labels since true interactive
manual segmentation requires a UI context that doesn't exist in batch processing.

For actual manual annotation, use external tools (napari, Fiji, etc.) and import
the resulting label images.
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.processing.materialization import csv_materializer
from openhcs.processing.backends.analysis.cell_counting_cpu import materialize_segmentation_masks


@dataclass
class ManualObjectStats:
    """Statistics for manually identified objects."""
    slice_index: int
    object_count: int
    mean_area: float
    mean_centroid_x: float
    mean_centroid_y: float


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(
    ("object_stats", csv_materializer(
        fields=["slice_index", "object_count", "mean_area", "mean_centroid_x", "mean_centroid_y"],
        analysis_type="manual_objects"
    )),
    ("labels", materialize_segmentation_masks)
)
def identify_objects_manually(
    image: np.ndarray,
    labels_input: np.ndarray = None,
    objects_name: str = "Cells",
) -> Tuple[np.ndarray, ManualObjectStats, np.ndarray]:
    """
    Placeholder for manual object identification.
    
    In CellProfiler, this module displays an interactive UI where users can
    manually outline objects using mouse tools (outline, zoom, erase).
    
    In OpenHCS batch processing context, this function:
    1. If labels_input is provided (pre-annotated), uses those labels
    2. Otherwise, returns empty labels (no objects)
    
    For actual manual annotation workflows:
    - Use napari, Fiji, or other annotation tools to create label images
    - Import the label images as a separate channel/input
    - Pass them via labels_input parameter
    
    Args:
        image: Input image to display for annotation, shape (H, W)
        labels_input: Optional pre-annotated label image, shape (H, W).
                     If None, returns empty labels.
        objects_name: Name for the identified objects (metadata only)
    
    Returns:
        Tuple of:
        - Original image (unchanged)
        - ManualObjectStats dataclass with object measurements
        - Label image where each object has a unique integer ID
    
    Note:
        This module cannot be used in batch mode in CellProfiler.
        The OpenHCS version provides a passthrough for pre-annotated labels
        or returns empty results for pipeline compatibility.
    """
    from skimage.measure import regionprops, label as relabel
    
    h, w = image.shape[:2] if image.ndim >= 2 else (image.shape[0], 1)
    
    # Use provided labels or create empty labels
    if labels_input is not None:
        # Ensure labels are integer type and properly formatted
        labels = np.asarray(labels_input, dtype=np.int32)
        if labels.shape != (h, w):
            # Resize if needed
            labels = np.zeros((h, w), dtype=np.int32)
        # Relabel to ensure consecutive integers
        if labels.max() > 0:
            labels = relabel(labels > 0).astype(np.int32)
    else:
        # No labels provided - return empty (no objects identified)
        # In interactive mode, this would open a GUI
        labels = np.zeros((h, w), dtype=np.int32)
    
    # Calculate object statistics
    object_count = int(labels.max())
    
    if object_count > 0:
        props = regionprops(labels)
        areas = [p.area for p in props]
        centroids_y = [p.centroid[0] for p in props]
        centroids_x = [p.centroid[1] for p in props]
        
        mean_area = float(np.mean(areas))
        mean_centroid_x = float(np.mean(centroids_x))
        mean_centroid_y = float(np.mean(centroids_y))
    else:
        mean_area = 0.0
        mean_centroid_x = 0.0
        mean_centroid_y = 0.0
    
    stats = ManualObjectStats(
        slice_index=0,
        object_count=object_count,
        mean_area=mean_area,
        mean_centroid_x=mean_centroid_x,
        mean_centroid_y=mean_centroid_y
    )
    
    # Return image unchanged, stats, and labels
    return image, stats, labels