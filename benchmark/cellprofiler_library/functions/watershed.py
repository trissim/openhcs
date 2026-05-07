"""
Converted from CellProfiler: Watershed
Original: watershed
"""

import numpy as np
from typing import Tuple, Literal
from dataclasses import dataclass
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.cellprofiler._backend import CellProfilerBackendProvider
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.processing.materialization import csv_materializer, segmentation_mask_rois


@dataclass
class WatershedStats:
    slice_index: int
    object_count: int
    mean_area: float


def cellprofiler_legacy_watershed(
    image: np.ndarray,
    *,
    markers: np.ndarray,
    mask: np.ndarray,
    connectivity: int | np.ndarray = 1,
    backend_provider: CellProfilerBackendProvider | None = None,
) -> np.ndarray:
    """Run CellProfiler 4.2/skimage 0.18 watershed semantics.

    Newer scikit-image raises a queued neighbor's priority to at least the
    source pixel priority. CellProfiler 4.2 did not, and that changes basin
    ownership for tied/near-tied object boundaries by a few pixels.
    """
    from openhcs.processing.backends.cellprofiler.watershed import (
        cellprofiler_legacy_watershed as _cellprofiler_legacy_watershed,
    )

    return _cellprofiler_legacy_watershed(
        image,
        markers=markers,
        mask=mask,
        connectivity=connectivity,
        backend_provider=backend_provider,
    )


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(
    ("watershed_stats", csv_materializer(fields=["slice_index", "object_count", "mean_area"])),
    ("labels", segmentation_mask_rois())
)
def watershed(
    image: np.ndarray,
    watershed_method: Literal["distance", "intensity", "markers"] = "distance",
    declump_method: Literal["shape", "intensity"] = "shape",
    seed_method: Literal["local", "regional"] = "local",
    max_seeds: int = -1,
    downsample: int = 1,
    min_distance: int = 1,
    min_intensity: float = 0.0,
    footprint: int = 8,
    connectivity: int = 1,
    compactness: float = 0.0,
    exclude_border: bool = False,
    watershed_line: bool = False,
    gaussian_sigma: float = 0.0,
    structuring_element: Literal[
        "ball", "cube", "diamond", "disk", "octahedron", "square", "star"
    ] = "disk",
    structuring_element_size: int = 1,
) -> Tuple[np.ndarray, WatershedStats, np.ndarray]:
    """
    Apply watershed segmentation to separate touching objects.
    
    Args:
        image: Input binary or grayscale image (H, W)
        watershed_method: Method for watershed - 'distance' uses distance transform,
                         'intensity' uses intensity image, 'markers' uses marker image
        declump_method: Method for declumping - 'shape' or 'intensity'
        seed_method: Seed detection method - 'local' for local maxima, 'regional' for regional
        max_seeds: Maximum number of seeds (-1 for unlimited)
        downsample: Downsampling factor for speed
        min_distance: Minimum distance between seeds
        min_intensity: Minimum intensity for seeds
        footprint: Footprint size for local maxima detection
        connectivity: Connectivity for watershed (1 or 2)
        compactness: Compactness parameter for watershed
        exclude_border: Whether to exclude objects touching border
        watershed_line: Whether to draw watershed lines between objects
        gaussian_sigma: Sigma for Gaussian smoothing (0 for no smoothing)
        structuring_element: Shape of structuring element for morphological operations
        structuring_element_size: Size of structuring element
    
    Returns:
        Tuple of (original image, watershed statistics, labeled image)
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter, label as ndi_label
    from skimage.segmentation import watershed as skimage_watershed
    from skimage.feature import peak_local_max
    from skimage.measure import regionprops
    from skimage.segmentation import clear_border
    
    # Handle input - assume binary or use threshold
    if image.dtype == bool:
        binary = image.astype(np.float32)
    else:
        # Normalize and threshold
        img_norm = (image - image.min()) / (image.max() - image.min() + 1e-10)
        binary = (img_norm > 0.5).astype(np.float32)
    
    # Apply Gaussian smoothing if specified
    if gaussian_sigma > 0:
        binary = gaussian_filter(binary, gaussian_sigma)
        binary = (binary > 0.5).astype(np.float32)
    
    # Compute distance transform for watershed
    if watershed_method == "distance":
        distance = distance_transform_edt(binary)
    elif watershed_method == "intensity":
        # Use inverted intensity as distance
        distance = 1.0 - (image - image.min()) / (image.max() - image.min() + 1e-10)
        distance = distance * binary
    else:
        # Default to distance transform
        distance = distance_transform_edt(binary)
    
    # Find seeds/markers
    if seed_method == "local":
        # Local maxima detection
        coords = peak_local_max(
            distance,
            min_distance=min_distance,
            footprint=np.ones((footprint,) * image.ndim),
            labels=binary.astype(int),
            exclude_border=exclude_border
        )
        
        # Limit seeds if specified
        if max_seeds > 0 and len(coords) > max_seeds:
            distances_at_coords = distance[tuple(coords.T)]
            top_indices = np.argsort(distances_at_coords)[-max_seeds:]
            coords = coords[top_indices]
        
        # Create marker image
        markers = np.zeros_like(binary, dtype=np.int32)
        for i, coord in enumerate(coords):
            markers[tuple(coord)] = i + 1
    else:
        # Regional maxima - use h-maxima approach
        from skimage.morphology import reconstruction
        h = min_intensity if min_intensity > 0 else 0.1
        seed = distance - h
        seed = np.clip(seed, 0, None)
        dilated = reconstruction(seed, distance, method='dilation')
        markers_binary = (distance - dilated) > 0
        markers, _ = ndi_label(markers_binary)
    
    # Apply watershed
    labels = skimage_watershed(
        -distance,
        markers=markers,
        mask=binary.astype(bool),
        connectivity=connectivity,
        compactness=compactness,
        watershed_line=watershed_line
    )
    
    # Exclude border objects if specified
    if exclude_border:
        labels = clear_border(labels)
    
    # Relabel to ensure consecutive labels
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]
    new_labels = np.zeros_like(labels)
    for new_label, old_label in enumerate(unique_labels, start=1):
        new_labels[labels == old_label] = new_label
    labels = new_labels
    
    # Compute statistics
    props = regionprops(labels)
    object_count = len(props)
    mean_area = np.mean([p.area for p in props]) if props else 0.0
    
    stats = WatershedStats(
        slice_index=0,
        object_count=object_count,
        mean_area=float(mean_area)
    )
    
    return image, stats, labels.astype(np.int32)
