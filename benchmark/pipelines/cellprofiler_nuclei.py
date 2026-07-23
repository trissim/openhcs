#!/usr/bin/env python3
"""
CellProfiler IdentifyPrimaryObjects - Exact Replication in OpenHCS

This module provides an exact reimplementation of CellProfiler's IdentifyPrimaryObjects
algorithm for nuclei segmentation, using the same algorithmic steps:

1. Smoothing (Gaussian blur with auto-calculated sigma)
2. Thresholding (Minimum Cross-Entropy or Otsu)
3. Declumping (Shape-based or Intensity-based watershed)
4. Fill holes
5. Filter by size
6. Remove border objects

Reference: CellProfiler source code - cellprofiler/modules/identifyprimaryobjects.py
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu, gaussian
from skimage.segmentation import watershed, clear_border
from skimage.measure import label, regionprops
from skimage.feature import peak_local_max

from openhcs.core.memory.decorators import numpy as numpy_func
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.processing.materialization import csv_materializer
from openhcs.processing.backends.analysis.cell_counting_cpu import materialize_segmentation_masks


@dataclass
class NucleiMeasurement:
    """Per-slice nuclei measurements matching CellProfiler output."""
    slice_index: int
    nuclei_count: int
    total_area: float
    mean_area: float
    mean_intensity: float


def minimum_cross_entropy_threshold(image: np.ndarray) -> float:
    """
    Minimum Cross-Entropy thresholding (Li's method).
    This is what CellProfiler calls "Minimum Cross-Entropy" in IdentifyPrimaryObjects.
    """
    from skimage.filters import threshold_li
    return threshold_li(image)


def declump_intensity(binary: np.ndarray, intensity: np.ndarray, 
                      min_distance: int = 7) -> np.ndarray:
    """
    Intensity-based declumping (CellProfiler's "Intensity" method).
    Uses intensity peaks as watershed seeds.
    """
    distance = ndimage.distance_transform_edt(binary)
    # Weight by intensity for intensity-based declumping
    weighted = distance * (intensity / intensity.max() if intensity.max() > 0 else 1)
    
    coords = peak_local_max(weighted, min_distance=min_distance, labels=binary)
    mask = np.zeros(binary.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers = label(mask)
    
    return watershed(-weighted, markers, mask=binary)


def declump_shape(binary: np.ndarray, min_distance: int = 7) -> np.ndarray:
    """
    Shape-based declumping (CellProfiler's "Shape" method).
    Uses distance transform peaks as watershed seeds.
    """
    distance = ndimage.distance_transform_edt(binary)
    coords = peak_local_max(distance, min_distance=min_distance, labels=binary)
    mask = np.zeros(binary.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers = label(mask)
    
    return watershed(-distance, markers, mask=binary)


@numpy_func
@special_outputs(
    ("nuclei_measurements", csv_materializer(
        fields=["slice_index", "nuclei_count", "total_area", "mean_area", "mean_intensity"],
        analysis_type="nuclei_counts"
    )),
    ("segmentation_masks", materialize_segmentation_masks)
)
def identify_primary_objects(
    image: np.ndarray,
    # Size parameters (CellProfiler: "Typical diameter of objects")
    min_diameter: int = 8,
    max_diameter: int = 80,
    # Threshold parameters
    threshold_method: str = "minimum_cross_entropy",  # or "otsu"
    threshold_correction: float = 1.0,
    # Declumping parameters
    declump_method: str = "intensity",  # or "shape"
    smoothing_filter_size: Optional[int] = None,  # None = auto-calculate
    min_allowed_distance: Optional[int] = None,   # None = auto-calculate
    # Post-processing
    fill_holes: bool = True,
    discard_border_objects: bool = True,
    discard_outside_diameter: bool = True,
) -> Tuple[np.ndarray, List[NucleiMeasurement], List[np.ndarray]]:
    """
    CellProfiler IdentifyPrimaryObjects - exact algorithm replication.
    
    Args:
        image: 3D array (slices, height, width)
        min_diameter: Minimum object diameter in pixels
        max_diameter: Maximum object diameter in pixels  
        threshold_method: "minimum_cross_entropy" (Li) or "otsu"
        threshold_correction: Multiply threshold by this factor
        declump_method: "intensity" or "shape"
        smoothing_filter_size: Gaussian sigma (None = diameter/3.5)
        min_allowed_distance: Min distance between peaks (None = diameter/2)
        fill_holes: Fill holes in objects
        discard_border_objects: Remove objects touching image border
        discard_outside_diameter: Discard objects outside diameter range
    """
    # Convert diameter to area (assuming circular objects)
    min_area = int(np.pi * (min_diameter / 2) ** 2)
    max_area = int(np.pi * (max_diameter / 2) ** 2)
    
    # Auto-calculate smoothing if not specified (CellProfiler default)
    sigma = smoothing_filter_size if smoothing_filter_size else min_diameter / 3.5
    min_dist = min_allowed_distance if min_allowed_distance else max(1, min_diameter // 2)
    
    measurements = []
    masks = []
    
    for i, slice_2d in enumerate(image):
        # Step 1: Smoothing
        smoothed = gaussian(slice_2d.astype(float), sigma=sigma)
        
        # Step 2: Thresholding
        if threshold_method == "minimum_cross_entropy":
            thresh_val = minimum_cross_entropy_threshold(smoothed) * threshold_correction
        else:
            thresh_val = threshold_otsu(smoothed) * threshold_correction
        binary = smoothed > thresh_val
        
        # Step 3: Fill holes (before declumping if specified)
        if fill_holes:
            binary = ndimage.binary_fill_holes(binary)
        
        # Step 4: Declumping
        if declump_method == "intensity":
            labeled = declump_intensity(binary, smoothed, min_distance=min_dist)
        else:
            labeled = declump_shape(binary, min_distance=min_dist)
        
        # Step 5: Remove border objects
        if discard_border_objects:
            labeled = clear_border(labeled)
        
        # Step 6: Filter by size
        if discard_outside_diameter:
            props = regionprops(labeled, intensity_image=slice_2d)
            valid_labels = [p.label for p in props if min_area <= p.area <= max_area]
            filtered = np.zeros_like(labeled)
            for lbl in valid_labels:
                filtered[labeled == lbl] = lbl
            labeled = filtered
            props = [p for p in props if p.label in valid_labels]
        else:
            props = regionprops(labeled, intensity_image=slice_2d)
        
        # Compute measurements
        measurements.append(NucleiMeasurement(
            slice_index=i,
            nuclei_count=len(props),
            total_area=float(sum(p.area for p in props)),
            mean_area=float(np.mean([p.area for p in props])) if props else 0.0,
            mean_intensity=float(np.mean([p.mean_intensity for p in props])) if props else 0.0
        ))
        masks.append(labeled)
    
    return image, measurements, masks

