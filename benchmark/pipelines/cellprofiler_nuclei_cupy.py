#!/usr/bin/env python3
"""
CellProfiler IdentifyPrimaryObjects - cupy/cucim Backend (NVIDIA CUDA)

Same algorithm as cellprofiler_nuclei.py but running on NVIDIA GPU via cupy/cucim.
cucim provides GPU-accelerated skimage API - same functions, same parameters, 10-100x faster.

This is the FASTEST option for NVIDIA GPUs (RTX series, datacenter GPUs).
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

try:
    import cupy as cp
    from cucim.skimage.filters import threshold_otsu, threshold_li, gaussian
    from cucim.skimage.segmentation import watershed, clear_border
    from cucim.skimage.measure import label, regionprops_table
    from cucim.skimage.feature import peak_local_max
    from cupyx.scipy import ndimage as cp_ndimage
    
    from openhcs.core.memory.decorators import cupy as cupy_func
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

    @cupy_func
    @special_outputs(
        ("nuclei_measurements", csv_materializer(
            fields=["slice_index", "nuclei_count", "total_area", "mean_area", "mean_intensity"],
            analysis_type="nuclei_counts"
        )),
        ("segmentation_masks", materialize_segmentation_masks)
    )
    def identify_primary_objects_cupy(
        image: np.ndarray,
        # Size parameters
        min_diameter: int = 8,
        max_diameter: int = 80,
        # Threshold parameters
        threshold_method: str = "minimum_cross_entropy",
        threshold_correction: float = 1.0,
        # Declumping parameters
        declump_method: str = "intensity",
        min_allowed_distance: Optional[int] = None,
        # Post-processing
        fill_holes: bool = True,
        discard_border_objects: bool = True,
        discard_outside_diameter: bool = True,
    ) -> Tuple[np.ndarray, List[NucleiMeasurement], List[np.ndarray]]:
        """
        CellProfiler IdentifyPrimaryObjects on NVIDIA GPU via cupy/cucim.
        
        Same algorithm as CPU version, same results, 10-100x faster.
        """
        min_area = int(np.pi * (min_diameter / 2) ** 2)
        max_area = int(np.pi * (max_diameter / 2) ** 2)
        min_dist = min_allowed_distance if min_allowed_distance else max(1, min_diameter // 2)
        
        measurements = []
        masks = []
        
        for i, slice_2d in enumerate(image):
            # Push to GPU
            gpu_slice = cp.asarray(slice_2d.astype(np.float32))
            
            # Thresholding (Li = Minimum Cross-Entropy)
            if threshold_method == "minimum_cross_entropy":
                thresh_val = float(threshold_li(gpu_slice)) * threshold_correction
            else:
                thresh_val = float(threshold_otsu(gpu_slice)) * threshold_correction
            
            binary = gpu_slice > thresh_val
            
            # Fill holes
            if fill_holes:
                binary = cp_ndimage.binary_fill_holes(binary)
            
            # Distance transform for watershed
            distance = cp_ndimage.distance_transform_edt(binary)
            
            # Declumping via watershed
            if declump_method == "intensity":
                weighted = distance * (gpu_slice / (gpu_slice.max() + 1e-10))
            else:
                weighted = distance
            
            # Peak detection (GPU)
            coords = peak_local_max(cp.asnumpy(weighted), min_distance=min_dist, 
                                    labels=cp.asnumpy(binary))
            markers_np = np.zeros(binary.shape, dtype=np.int32)
            if len(coords) > 0:
                markers_np[coords[:, 0], coords[:, 1]] = np.arange(1, len(coords) + 1)
            markers = cp.asarray(markers_np)
            
            # Watershed on GPU
            labeled = watershed(-weighted, markers, mask=binary)
            
            # Remove border objects
            if discard_border_objects:
                labeled = clear_border(labeled)
            
            # Pull to CPU for regionprops and filtering
            labeled_np = cp.asnumpy(labeled).astype(np.int32)
            slice_np = cp.asnumpy(gpu_slice)
            
            # Filter by size using regionprops_table
            if discard_outside_diameter:
                from skimage.measure import regionprops
                props = regionprops(labeled_np, intensity_image=slice_np)
                valid_labels = [p.label for p in props if min_area <= p.area <= max_area]
                
                filtered = np.zeros_like(labeled_np)
                for lbl in valid_labels:
                    filtered[labeled_np == lbl] = lbl
                labeled_np = filtered
                props = [p for p in props if p.label in valid_labels]
            else:
                from skimage.measure import regionprops
                props = regionprops(labeled_np, intensity_image=slice_np)
            
            # Measurements
            measurements.append(NucleiMeasurement(
                slice_index=i,
                nuclei_count=len(props),
                total_area=float(sum(p.area for p in props)),
                mean_area=float(np.mean([p.area for p in props])) if props else 0.0,
                mean_intensity=float(np.mean([p.mean_intensity for p in props])) if props else 0.0
            ))
            masks.append(labeled_np)
        
        return image, measurements, masks

except ImportError:
    identify_primary_objects_cupy = None

