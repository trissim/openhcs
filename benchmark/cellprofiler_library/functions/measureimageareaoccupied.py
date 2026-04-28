"""
Converted from CellProfiler: MeasureImageAreaOccupied
Measures the total area in an image that is occupied by objects or foreground.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.processing.materialization import csv_materializer


class OperandChoice(Enum):
    BINARY_IMAGE = "binary_image"
    OBJECTS = "objects"


@dataclass
class AreaOccupiedMeasurement:
    """Measurements for area occupied analysis."""
    slice_index: int
    area_occupied: float
    perimeter: float
    total_area: float


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(("area_measurements", csv_materializer(
    fields=["slice_index", "area_occupied", "perimeter", "total_area"],
    analysis_type="area_occupied"
)))
def measure_image_area_occupied_binary(
    image: np.ndarray,
) -> Tuple[np.ndarray, AreaOccupiedMeasurement]:
    """
    Measure area occupied by foreground in a binary image.
    
    Args:
        image: Binary image (H, W) where foreground > 0
        
    Returns:
        Tuple of (original image, AreaOccupiedMeasurement)
    """
    from skimage.measure import perimeter as measure_perimeter
    
    # Calculate area occupied (number of foreground pixels)
    binary_mask = image > 0
    area_occupied = float(np.sum(binary_mask))
    
    # Calculate perimeter
    if area_occupied > 0:
        perimeter_value = float(measure_perimeter(binary_mask))
    else:
        perimeter_value = 0.0
    
    # Total area is the total number of pixels
    total_area = float(np.prod(image.shape))
    
    measurement = AreaOccupiedMeasurement(
        slice_index=0,
        area_occupied=area_occupied,
        perimeter=perimeter_value,
        total_area=total_area
    )
    
    return image, measurement


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(("area_measurements", csv_materializer(
    fields=["slice_index", "area_occupied", "perimeter", "total_area"],
    analysis_type="area_occupied"
)))
def measure_image_area_occupied_objects(
    image: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, AreaOccupiedMeasurement]:
    """
    Measure area occupied by labeled objects.
    
    Args:
        image: Intensity image (H, W)
        labels: Label image from segmentation (H, W)
        
    Returns:
        Tuple of (original image, AreaOccupiedMeasurement)
    """
    from skimage.measure import regionprops
    
    # Get region properties
    region_properties = regionprops(labels.astype(np.int32))
    
    # Calculate area occupied (sum of all object areas)
    area_occupied = float(np.sum([region.area for region in region_properties]))
    
    # Calculate perimeter (sum of all object perimeters)
    if area_occupied > 0:
        perimeter_value = float(np.sum(
            [np.round(region.perimeter) for region in region_properties]
        ))
    else:
        perimeter_value = 0.0
    
    # Total area is the total number of pixels
    total_area = float(np.prod(labels.shape))
    
    measurement = AreaOccupiedMeasurement(
        slice_index=0,
        area_occupied=area_occupied,
        perimeter=perimeter_value,
        total_area=total_area
    )
    
    return image, measurement


@dataclass
class VolumeOccupiedMeasurement:
    """Measurements for volume occupied analysis (3D)."""
    volume_occupied: float
    surface_area: float
    total_volume: float


def _compute_surface_area(label_image: np.ndarray, spacing: Optional[Tuple[float, ...]] = None) -> float:
    """
    Compute surface area of labeled regions using marching cubes.
    
    Args:
        label_image: 3D label image
        spacing: Voxel spacing (z, y, x)
        
    Returns:
        Total surface area
    """
    from skimage.measure import marching_cubes, mesh_surface_area
    
    if spacing is None:
        spacing = (1.0,) * label_image.ndim
    
    unique_labels = np.unique(label_image)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background
    
    if len(unique_labels) == 0:
        return 0.0
    
    total_surface = 0.0
    for label in unique_labels:
        binary_mask = (label_image == label).astype(np.float32)
        try:
            verts, faces, _, _ = marching_cubes(
                binary_mask, spacing=spacing, level=0.5, method="lorensen"
            )
            total_surface += mesh_surface_area(verts, faces)
        except (ValueError, RuntimeError):
            # marching_cubes can fail on very small objects
            continue
    
    return float(np.round(total_surface))


@numpy(contract=ProcessingContract.PURE_3D)
@special_outputs(("volume_measurements", csv_materializer(
    fields=["volume_occupied", "surface_area", "total_volume"],
    analysis_type="volume_occupied"
)))
def measure_image_volume_occupied_binary(
    image: np.ndarray,
    spacing: Optional[Tuple[float, float, float]] = None,
) -> Tuple[np.ndarray, VolumeOccupiedMeasurement]:
    """
    Measure volume occupied by foreground in a 3D binary image.
    
    Args:
        image: 3D binary image (D, H, W) where foreground > 0
        spacing: Voxel spacing (z, y, x) for surface area calculation
        
    Returns:
        Tuple of (original image, VolumeOccupiedMeasurement)
    """
    # Calculate volume occupied (number of foreground voxels)
    binary_mask = image > 0
    volume_occupied = float(np.sum(binary_mask))
    
    # Calculate surface area
    if volume_occupied > 0:
        surface_area_value = _compute_surface_area(
            binary_mask.astype(np.int32), spacing=spacing
        )
    else:
        surface_area_value = 0.0
    
    # Total volume is the total number of voxels
    total_volume = float(np.prod(image.shape))
    
    measurement = VolumeOccupiedMeasurement(
        volume_occupied=volume_occupied,
        surface_area=surface_area_value,
        total_volume=total_volume
    )
    
    return image, measurement


@numpy(contract=ProcessingContract.PURE_3D)
@special_inputs("labels")
@special_outputs(("volume_measurements", csv_materializer(
    fields=["volume_occupied", "surface_area", "total_volume"],
    analysis_type="volume_occupied"
)))
def measure_image_volume_occupied_objects(
    image: np.ndarray,
    labels: np.ndarray,
    spacing: Optional[Tuple[float, float, float]] = None,
) -> Tuple[np.ndarray, VolumeOccupiedMeasurement]:
    """
    Measure volume occupied by labeled objects in 3D.
    
    Args:
        image: 3D intensity image (D, H, W)
        labels: 3D label image from segmentation (D, H, W)
        spacing: Voxel spacing (z, y, x) for surface area calculation
        
    Returns:
        Tuple of (original image, VolumeOccupiedMeasurement)
    """
    from skimage.measure import regionprops
    
    # Get region properties
    region_properties = regionprops(labels.astype(np.int32))
    
    # Calculate volume occupied (sum of all object volumes)
    volume_occupied = float(np.sum([region.area for region in region_properties]))
    
    # Calculate surface area
    if volume_occupied > 0:
        surface_area_value = _compute_surface_area(
            labels.astype(np.int32), spacing=spacing
        )
    else:
        surface_area_value = 0.0
    
    # Total volume is the total number of voxels
    total_volume = float(np.prod(labels.shape))
    
    measurement = VolumeOccupiedMeasurement(
        volume_occupied=volume_occupied,
        surface_area=surface_area_value,
        total_volume=total_volume
    )
    
    return image, measurement