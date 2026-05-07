"""
Converted from CellProfiler: DilateObjects
Original: DilateObjects.run

Expands/dilates labeled objects using morphological dilation.
Supports both 2D and 3D objects with configurable structuring elements.
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.core.runtime_values import object_label_dense_array
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.processing.materialization import csv_materializer, segmentation_mask_rois
from openhcs.processing.backends.analysis.region_properties import (
    LabelRegionPropertiesBackendStrategy,
)


class StructuringElementShape(Enum):
    DISK = "disk"
    SQUARE = "square"
    DIAMOND = "diamond"
    OCTAGON = "octagon"
    BALL = "ball"  # 3D
    CUBE = "cube"  # 3D


@dataclass
class DilationStats:
    slice_index: int
    object_count: int
    mean_area_before: float
    mean_area_after: float


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(
    ("dilation_stats", csv_materializer(
        fields=["slice_index", "object_count", "mean_area_before", "mean_area_after"],
        analysis_type="dilation"
    )),
    ("dilated_labels", segmentation_mask_rois())
)
def dilate_objects(
    image: np.ndarray,
    labels: np.ndarray,
    structuring_element_shape: StructuringElementShape = StructuringElementShape.DISK,
    structuring_element_size: int = 1,
) -> Tuple[np.ndarray, DilationStats, np.ndarray]:
    """
    Dilate labeled objects using morphological dilation.
    
    Unlike ExpandOrShrinkObjects, when two objects meet during dilation,
    the object with the larger label number will expand on top of the
    object with the smaller label number.
    
    Args:
        image: Input image (H, W), passed through unchanged
        labels: Label image where each object has a unique integer ID
        structuring_element_shape: Shape of the structuring element
        structuring_element_size: Size/radius of the structuring element
    
    Returns:
        Tuple of (image, dilation_stats, dilated_labels)
    """
    from scipy.ndimage import grey_dilation, maximum_filter
    from skimage.morphology import disk, square, diamond, octagon
    label_array = object_label_dense_array(labels, dtype=np.int32)
    
    # Measure original areas
    props_before = LabelRegionPropertiesBackendStrategy.for_memory_type().measure_2d(
        label_array
    )
    mean_area_before = (
        float(np.mean(props_before.area)) if props_before.label.size else 0.0
    )
    
    # Create structuring element based on shape
    if structuring_element_shape == StructuringElementShape.DISK:
        selem = disk(structuring_element_size)
    elif structuring_element_shape == StructuringElementShape.SQUARE:
        selem = square(2 * structuring_element_size + 1)
    elif structuring_element_shape == StructuringElementShape.DIAMOND:
        selem = diamond(structuring_element_size)
    elif structuring_element_shape == StructuringElementShape.OCTAGON:
        selem = octagon(structuring_element_size, structuring_element_size)
    else:
        selem = disk(structuring_element_size)
    
    # Perform grey dilation on labels
    # Grey dilation with labels means higher label values will expand over lower ones
    # This matches CellProfiler's behavior where larger object numbers expand on top
    dilated_labels = grey_dilation(label_array, footprint=selem)
    
    # Measure dilated areas
    props_after = LabelRegionPropertiesBackendStrategy.for_memory_type().measure_2d(
        dilated_labels.astype(np.int32, copy=False)
    )
    mean_area_after = (
        float(np.mean(props_after.area)) if props_after.label.size else 0.0
    )
    
    stats = DilationStats(
        slice_index=0,
        object_count=int(props_after.label.size),
        mean_area_before=mean_area_before,
        mean_area_after=mean_area_after
    )
    
    return image, stats, dilated_labels.astype(np.float32)


@numpy(contract=ProcessingContract.PURE_3D)
@special_inputs("labels")
@special_outputs(
    ("dilation_stats_3d", csv_materializer(
        fields=["object_count", "mean_volume_before", "mean_volume_after"],
        analysis_type="dilation_3d"
    )),
    ("dilated_labels", segmentation_mask_rois())
)
def dilate_objects_3d(
    image: np.ndarray,
    labels: np.ndarray,
    structuring_element_shape: StructuringElementShape = StructuringElementShape.BALL,
    structuring_element_size: int = 1,
) -> Tuple[np.ndarray, "DilationStats3D", np.ndarray]:
    """
    Dilate labeled objects in 3D using morphological dilation.
    
    Args:
        image: Input image (D, H, W), passed through unchanged
        labels: 3D label image where each object has a unique integer ID
        structuring_element_shape: Shape of the 3D structuring element
        structuring_element_size: Size/radius of the structuring element
    
    Returns:
        Tuple of (image, dilation_stats, dilated_labels)
    """
    from scipy.ndimage import grey_dilation
    from skimage.morphology import ball
    from skimage.measure import regionprops
    label_array = object_label_dense_array(labels, dtype=np.int32)
    
    @dataclass
    class DilationStats3D:
        object_count: int
        mean_volume_before: float
        mean_volume_after: float
    
    # Measure original volumes
    props_before = regionprops(label_array)
    volumes_before = [p.area for p in props_before]  # In 3D, 'area' is actually volume
    mean_volume_before = float(np.mean(volumes_before)) if volumes_before else 0.0
    
    # Create 3D structuring element
    if structuring_element_shape == StructuringElementShape.BALL:
        selem = ball(structuring_element_size)
    elif structuring_element_shape == StructuringElementShape.CUBE:
        size = 2 * structuring_element_size + 1
        selem = np.ones((size, size, size), dtype=bool)
    else:
        selem = ball(structuring_element_size)
    
    # Perform grey dilation on 3D labels
    dilated_labels = grey_dilation(label_array, footprint=selem)
    
    # Measure dilated volumes
    props_after = regionprops(dilated_labels)
    volumes_after = [p.area for p in props_after]
    mean_volume_after = float(np.mean(volumes_after)) if volumes_after else 0.0
    
    stats = DilationStats3D(
        object_count=len(props_after),
        mean_volume_before=mean_volume_before,
        mean_volume_after=mean_volume_after
    )
    
    return image, stats, dilated_labels.astype(np.float32)
