"""
Converted from CellProfiler: ResizeObjects
Original: ResizeObjects module

Resizes object label matrices by a factor or to specific dimensions.
Uses nearest neighbor interpolation to preserve object labels.
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.core.runtime_semantics import (
    ParentChildRelationshipPayload,
    object_label_lineage_payload,
)
from openhcs.core.runtime_values import object_label_dense_array
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_outputs, special_inputs
from openhcs.processing.materialization import csv_materializer, segmentation_mask_rois
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum

from benchmark.cellprofiler_library.functions.spatial_axes import (
    trailing_spatial_factors,
    trailing_spatial_target_shape,
)


class ResizeMethod(Enum):
    DIMENSIONS = ("dimensions", "to_size", "manual")
    FACTOR = ("factor", "by_factor")

    def __new__(cls, value: str, *cellprofiler_literals: str):
        member = object.__new__(cls)
        member._value_ = value
        member.cellprofiler_literals = cellprofiler_literals
        return member


@dataclass
class ResizeObjectsStats:
    slice_index: int
    original_height: int
    original_width: int
    new_height: int
    new_width: int
    object_count: int


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(
    ("resize_stats", csv_materializer(
        fields=["slice_index", "original_height", "original_width", "new_height", "new_width", "object_count"],
        analysis_type="resize_objects"
    )),
    "parent_child_relationship",
    ("resized_labels", segmentation_mask_rois())
)
def resize_objects(
    image: np.ndarray,
    labels: np.ndarray,
    method: ResizeMethod = ResizeMethod.FACTOR,
    factor_x: float = 0.25,
    factor_y: float = 0.25,
    factor_z: float = 1.0,
    width: int = 100,
    height: int = 100,
    planes: int = 10,
) -> Tuple[np.ndarray, ResizeObjectsStats, ParentChildRelationshipPayload, np.ndarray]:
    """
    Resize object label matrices by a factor or to specific dimensions.
    
    Uses nearest neighbor interpolation to preserve object labels after resizing.
    Useful for processing large data to reduce computation time - downsize for
    segmentation, then upsize back to original dimensions for measurements.
    
    Args:
        image: Input image array with shape (H, W)
        labels: Object label matrix with shape (H, W)
        method: Resize method - DIMENSIONS for specific size, FACTOR for scaling
        factor_x: X scaling factor (used if method=FACTOR). <1 shrinks, >1 enlarges
        factor_y: Y scaling factor (used if method=FACTOR). <1 shrinks, >1 enlarges
        width: Target width in pixels (used if method=DIMENSIONS)
        height: Target height in pixels (used if method=DIMENSIONS)
    
    Returns:
        Tuple of (original image, resize statistics, resized labels)
    """
    from scipy.ndimage import zoom
    labels = object_label_dense_array(labels, dtype=np.int32)
    
    original_shape = labels.shape
    
    method = coerce_cellprofiler_enum(ResizeMethod, method)

    if method == ResizeMethod.DIMENSIONS:
        target_size = _resize_objects_target_shape(
            labels.shape,
            planes=planes,
            height=height,
            width=width,
        )
        zoom_factors = np.divide(np.multiply(1.0, target_size), labels.shape)
    else:
        zoom_factors = _resize_objects_zoom_factors(
            labels.ndim,
            factor_z=factor_z,
            factor_y=factor_y,
            factor_x=factor_x,
        )
    resized_labels = zoom(labels, zoom_factors, order=0, mode="nearest")
    
    # Ensure labels remain integer type
    resized_labels = resized_labels.astype(np.int32)
    
    # Count unique objects (excluding background 0)
    unique_labels = np.unique(resized_labels)
    object_count = len(unique_labels[unique_labels > 0])
    
    stats = ResizeObjectsStats(
        slice_index=0,
        original_height=original_shape[-2],
        original_width=original_shape[-1],
        new_height=resized_labels.shape[-2],
        new_width=resized_labels.shape[-1],
        object_count=object_count
    )
    
    relationship = object_label_lineage_payload(labels, resized_labels)
    return image, stats, relationship, resized_labels

def _resize_objects_target_shape(
    shape: tuple[int, ...],
    *,
    planes: int,
    height: int,
    width: int,
) -> tuple[int, ...]:
    spatial_shape = (planes, height, width) if len(shape) >= 3 else (height, width)
    return trailing_spatial_target_shape(shape, spatial_shape)


def _resize_objects_zoom_factors(
    ndim: int,
    *,
    factor_z: float,
    factor_y: float,
    factor_x: float,
) -> tuple[float, ...]:
    spatial_factors = (
        (factor_z, factor_y, factor_x)
        if ndim >= 3
        else (factor_y, factor_x)
    )
    return trailing_spatial_factors(ndim, spatial_factors)


@numpy(contract=ProcessingContract.PURE_3D)
@special_inputs("labels")
@special_outputs(
    ("resize_stats_3d", csv_materializer(
        fields=["original_depth", "original_height", "original_width", 
                "new_depth", "new_height", "new_width", "object_count"],
        analysis_type="resize_objects_3d"
    )),
    "parent_child_relationship",
    ("resized_labels", segmentation_mask_rois())
)
def resize_objects_3d(
    image: np.ndarray,
    labels: np.ndarray,
    method: ResizeMethod = ResizeMethod.FACTOR,
    factor_x: float = 0.25,
    factor_y: float = 0.25,
    factor_z: float = 0.25,
    width: int = 100,
    height: int = 100,
    planes: int = 10,
) -> Tuple[np.ndarray, dict, ParentChildRelationshipPayload, np.ndarray]:
    """
    Resize 3D object label matrices by a factor or to specific dimensions.
    
    Uses nearest neighbor interpolation to preserve object labels after resizing.
    
    Args:
        image: Input image array with shape (D, H, W)
        labels: Object label matrix with shape (D, H, W)
        method: Resize method - DIMENSIONS for specific size, FACTOR for scaling
        factor_x: X scaling factor (used if method=FACTOR)
        factor_y: Y scaling factor (used if method=FACTOR)
        factor_z: Z scaling factor (used if method=FACTOR)
        width: Target width in pixels (used if method=DIMENSIONS)
        height: Target height in pixels (used if method=DIMENSIONS)
        planes: Target depth/planes (used if method=DIMENSIONS)
    
    Returns:
        Tuple of (original image, resize statistics dict, resized labels)
    """
    from scipy.ndimage import zoom
    labels = object_label_dense_array(labels, dtype=np.int32)
    
    original_shape = labels.shape
    
    if method == ResizeMethod.DIMENSIONS:
        # Resize to specific dimensions
        target_size = (planes, height, width)
        zoom_factors = np.divide(np.multiply(1.0, target_size), labels.shape)
        resized_labels = zoom(labels, zoom_factors, order=0, mode="nearest")
    else:
        # Resize by factor
        zoom_factors = (factor_z, factor_y, factor_x)
        resized_labels = zoom(labels, zoom_factors, order=0, mode="nearest")
    
    # Ensure labels remain integer type
    resized_labels = resized_labels.astype(np.int32)
    
    # Count unique objects (excluding background 0)
    unique_labels = np.unique(resized_labels)
    object_count = len(unique_labels[unique_labels > 0])
    
    stats = {
        "original_depth": original_shape[0],
        "original_height": original_shape[1],
        "original_width": original_shape[2],
        "new_depth": resized_labels.shape[0],
        "new_height": resized_labels.shape[1],
        "new_width": resized_labels.shape[2],
        "object_count": object_count
    }
    
    relationship = object_label_lineage_payload(labels, resized_labels)
    return image, stats, relationship, resized_labels
