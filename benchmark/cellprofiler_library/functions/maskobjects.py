"""Converted from CellProfiler: MaskObjects

Removes objects outside of a specified region or regions.
This module allows you to delete the objects or portions of objects that
are outside of a region (mask) you specify.
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.core.runtime_semantics import (
    ParentChildRelationshipPayload,
    aligned_dense_object_label_arrays,
    project_dense_object_label_stack,
)
from openhcs.processing.backends.cellprofiler._backend import CellProfilerBackendProvider
from openhcs.processing.backends.cellprofiler.relationships import (
    ObjectRelationshipBackendStrategy,
)
from openhcs.processing.materialization import csv_materializer, segmentation_mask_rois

from benchmark.cellprofiler_library.functions._enum import _coerce_function_enum


class MaskChoice(Enum):
    OBJECTS = "objects"
    IMAGE = "image"


class OverlapHandling(Enum):
    MASK = "keep_overlapping_region"  # Keep only overlapping portion
    KEEP = "keep"  # Keep whole object if any overlap
    REMOVE = "remove"  # Remove if any part outside
    REMOVE_PERCENTAGE = "remove_depending_on_overlap"  # Remove based on fraction


class NumberingChoice(Enum):
    RENUMBER = "renumber"  # Consecutive numbering
    RETAIN = "retain"  # Keep original labels


@dataclass
class MaskObjectsStats:
    slice_index: int
    original_object_count: int
    remaining_object_count: int
    objects_removed: int


@numpy
@special_inputs("labels", "mask")
@special_outputs(
    ("mask_stats", csv_materializer(
        fields=["slice_index", "original_object_count", "remaining_object_count", "objects_removed"],
        analysis_type="mask_objects"
    )),
    "object_relationships",
    ("masked_labels", segmentation_mask_rois())
)
def mask_objects(
    image: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    overlap_handling: OverlapHandling = OverlapHandling.MASK,
    overlap_fraction: float = 0.5,
    numbering: NumberingChoice = NumberingChoice.RENUMBER,
    invert_mask: bool = False,
    relationship_backend_provider: CellProfilerBackendProvider | None = None,
) -> Tuple[np.ndarray, MaskObjectsStats, ParentChildRelationshipPayload, np.ndarray]:
    """
    Mask objects based on a binary mask or masking objects.
    
    Args:
        image: Input image, shape (D, H, W) - passed through unchanged
        labels: Label image of objects to mask, shape (H, W)
        mask: Binary mask or label image defining masking region, shape (H, W)
        overlap_handling: How to handle partially masked objects
            - MASK: Keep only the overlapping portion
            - KEEP: Keep whole object if any part overlaps
            - REMOVE: Remove object if any part is outside mask
            - REMOVE_PERCENTAGE: Remove based on overlap fraction
        overlap_fraction: Minimum fraction of object that must overlap (for REMOVE_PERCENTAGE)
        numbering: Whether to renumber objects consecutively or retain original labels
        invert_mask: If True, use the inverse of the mask
    
    Returns:
        Tuple of (image, stats, parent-child relationship, masked_labels)
    """
    import scipy.ndimage as ndi

    overlap_handling = _coerce_function_enum(OverlapHandling, overlap_handling)
    numbering = _coerce_function_enum(NumberingChoice, numbering)
    
    try:
        labels = project_dense_object_label_stack(labels).astype(np.int32, copy=False)
    except ValueError as exc:
        raise ValueError(
            "MaskObjects could not project object labels; "
            f"image shape={getattr(image, 'shape', None)!r}, "
            f"labels shape={getattr(labels, 'shape', None)!r}, "
            f"mask shape={getattr(mask, 'shape', None)!r}."
        ) from exc
    _aligned_labels, mask = aligned_dense_object_label_arrays(labels, mask)
    labels = _aligned_labels.astype(np.int32, copy=False)

    # Handle mask - convert label image to binary if needed
    if mask.max() > 1:
        binary_mask = mask > 0
    else:
        binary_mask = mask.astype(bool)
    
    if invert_mask:
        binary_mask = ~binary_mask
    
    # Make a copy of labels to modify
    masked_labels = labels.copy()
    nobjects = int(np.max(labels))
    
    if nobjects == 0:
        # No objects to mask
        stats = MaskObjectsStats(
            slice_index=0,
            original_object_count=0,
            remaining_object_count=0,
            objects_removed=0
        )
        relationships = ParentChildRelationshipPayload(parent_ids=(), child_ids=())
        return image, stats, relationships, masked_labels
    
    # CellProfiler size_similarly semantics: masks smaller than labels create
    # manufactured false pixels; larger masks are cropped to label geometry.
    binary_mask = _size_binary_mask_like_labels(labels, binary_mask)
    
    # Apply mask according to overlap choice
    if overlap_handling == OverlapHandling.MASK:
        # Keep only overlapping region
        masked_labels = masked_labels * binary_mask.astype(masked_labels.dtype)
    else:
        # Calculate pixel counts within mask for each object
        object_indices = np.arange(1, nobjects + 1, dtype=np.int32)
        
        pixel_counts = ndi.sum(
            binary_mask.astype(np.float64),
            labels,
            object_indices
        )
        pixel_counts = np.atleast_1d(pixel_counts)
        
        if overlap_handling == OverlapHandling.KEEP:
            # Keep if any overlap
            keep = pixel_counts > 0
        else:
            # Calculate total pixels per object
            total_pixels = ndi.sum(
                np.ones(labels.shape, dtype=np.float64),
                labels,
                object_indices
            )
            total_pixels = np.atleast_1d(total_pixels)
            
            if overlap_handling == OverlapHandling.REMOVE:
                # Keep only if fully inside mask
                keep = pixel_counts == total_pixels
            elif overlap_handling == OverlapHandling.REMOVE_PERCENTAGE:
                # Keep if fraction overlaps
                with np.errstate(divide='ignore', invalid='ignore'):
                    fractions = np.where(total_pixels > 0, pixel_counts / total_pixels, 0)
                keep = fractions >= overlap_fraction
            else:
                keep = pixel_counts > 0
        
        # Create lookup table: prepend False for background (label 0)
        keep_lookup = np.concatenate([[False], keep])
        
        # Remove objects that don't meet criteria
        masked_labels[~keep_lookup[labels]] = 0
    
    # Renumber if requested
    if numbering == NumberingChoice.RENUMBER:
        unique_labels = np.unique(masked_labels[masked_labels != 0])
        if len(unique_labels) > 0:
            indexer = np.zeros(nobjects + 1, dtype=np.int32)
            indexer[unique_labels] = np.arange(1, len(unique_labels) + 1, dtype=np.int32)
            masked_labels = indexer[masked_labels]
            remaining_count = len(unique_labels)
        else:
            remaining_count = 0
    else:
        remaining_count = len(np.unique(masked_labels[masked_labels != 0]))
    
    stats = MaskObjectsStats(
        slice_index=0,
        original_object_count=nobjects,
        remaining_object_count=remaining_count,
        objects_removed=nobjects - remaining_count
    )
    relationship_backend = ObjectRelationshipBackendStrategy.for_memory_type(
        backend_provider=relationship_backend_provider,
    )
    relationships = relationship_backend.parent_child_payload_from_labels(
        labels,
        masked_labels,
    )

    return image, stats, relationships, masked_labels


def _size_binary_mask_like_labels(
    labels: np.ndarray,
    binary_mask: np.ndarray,
) -> np.ndarray:
    """Return a binary mask sized like CP size_similarly(labels, mask)."""
    if binary_mask.shape == labels.shape:
        return binary_mask
    result = np.zeros(labels.shape, dtype=bool)
    common_slices = tuple(
        slice(0, min(label_extent, mask_extent))
        for label_extent, mask_extent in zip(labels.shape, binary_mask.shape, strict=False)
    )
    if not common_slices:
        return result
    result[common_slices] = binary_mask[common_slices]
    return result
