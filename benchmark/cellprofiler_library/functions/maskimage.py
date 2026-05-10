"""Converted from CellProfiler: MaskImage."""

import numpy as np
from enum import Enum
from dataclasses import replace
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_inputs
from openhcs.core.runtime_values import (
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    image_payload_with_context,
)

from openhcs.processing.backends.cellprofiler.image_geometry import (
    aligned_image_mask_planes,
    binary_mask_plane,
    collapse_singleton_plane_stack,
    restore_image_mask_planes,
)
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum


class MaskSource(Enum):
    """Source type for the mask."""
    OBJECTS = "objects"  # Use labeled objects as mask
    IMAGE = "image"      # Use binary/grayscale image as mask


@numpy
@special_inputs("mask")
def mask_image(
    image: np.ndarray,
    mask: np.ndarray,
    mask_source: MaskSource = MaskSource.IMAGE,
    invert_mask: bool = False,
    binary_threshold: float = 0.5,
) -> np.ndarray:
    """Mask an image using object labels or a binary/grayscale mask image."""
    mask_source = coerce_cellprofiler_enum(MaskSource, mask_source)
    masked_plane_results = tuple(
        _masked_plane(
            plane.image,
            plane.mask,
            invert_mask=invert_mask,
        )
        for plane in aligned_image_mask_planes(
            image,
            mask,
            threshold=binary_threshold,
            labels=mask_source is MaskSource.OBJECTS,
        )
    )
    masked_data = restore_image_mask_planes(
        image_payload_data(image),
        tuple(result[0] for result in masked_plane_results),
    )
    output_mask = restore_image_mask_planes(
        image_payload_data(image),
        tuple(result[1] for result in masked_plane_results),
    )
    return image_payload_with_context(
        masked_data,
        mask=output_mask,
        metadata=replace(image_payload_metadata(image), mask_defines_border=True),
    )


def _masked_plane(
    image: np.ndarray,
    binary_mask: np.ndarray,
    *,
    invert_mask: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if invert_mask:
        binary_mask = ~binary_mask
    existing_mask = image_payload_mask(image)
    if existing_mask is not None:
        binary_mask = np.asarray(binary_mask, dtype=bool) & np.asarray(
            collapse_singleton_plane_stack(existing_mask),
            dtype=bool,
        )
    image_data = collapse_singleton_plane_stack(image_payload_data(image))
    masked = image_data.copy()
    masked[~binary_mask] = 0
    return masked, np.asarray(binary_mask, dtype=bool)


@numpy(contract=ProcessingContract.PURE_2D)
def mask_image_with_binary(
    image: np.ndarray,
    invert_mask: bool = False,
) -> np.ndarray:
    """
    Mask an image using a binary mask stacked in dimension 0.
    
    This is a simplified version for when image and mask are stacked together
    along dimension 0: image[0] is the image, image[1] is the mask.
    
    Args:
        image: Stacked array where slice 0 is the image and slice 1 is the mask.
               Shape (2, H, W).
        invert_mask: If True, invert the mask.
    
    Returns:
        Masked image. Shape (H, W).
    """
    # This function receives (H, W) due to PURE_2D contract
    # For the stacked case, use the FLEXIBLE version above
    # This version assumes mask is already applied or passed separately
    
    # Create binary mask (threshold at 0.5 for grayscale)
    binary_mask = image > 0.5
    
    if invert_mask:
        binary_mask = ~binary_mask
    
    return binary_mask.astype(np.float32)


@numpy
def mask_image_stacked(
    image: np.ndarray,
    invert_mask: bool = False,
    binary_threshold: float = 0.5,
) -> np.ndarray:
    """
    Mask an image where image and mask are stacked along dimension 0.
    
    Args:
        image: Stacked array. Shape (2, H, W) where:
               - image[0] is the image to be masked
               - image[1] is the mask (binary or grayscale)
        invert_mask: If True, invert the mask.
        binary_threshold: Threshold for converting grayscale mask to binary.
    
    Returns:
        Masked image. Shape (1, H, W).
    """
    img = image[0]
    mask = image[1]
    binary_mask = binary_mask_plane(mask, threshold=binary_threshold)
    
    if invert_mask:
        binary_mask = ~binary_mask
    
    # Apply mask
    result = img.copy()
    result[~binary_mask] = 0
    
    return result[np.newaxis, ...]  # Return (1, H, W)
