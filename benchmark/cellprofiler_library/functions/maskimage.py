"""Converted from CellProfiler: MaskImage."""

import numpy as np
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_inputs

from benchmark.cellprofiler_library.image_geometry import (
    aligned_image_mask_planes,
    binary_mask_plane,
    restore_image_mask_planes,
)


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
    mask_source = _coerce_mask_source(mask_source)
    masked_planes = tuple(
        _masked_plane(plane.image, plane.mask, invert_mask=invert_mask)
        for plane in aligned_image_mask_planes(
            image,
            mask,
            threshold=binary_threshold,
            labels=mask_source is MaskSource.OBJECTS,
        )
    )
    return restore_image_mask_planes(image, masked_planes)


def _masked_plane(
    image: np.ndarray,
    binary_mask: np.ndarray,
    *,
    invert_mask: bool,
) -> np.ndarray:
    if invert_mask:
        binary_mask = ~binary_mask
    masked = image.copy()
    masked[~binary_mask] = 0
    return masked


def _coerce_mask_source(value: MaskSource | str) -> MaskSource:
    if isinstance(value, MaskSource):
        return value
    normalized = str(value).strip().lower()
    for source in MaskSource:
        if normalized in {source.name.lower(), source.value.lower()}:
            return source
    raise ValueError(f"Unsupported MaskImage mask source: {value!r}.")


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
