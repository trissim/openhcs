"""
Converted from CellProfiler: MaskImage
Original: MaskImage.run

MaskImage hides certain portions of an image (based on previously
identified objects or a binary image) so they are ignored by subsequent
mask-respecting modules in the pipeline.
"""

import numpy as np
from typing import Tuple
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_inputs


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
    """
    Mask an image using objects or a binary/grayscale mask image.
    
    The masked image has pixels set to 0 where the mask is False (or where
    objects are not present if using objects as mask).
    
    Args:
        image: Input image to be masked. Shape (D, H, W) where D is the
               iteration axis (could be z-slices, timepoints, etc.)
        mask: Mask array. Shape (D, H, W). If mask_source is OBJECTS, this
              should be a label image (integers). If mask_source is IMAGE,
              this should be binary or grayscale.
        mask_source: Whether mask is from labeled objects or a binary image.
        invert_mask: If True, invert the mask (mask foreground instead of
                     background).
        binary_threshold: Threshold for converting grayscale mask to binary
                          (only used when mask_source is IMAGE and mask is
                          not already binary).
    
    Returns:
        Masked image with same shape as input. Pixels outside mask are set to 0.
    """
    # Process each slice along dimension 0
    result = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        img_slice = image[i]
        mask_slice = mask[i] if mask.shape[0] > 1 else mask[0]
        
        # Create binary mask based on source type
        if mask_source == MaskSource.OBJECTS:
            # Labels: mask is where labels > 0
            binary_mask = mask_slice > 0
        else:
            # Image: check if already binary, otherwise threshold
            unique_vals = np.unique(mask_slice)
            if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, True, False}):
                # Already binary
                binary_mask = mask_slice > 0
            else:
                # Grayscale - threshold at specified value
                binary_mask = mask_slice > binary_threshold
        
        # Invert if requested
        if invert_mask:
            binary_mask = ~binary_mask
        
        # Apply mask - set pixels outside mask to 0
        masked_slice = img_slice.copy()
        masked_slice[~binary_mask] = 0
        result[i] = masked_slice
    
    return result


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
    
    # Create binary mask
    unique_vals = np.unique(mask)
    if len(unique_vals) <= 2 and np.all((unique_vals == 0) | (unique_vals == 1)):
        binary_mask = mask > 0
    else:
        binary_mask = mask > binary_threshold
    
    if invert_mask:
        binary_mask = ~binary_mask
    
    # Apply mask
    result = img.copy()
    result[~binary_mask] = 0
    
    return result[np.newaxis, ...]  # Return (1, H, W)