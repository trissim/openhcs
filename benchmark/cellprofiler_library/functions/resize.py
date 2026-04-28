"""
Converted from CellProfiler: Resize
Original: Resize module

Resizes images (changes their resolution) by applying a resizing factor
or by specifying desired dimensions in pixels.
"""

import numpy as np
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


class ResizeMethod(Enum):
    BY_FACTOR = "by_factor"
    TO_SIZE = "to_size"


class InterpolationMethod(Enum):
    NEAREST_NEIGHBOR = "nearest_neighbor"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"


@numpy(contract=ProcessingContract.PURE_2D)
def resize(
    image: np.ndarray,
    resize_method: ResizeMethod = ResizeMethod.BY_FACTOR,
    resizing_factor_x: float = 0.25,
    resizing_factor_y: float = 0.25,
    specific_width: int = 100,
    specific_height: int = 100,
    interpolation: InterpolationMethod = InterpolationMethod.NEAREST_NEIGHBOR,
) -> np.ndarray:
    """
    Resize an image by a factor or to specific dimensions.
    
    Args:
        image: Input image with shape (H, W)
        resize_method: Whether to resize by factor or to specific size
        resizing_factor_x: X scaling factor (used if resize_method is BY_FACTOR)
        resizing_factor_y: Y scaling factor (used if resize_method is BY_FACTOR)
        specific_width: Target width in pixels (used if resize_method is TO_SIZE)
        specific_height: Target height in pixels (used if resize_method is TO_SIZE)
        interpolation: Interpolation method to use
        
    Returns:
        Resized image with shape (new_H, new_W)
    """
    import skimage.transform
    
    height, width = image.shape[:2]
    
    # Determine new shape based on resize method
    if resize_method == ResizeMethod.BY_FACTOR:
        new_height = int(np.round(height * resizing_factor_y))
        new_width = int(np.round(width * resizing_factor_x))
    else:  # TO_SIZE
        new_height = specific_height
        new_width = specific_width
    
    new_shape = (new_height, new_width)
    
    # Determine interpolation order
    if interpolation == InterpolationMethod.NEAREST_NEIGHBOR:
        order = 0
    elif interpolation == InterpolationMethod.BILINEAR:
        order = 1
    else:  # BICUBIC
        order = 3
    
    # Perform resize
    output_pixels = skimage.transform.resize(
        image,
        new_shape,
        order=order,
        mode="symmetric",
        preserve_range=True,
    )
    
    return output_pixels.astype(image.dtype)


@numpy(contract=ProcessingContract.PURE_3D)
def resize_volumetric(
    image: np.ndarray,
    resize_method: ResizeMethod = ResizeMethod.BY_FACTOR,
    resizing_factor_x: float = 0.25,
    resizing_factor_y: float = 0.25,
    resizing_factor_z: float = 0.25,
    specific_width: int = 100,
    specific_height: int = 100,
    specific_planes: int = 10,
    interpolation: InterpolationMethod = InterpolationMethod.NEAREST_NEIGHBOR,
) -> np.ndarray:
    """
    Resize a 3D volumetric image by a factor or to specific dimensions.
    
    Args:
        image: Input volumetric image with shape (D, H, W)
        resize_method: Whether to resize by factor or to specific size
        resizing_factor_x: X scaling factor (used if resize_method is BY_FACTOR)
        resizing_factor_y: Y scaling factor (used if resize_method is BY_FACTOR)
        resizing_factor_z: Z scaling factor (used if resize_method is BY_FACTOR)
        specific_width: Target width in pixels (used if resize_method is TO_SIZE)
        specific_height: Target height in pixels (used if resize_method is TO_SIZE)
        specific_planes: Target number of planes (used if resize_method is TO_SIZE)
        interpolation: Interpolation method to use
        
    Returns:
        Resized volumetric image with shape (new_D, new_H, new_W)
    """
    import skimage.transform
    
    planes, height, width = image.shape[:3]
    
    # Determine new shape based on resize method
    if resize_method == ResizeMethod.BY_FACTOR:
        new_planes = int(np.round(planes * resizing_factor_z))
        new_height = int(np.round(height * resizing_factor_y))
        new_width = int(np.round(width * resizing_factor_x))
    else:  # TO_SIZE
        new_planes = specific_planes
        new_height = specific_height
        new_width = specific_width
    
    new_shape = (new_planes, new_height, new_width)
    
    # Determine interpolation order
    if interpolation == InterpolationMethod.NEAREST_NEIGHBOR:
        order = 0
    elif interpolation == InterpolationMethod.BILINEAR:
        order = 1
    else:  # BICUBIC
        order = 3
    
    # Perform 3D resize
    output_pixels = skimage.transform.resize(
        image,
        new_shape,
        order=order,
        mode="symmetric",
        preserve_range=True,
    )
    
    return output_pixels.astype(image.dtype)