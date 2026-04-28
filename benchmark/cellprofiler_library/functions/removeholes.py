"""
Converted from CellProfiler: RemoveHoles
Original: fill_holes

Fills holes smaller than the specified diameter in binary/labeled images.
Works on both 2D and 3D images. Output is always binary.
"""

import numpy as np
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


@numpy(contract=ProcessingContract.PURE_2D)
def remove_holes(
    image: np.ndarray,
    diameter: float = 1.0,
) -> np.ndarray:
    """
    Fill holes smaller than the specified diameter in a binary or labeled image.
    
    Args:
        image: Input image (H, W). Grayscale images are converted to binary
               by thresholding at 50% of the data range.
        diameter: Holes smaller than this diameter will be filled.
                  For 2D images, area threshold = pi * (diameter/2)^2.
    
    Returns:
        Binary image with small holes filled, shape (H, W), dtype float32.
    """
    import skimage.morphology
    from skimage import img_as_bool
    
    # Convert to binary if needed
    if image.dtype.kind == 'f':
        # For float images, use skimage's conversion which thresholds at 0.5
        binary_image = img_as_bool(image)
    elif image.dtype.kind in ('u', 'i'):
        # For integer images (labels), convert non-zero to True
        binary_image = image > 0
    else:
        binary_image = image.astype(bool)
    
    # Calculate area threshold from diameter
    # For 2D: area = pi * r^2
    radius = diameter / 2.0
    area_threshold = np.pi * (radius ** 2)
    
    # Ensure minimum area of 1
    area_threshold = max(1, int(area_threshold))
    
    # Remove small holes
    result = skimage.morphology.remove_small_holes(binary_image, area_threshold=area_threshold)
    
    return result.astype(np.float32)


@numpy(contract=ProcessingContract.PURE_3D)
def remove_holes_3d(
    image: np.ndarray,
    diameter: float = 1.0,
) -> np.ndarray:
    """
    Fill holes smaller than the specified diameter in a 3D binary or labeled image.
    
    Args:
        image: Input 3D image (D, H, W). Grayscale images are converted to binary
               by thresholding at 50% of the data range.
        diameter: Holes smaller than this diameter (in voxels) will be filled.
                  For 3D images, volume threshold = (4/3) * pi * (diameter/2)^3.
    
    Returns:
        Binary image with small holes filled, shape (D, H, W), dtype float32.
    """
    import skimage.morphology
    from skimage import img_as_bool
    
    # Convert to binary if needed
    if image.dtype.kind == 'f':
        binary_image = img_as_bool(image)
    elif image.dtype.kind in ('u', 'i'):
        binary_image = image > 0
    else:
        binary_image = image.astype(bool)
    
    # Calculate volume threshold from diameter
    # For 3D: volume = (4/3) * pi * r^3
    radius = diameter / 2.0
    volume_threshold = (4.0 / 3.0) * np.pi * (radius ** 3)
    
    # Ensure minimum volume of 1
    volume_threshold = max(1, int(volume_threshold))
    
    # Remove small holes (3D)
    result = skimage.morphology.remove_small_holes(binary_image, area_threshold=volume_threshold)
    
    return result.astype(np.float32)