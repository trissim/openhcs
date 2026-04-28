"""
Converted from CellProfiler: MorphologicalSkeleton
Original: morphologicalskeleton
"""

import numpy as np
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


@numpy(contract=ProcessingContract.PURE_2D)
def morphological_skeleton_2d(
    image: np.ndarray,
) -> np.ndarray:
    """Compute morphological skeleton of a 2D binary image.
    
    The skeleton is a thin representation of the shape that preserves
    the topology and is equidistant from the boundaries.
    
    Args:
        image: Input binary image with shape (H, W)
        
    Returns:
        Skeletonized binary image with shape (H, W)
    """
    from skimage.morphology import skeletonize
    
    # Ensure binary input
    binary = image > 0
    
    # Compute skeleton
    skeleton = skeletonize(binary)
    
    return skeleton.astype(np.float32)


@numpy(contract=ProcessingContract.PURE_3D)
def morphological_skeleton_3d(
    image: np.ndarray,
) -> np.ndarray:
    """Compute morphological skeleton of a 3D binary volume.
    
    The 3D skeleton preserves topology across the entire volume,
    considering connectivity in all three dimensions.
    
    Args:
        image: Input binary volume with shape (D, H, W)
        
    Returns:
        Skeletonized binary volume with shape (D, H, W)
    """
    from skimage.morphology import skeletonize_3d
    
    # Ensure binary input
    binary = image > 0
    
    # Compute 3D skeleton
    skeleton = skeletonize_3d(binary)
    
    return skeleton.astype(np.float32)


@numpy
def morphologicalskeleton(
    image: np.ndarray,
    volumetric: bool = False,
) -> np.ndarray:
    """Compute morphological skeleton of a binary image or volume.
    
    The skeleton is a thin representation of the shape that preserves
    the topology and is equidistant from the boundaries.
    
    Args:
        image: Input binary image with shape (D, H, W)
        volumetric: If True, compute 3D skeleton treating the entire
                   volume as connected. If False, compute 2D skeleton
                   on each slice independently.
        
    Returns:
        Skeletonized binary image/volume with shape (D, H, W)
    """
    from skimage.morphology import skeletonize, skeletonize_3d
    
    # Ensure binary input
    binary = image > 0
    
    if volumetric:
        # 3D skeletonization - treats entire volume as connected
        skeleton = skeletonize_3d(binary)
        return skeleton.astype(np.float32)
    else:
        # 2D skeletonization - process each slice independently
        result = np.zeros_like(image, dtype=np.float32)
        for i in range(image.shape[0]):
            result[i] = skeletonize(binary[i]).astype(np.float32)
        return result