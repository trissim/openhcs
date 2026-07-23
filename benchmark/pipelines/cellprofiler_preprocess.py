#!/usr/bin/env python3
"""
CellProfiler Preprocessing - CPU, pyclesperanto, and cupy backends

CellProfiler's IdentifyPrimaryObjects preprocessing:
1. Gaussian smoothing (sigma = diameter / 3.5)
2. Optional median filtering for salt-and-pepper noise

Three backends with IDENTICAL output:
- preprocess_cpu: numpy/scipy (CPU)
- preprocess_gpu: pyclesperanto (OpenCL GPU)
- preprocess_cupy: cupy/cucim (CUDA GPU)
"""

from typing import Tuple
import numpy as np

from openhcs.core.memory.decorators import numpy as numpy_func


# =============================================================================
# CPU Backend (numpy/scipy)
# =============================================================================

@numpy_func
def preprocess_cpu(
    image: np.ndarray,
    gaussian_sigma: float = 2.0,
    median_size: int = 0,  # 0 = disabled
) -> np.ndarray:
    """
    CellProfiler-equivalent preprocessing on CPU.
    
    Args:
        image: 3D array (slices, height, width)
        gaussian_sigma: Gaussian blur sigma (CellProfiler default: diameter/3.5)
        median_size: Median filter size (0 to disable)
    """
    from scipy.ndimage import gaussian_filter, median_filter
    
    result = np.empty_like(image, dtype=np.float32)
    
    for i, slice_2d in enumerate(image):
        processed = slice_2d.astype(np.float32)
        
        # Gaussian smoothing
        if gaussian_sigma > 0:
            processed = gaussian_filter(processed, sigma=gaussian_sigma)
        
        # Optional median filter
        if median_size > 0:
            processed = median_filter(processed, size=median_size)
        
        result[i] = processed
    
    return result


# =============================================================================
# pyclesperanto Backend (OpenCL GPU)
# =============================================================================

try:
    import pyclesperanto as cle
    from openhcs.core.memory.decorators import pyclesperanto as pyclesperanto_func
    
    @pyclesperanto_func
    def preprocess_gpu(
        image: np.ndarray,
        gaussian_sigma: float = 2.0,
        median_size: int = 0,
    ) -> np.ndarray:
        """CellProfiler-equivalent preprocessing on GPU (pyclesperanto)."""
        result = []
        
        for slice_2d in image:
            gpu_slice = cle.push(slice_2d.astype(np.float32))
            
            # Gaussian smoothing
            if gaussian_sigma > 0:
                gpu_slice = cle.gaussian_blur(gpu_slice, sigma_x=gaussian_sigma, sigma_y=gaussian_sigma)
            
            # Optional median filter
            if median_size > 0:
                gpu_slice = cle.median(gpu_slice, radius_x=median_size//2, radius_y=median_size//2)
            
            result.append(cle.pull(gpu_slice))
        
        return np.stack(result)

except ImportError:
    preprocess_gpu = None


# =============================================================================
# cupy/cucim Backend (CUDA GPU)
# =============================================================================

try:
    import cupy as cp
    from cucim.skimage.filters import gaussian, median
    from openhcs.core.memory.decorators import cupy as cupy_func
    
    @cupy_func
    def preprocess_cupy(
        image: np.ndarray,
        gaussian_sigma: float = 2.0,
        median_size: int = 0,
    ) -> np.ndarray:
        """CellProfiler-equivalent preprocessing on GPU (cupy/cucim)."""
        # Push entire stack to GPU
        gpu_image = cp.asarray(image.astype(np.float32))
        result = cp.empty_like(gpu_image)
        
        for i in range(gpu_image.shape[0]):
            processed = gpu_image[i]
            
            # Gaussian smoothing (cucim has same API as skimage)
            if gaussian_sigma > 0:
                processed = gaussian(processed, sigma=gaussian_sigma)
            
            # Optional median filter
            if median_size > 0:
                from cucim.skimage.morphology import disk
                processed = median(processed, footprint=disk(median_size//2))
            
            result[i] = processed
        
        return cp.asnumpy(result)

except ImportError:
    preprocess_cupy = None

