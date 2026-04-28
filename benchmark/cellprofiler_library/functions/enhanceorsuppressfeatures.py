"""
Converted from CellProfiler: EnhanceOrSuppressFeatures
Original: enhance_or_suppress_features
"""

import numpy as np
from typing import Tuple, Optional
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


class OperationMethod(Enum):
    ENHANCE = "enhance"
    SUPPRESS = "suppress"


class EnhanceMethod(Enum):
    SPECKLES = "speckles"
    NEURITES = "neurites"
    DARK_HOLES = "dark_holes"
    CIRCLES = "circles"
    TEXTURE = "texture"
    DIC = "dic"


class SpeckleAccuracy(Enum):
    FAST = "fast"
    SLOW = "slow"


class NeuriteMethod(Enum):
    GRADIENT = "gradient"
    TUBENESS = "tubeness"


def _enhance_speckles(image: np.ndarray, radius: float, accuracy: SpeckleAccuracy) -> np.ndarray:
    """Enhance speckle-like features using white tophat morphology."""
    from scipy.ndimage import white_tophat
    from skimage.morphology import disk
    
    selem = disk(int(radius))
    
    if accuracy == SpeckleAccuracy.FAST:
        # Fast mode: single tophat
        result = white_tophat(image, footprint=selem)
    else:
        # Slow mode: more accurate multi-scale approach
        result = white_tophat(image, footprint=selem)
        # Additional smoothing for accuracy
        from scipy.ndimage import gaussian_filter
        result = gaussian_filter(result, sigma=radius / 4)
    
    return result


def _enhance_neurites(image: np.ndarray, smoothing: float, radius: float, 
                      method: NeuriteMethod, rescale: bool) -> np.ndarray:
    """Enhance neurite/tubular structures using Hessian-based methods."""
    from scipy.ndimage import gaussian_filter
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals
    
    # Apply initial smoothing
    if smoothing > 0:
        smoothed = gaussian_filter(image, sigma=smoothing)
    else:
        smoothed = image
    
    if method == NeuriteMethod.GRADIENT:
        # Gradient-based enhancement
        from scipy.ndimage import sobel
        gx = sobel(smoothed, axis=1)
        gy = sobel(smoothed, axis=0)
        result = np.sqrt(gx**2 + gy**2)
    else:
        # Tubeness using Hessian eigenvalues
        sigma = radius / 2
        H = hessian_matrix(smoothed, sigma=sigma, order='rc')
        eigvals = hessian_matrix_eigvals(H)
        # For tubular structures, use the smaller eigenvalue magnitude
        result = np.abs(eigvals[1])
    
    if rescale:
        result = (result - result.min()) / (result.max() - result.min() + 1e-10)
    
    return result


def _enhance_dark_holes(image: np.ndarray, radius_min: int, radius_max: int) -> np.ndarray:
    """Enhance dark circular holes using morphological reconstruction."""
    from scipy.ndimage import grey_opening
    from skimage.morphology import disk, reconstruction
    
    # Use morphological opening with varying radii
    result = np.zeros_like(image)
    
    for r in range(radius_min, radius_max + 1):
        selem = disk(r)
        opened = grey_opening(image, footprint=selem)
        # Dark holes are where original is darker than opened
        holes = opened - image
        result = np.maximum(result, holes)
    
    return np.clip(result, 0, None)


def _enhance_circles(image: np.ndarray, radius: float) -> np.ndarray:
    """Enhance circular features using Hough-like approach or LoG."""
    from scipy.ndimage import gaussian_laplace
    
    # Laplacian of Gaussian for blob detection
    sigma = radius / np.sqrt(2)
    log_response = -gaussian_laplace(image, sigma=sigma) * sigma**2
    
    # Normalize
    result = np.clip(log_response, 0, None)
    if result.max() > 0:
        result = result / result.max()
    
    return result


def _enhance_texture(image: np.ndarray, smoothing: float) -> np.ndarray:
    """Enhance texture by computing local variance."""
    from scipy.ndimage import uniform_filter, gaussian_filter
    
    if smoothing > 0:
        smoothed = gaussian_filter(image, sigma=smoothing)
    else:
        smoothed = image
    
    # Local variance as texture measure
    size = max(3, int(smoothing * 2) + 1)
    local_mean = uniform_filter(smoothed, size=size)
    local_sqr_mean = uniform_filter(smoothed**2, size=size)
    local_var = local_sqr_mean - local_mean**2
    
    result = np.sqrt(np.clip(local_var, 0, None))
    
    return result


def _enhance_dic(image: np.ndarray, angle: float, decay: float, smoothing: float) -> np.ndarray:
    """Enhance DIC (Differential Interference Contrast) images."""
    from scipy.ndimage import gaussian_filter
    
    if smoothing > 0:
        smoothed = gaussian_filter(image, sigma=smoothing)
    else:
        smoothed = image
    
    # DIC integration along the shear direction
    angle_rad = np.deg2rad(angle)
    
    # Compute directional derivative
    dy = np.cos(angle_rad)
    dx = np.sin(angle_rad)
    
    # Gradient in shear direction
    from scipy.ndimage import sobel
    grad_y = sobel(smoothed, axis=0)
    grad_x = sobel(smoothed, axis=1)
    directional_grad = grad_x * dx + grad_y * dy
    
    # Integrate with decay (simple cumulative sum with decay)
    h, w = image.shape
    result = np.zeros_like(image)
    
    # Integration along angle direction
    if abs(dx) > abs(dy):
        for i in range(1, w):
            result[:, i] = decay * result[:, i-1] + directional_grad[:, i]
    else:
        for i in range(1, h):
            result[i, :] = decay * result[i-1, :] + directional_grad[i, :]
    
    return result


def _suppress(image: np.ndarray, radius: float) -> np.ndarray:
    """Suppress features smaller than the specified radius."""
    from scipy.ndimage import gaussian_filter
    
    # Gaussian smoothing to suppress small features
    sigma = radius / 2
    result = gaussian_filter(image, sigma=sigma)
    
    return result


@numpy(contract=ProcessingContract.PURE_2D)
def enhance_or_suppress_features(
    image: np.ndarray,
    method: OperationMethod = OperationMethod.ENHANCE,
    enhance_method: EnhanceMethod = EnhanceMethod.SPECKLES,
    radius: float = 10.0,
    speckle_accuracy: SpeckleAccuracy = SpeckleAccuracy.FAST,
    neurite_method: NeuriteMethod = NeuriteMethod.GRADIENT,
    neurite_rescale: bool = False,
    dark_hole_radius_min: int = 1,
    dark_hole_radius_max: int = 10,
    smoothing_value: float = 2.0,
    dic_angle: float = 0.0,
    dic_decay: float = 0.95,
) -> np.ndarray:
    """
    Enhance or suppress image features based on size and type.
    
    This module enhances or suppresses certain image features based on their
    size, shape, or texture characteristics.
    
    Args:
        image: Input grayscale image (H, W)
        method: Operation method - ENHANCE or SUPPRESS
        enhance_method: Type of feature to enhance (SPECKLES, NEURITES, DARK_HOLES, 
                       CIRCLES, TEXTURE, DIC)
        radius: Feature size in pixels
        speckle_accuracy: Speed/accuracy tradeoff for speckle enhancement
        neurite_method: Method for neurite enhancement (GRADIENT or TUBENESS)
        neurite_rescale: Whether to rescale neurite result to 0-1
        dark_hole_radius_min: Minimum radius for dark hole detection
        dark_hole_radius_max: Maximum radius for dark hole detection
        smoothing_value: Smoothing sigma for texture/neurite/DIC enhancement
        dic_angle: Shear angle for DIC enhancement in degrees
        dic_decay: Decay factor for DIC integration
    
    Returns:
        Enhanced or suppressed image (H, W)
    """
    # Ensure float image
    if image.dtype != np.float32 and image.dtype != np.float64:
        image = image.astype(np.float32)
    
    if method == OperationMethod.ENHANCE:
        if enhance_method == EnhanceMethod.SPECKLES:
            result = _enhance_speckles(image, radius, speckle_accuracy)
        
        elif enhance_method == EnhanceMethod.NEURITES:
            result = _enhance_neurites(image, smoothing_value, radius, 
                                       neurite_method, neurite_rescale)
        
        elif enhance_method == EnhanceMethod.DARK_HOLES:
            result = _enhance_dark_holes(image, dark_hole_radius_min, 
                                         dark_hole_radius_max)
        
        elif enhance_method == EnhanceMethod.CIRCLES:
            result = _enhance_circles(image, radius)
        
        elif enhance_method == EnhanceMethod.TEXTURE:
            result = _enhance_texture(image, smoothing_value)
        
        elif enhance_method == EnhanceMethod.DIC:
            result = _enhance_dic(image, dic_angle, dic_decay, smoothing_value)
        
        else:
            raise NotImplementedError(f"Unimplemented enhance method: {enhance_method}")
    
    elif method == OperationMethod.SUPPRESS:
        result = _suppress(image, radius)
    
    else:
        raise ValueError(f"Unknown filtering method: {method}")
    
    return result.astype(np.float32)