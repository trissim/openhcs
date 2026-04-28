"""
Converted from CellProfiler: EnhanceEdges
Original: enhanceedges
"""

import numpy as np
from typing import Tuple, Optional
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


class EdgeMethod(Enum):
    SOBEL = "sobel"
    LOG = "log"
    PREWITT = "prewitt"
    CANNY = "canny"
    ROBERTS = "roberts"
    KIRSCH = "kirsch"


class EdgeDirection(Enum):
    ALL = "all"
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


def _enhance_edges_sobel(image: np.ndarray, mask: np.ndarray, direction: EdgeDirection) -> np.ndarray:
    """Apply Sobel edge detection."""
    from scipy.ndimage import sobel
    
    if direction == EdgeDirection.ALL:
        sobel_x = sobel(image, axis=1)
        sobel_y = sobel(image, axis=0)
        output = np.hypot(sobel_x, sobel_y)
    elif direction == EdgeDirection.HORIZONTAL:
        output = np.abs(sobel(image, axis=0))
    elif direction == EdgeDirection.VERTICAL:
        output = np.abs(sobel(image, axis=1))
    else:
        sobel_x = sobel(image, axis=1)
        sobel_y = sobel(image, axis=0)
        output = np.hypot(sobel_x, sobel_y)
    
    output[~mask] = 0
    return output


def _enhance_edges_prewitt(image: np.ndarray, mask: np.ndarray, direction: EdgeDirection) -> np.ndarray:
    """Apply Prewitt edge detection."""
    from scipy.ndimage import prewitt
    
    if direction == EdgeDirection.ALL:
        prewitt_x = prewitt(image, axis=1)
        prewitt_y = prewitt(image, axis=0)
        output = np.hypot(prewitt_x, prewitt_y)
    elif direction == EdgeDirection.HORIZONTAL:
        output = np.abs(prewitt(image, axis=0))
    elif direction == EdgeDirection.VERTICAL:
        output = np.abs(prewitt(image, axis=1))
    else:
        prewitt_x = prewitt(image, axis=1)
        prewitt_y = prewitt(image, axis=0)
        output = np.hypot(prewitt_x, prewitt_y)
    
    output[~mask] = 0
    return output


def _enhance_edges_log(image: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Laplacian of Gaussian edge detection."""
    from scipy.ndimage import gaussian_laplace
    
    # Apply LoG filter
    output = -gaussian_laplace(image, sigma=sigma)
    
    # Normalize to [0, 1] range
    output = output - output.min()
    if output.max() > 0:
        output = output / output.max()
    
    output[~mask] = 0
    return output


def _enhance_edges_canny(
    image: np.ndarray,
    mask: np.ndarray,
    auto_threshold: bool,
    auto_low_threshold: bool,
    sigma: float,
    low_threshold: float,
    manual_threshold: float,
    threshold_adjustment_factor: float,
) -> np.ndarray:
    """Apply Canny edge detection."""
    from skimage.feature import canny
    from skimage.filters import threshold_otsu
    
    # Determine high threshold
    if auto_threshold:
        # Use Otsu's method to find threshold
        try:
            high_threshold = threshold_otsu(image[mask]) * threshold_adjustment_factor
        except ValueError:
            high_threshold = 0.5 * threshold_adjustment_factor
    else:
        high_threshold = manual_threshold * threshold_adjustment_factor
    
    # Determine low threshold
    if auto_low_threshold:
        low_thresh = high_threshold * 0.4  # Typical ratio
    else:
        low_thresh = low_threshold
    
    # Ensure low < high
    low_thresh = min(low_thresh, high_threshold * 0.99)
    
    # Apply Canny
    output = canny(
        image,
        sigma=sigma,
        low_threshold=low_thresh,
        high_threshold=high_threshold,
        mask=mask,
    ).astype(np.float32)
    
    return output


def _enhance_edges_roberts(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply Roberts cross edge detection."""
    from skimage.filters import roberts
    
    output = roberts(image)
    output[~mask] = 0
    return output


def _enhance_edges_kirsch(image: np.ndarray) -> np.ndarray:
    """Apply Kirsch edge detection using 8 directional kernels."""
    from scipy.ndimage import convolve
    
    # Kirsch kernels for 8 directions
    kernels = [
        np.array([[ 5,  5,  5], [-3,  0, -3], [-3, -3, -3]], dtype=np.float32),
        np.array([[ 5,  5, -3], [ 5,  0, -3], [-3, -3, -3]], dtype=np.float32),
        np.array([[ 5, -3, -3], [ 5,  0, -3], [ 5, -3, -3]], dtype=np.float32),
        np.array([[-3, -3, -3], [ 5,  0, -3], [ 5,  5, -3]], dtype=np.float32),
        np.array([[-3, -3, -3], [-3,  0, -3], [ 5,  5,  5]], dtype=np.float32),
        np.array([[-3, -3, -3], [-3,  0,  5], [-3,  5,  5]], dtype=np.float32),
        np.array([[-3, -3,  5], [-3,  0,  5], [-3, -3,  5]], dtype=np.float32),
        np.array([[-3,  5,  5], [-3,  0,  5], [-3, -3, -3]], dtype=np.float32),
    ]
    
    # Apply all kernels and take maximum response
    responses = [convolve(image, k) for k in kernels]
    output = np.maximum.reduce(responses)
    
    # Normalize
    output = output - output.min()
    if output.max() > 0:
        output = output / output.max()
    
    return output


@numpy(contract=ProcessingContract.PURE_2D)
def enhance_edges(
    image: np.ndarray,
    method: EdgeMethod = EdgeMethod.SOBEL,
    direction: EdgeDirection = EdgeDirection.ALL,
    automatic_threshold: bool = True,
    automatic_gaussian: bool = True,
    sigma: float = 10.0,
    manual_threshold: float = 0.2,
    threshold_adjustment_factor: float = 1.0,
    automatic_low_threshold: bool = True,
    low_threshold: float = 0.1,
) -> np.ndarray:
    """Enhance edges in an image using various edge detection algorithms.
    
    This function applies edge detection algorithms to highlight edges in the image.
    Different methods are suitable for different applications.
    
    Parameters
    ----------
    image : np.ndarray
        Input image with shape (H, W), values typically in [0, 1] range.
    method : EdgeMethod
        Edge detection algorithm to apply:
        - SOBEL: Gradient-based, good general purpose
        - LOG: Laplacian of Gaussian, good for blob detection
        - PREWITT: Similar to Sobel, slightly different kernel
        - CANNY: Multi-stage, produces thin edges
        - ROBERTS: Simple diagonal gradient
        - KIRSCH: 8-directional compass operator
    direction : EdgeDirection
        For Sobel and Prewitt only - which edge direction to detect:
        - ALL: Both horizontal and vertical (magnitude)
        - HORIZONTAL: Horizontal edges only
        - VERTICAL: Vertical edges only
    automatic_threshold : bool
        For Canny only - automatically determine high threshold using Otsu's method.
    automatic_gaussian : bool
        For Canny and LOG - if True, use default sigma; if False, use sigma parameter.
    sigma : float
        Gaussian smoothing sigma for Canny and LOG methods. Only used if automatic_gaussian is False.
    manual_threshold : float
        For Canny only - manual high threshold value when automatic_threshold is False.
    threshold_adjustment_factor : float
        For Canny only - multiplier applied to the threshold.
    automatic_low_threshold : bool
        For Canny only - automatically determine low threshold as fraction of high.
    low_threshold : float
        For Canny only - manual low threshold when automatic_low_threshold is False.
    
    Returns
    -------
    np.ndarray
        Edge-enhanced image with shape (H, W), values in [0, 1] range.
    """
    import warnings
    
    # Validate low_threshold
    if not 0 <= low_threshold <= 1:
        warnings.warn(
            f"low_threshold value of {low_threshold} is outside of the [0-1] range."
        )
    
    # Create default mask (all True)
    mask = np.ones(image.shape, dtype=bool)
    
    # Determine effective sigma
    effective_sigma = sigma if not automatic_gaussian else 2.0
    
    # Apply selected edge detection method
    if method == EdgeMethod.SOBEL:
        output = _enhance_edges_sobel(image, mask, direction)
    elif method == EdgeMethod.LOG:
        output = _enhance_edges_log(image, mask, effective_sigma)
    elif method == EdgeMethod.PREWITT:
        output = _enhance_edges_prewitt(image, mask, direction)
    elif method == EdgeMethod.CANNY:
        output = _enhance_edges_canny(
            image,
            mask,
            auto_threshold=automatic_threshold,
            auto_low_threshold=automatic_low_threshold,
            sigma=effective_sigma,
            low_threshold=low_threshold,
            manual_threshold=manual_threshold,
            threshold_adjustment_factor=threshold_adjustment_factor,
        )
    elif method == EdgeMethod.ROBERTS:
        output = _enhance_edges_roberts(image, mask)
    elif method == EdgeMethod.KIRSCH:
        output = _enhance_edges_kirsch(image)
    else:
        raise NotImplementedError(f"{method} edge detection method is not implemented.")
    
    # Ensure output is float32 and in valid range
    output = output.astype(np.float32)
    
    return output