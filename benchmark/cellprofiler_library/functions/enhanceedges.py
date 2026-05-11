"""
Converted from CellProfiler: EnhanceEdges
Original: enhanceedges
"""

import numpy as np

from openhcs.core.memory.decorators import numpy
from openhcs.core.runtime_values import (
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    with_image_payload_data,
)
from openhcs.processing.backends.cellprofiler._backend import (
    CellProfilerBackendProvider,
)
from openhcs.processing.backends.cellprofiler.edge import (
    EdgeDirection,
    EdgeEnhancementRequest,
    EdgeEnhancementStrategy,
    EdgeMethod,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.backends.cellprofiler.image_geometry import CellProfilerPlaneGeometry
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum


@numpy(contract=ProcessingContract.PURE_2D)
def enhance_edges(
    image: np.ndarray,
    method: EdgeMethod = EdgeMethod.SOBEL,
    direction: EdgeDirection = EdgeDirection.ALL,
    edge_backend_provider: CellProfilerBackendProvider | None = None,
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

    method = coerce_cellprofiler_enum(EdgeMethod, method)
    direction = coerce_cellprofiler_enum(EdgeDirection, direction)
    
    # Validate low_threshold
    if not 0 <= low_threshold <= 1:
        warnings.warn(
            f"low_threshold value of {low_threshold} is outside of the [0-1] range."
        )
    
    pixel_data = np.asarray(image_payload_data(image), dtype=np.float32)
    payload_mask = image_payload_mask(image)
    operation_mask = (
        np.ones(pixel_data.shape[:2], dtype=bool)
        if payload_mask is None
        else CellProfilerPlaneGeometry.from_image_plane(pixel_data).binary_mask(
            np.asarray(payload_mask),
        )
    )
    
    # Determine effective sigma
    effective_sigma = sigma if not automatic_gaussian else 2.0
    request = EdgeEnhancementRequest.build(
        image=pixel_data,
        mask=operation_mask,
        method=method,
        direction=direction,
        backend_provider=edge_backend_provider,
        automatic_threshold=automatic_threshold,
        automatic_low_threshold=automatic_low_threshold,
        sigma=effective_sigma,
        low_threshold=low_threshold,
        manual_threshold=manual_threshold,
        threshold_adjustment_factor=threshold_adjustment_factor,
    )
    output = EdgeEnhancementStrategy.for_request(request).enhance(request)
    
    # Ensure output is float32 and in valid range
    output = output.astype(np.float32)
    
    return with_image_payload_data(
        image,
        output,
        mask=operation_mask if payload_mask is not None else None,
        metadata=image_payload_metadata(image).without_unit_interval_intensity_scale(),
    )
