"""
Converted from CellProfiler: EnhanceEdges
Original: enhanceedges
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from enum import Enum
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta
from numba import njit, prange

from openhcs.core.memory.decorators import numpy
from openhcs.core.runtime_values import (
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    with_image_payload_data,
)
from openhcs.processing.backends.cellprofiler._backend import (
    CellProfilerBackendProvider,
    normalize_cellprofiler_backend_provider,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.backends.cellprofiler.image_geometry import CellProfilerPlaneGeometry
from benchmark.cellprofiler_library.functions._enum import _coerce_function_enum


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


@dataclass(frozen=True, slots=True)
class EdgeEnhancementStrategyKey:
    backend_provider: CellProfilerBackendProvider
    method: EdgeMethod
    direction: EdgeDirection


@dataclass(frozen=True, slots=True)
class EdgeEnhancementRequest:
    image: np.ndarray
    mask: np.ndarray
    backend_provider: CellProfilerBackendProvider
    method: EdgeMethod
    direction: EdgeDirection
    automatic_threshold: bool
    automatic_low_threshold: bool
    sigma: float
    low_threshold: float
    manual_threshold: float
    threshold_adjustment_factor: float


def _edge_enhancement_strategy_label(
    backend_provider: CellProfilerBackendProvider,
    method: EdgeMethod,
    direction: EdgeDirection,
) -> str:
    return f"{backend_provider.value}:{method.value}:{direction.value}"


def _default_edge_backend_provider(
    method: EdgeMethod,
    backend_provider: CellProfilerBackendProvider | None,
) -> CellProfilerBackendProvider:
    if backend_provider is not None:
        return normalize_cellprofiler_backend_provider(backend_provider)
    if method is EdgeMethod.SOBEL:
        return CellProfilerBackendProvider.NUMBA
    return CellProfilerBackendProvider.NATIVE


class EdgeEnhancementStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal dispatch point for one backend/method/direction edge algorithm."""

    __registry_key__ = "strategy_label"
    __skip_if_no_key__ = True
    strategy_label: ClassVar[str | None] = None
    strategy_key: ClassVar[EdgeEnhancementStrategyKey | None] = None

    @classmethod
    def for_request(cls, request: EdgeEnhancementRequest) -> "EdgeEnhancementStrategy":
        strategy_type = cls.__registry__.get(
            _edge_enhancement_strategy_label(
                request.backend_provider,
                request.method,
                request.direction,
            )
        )
        if strategy_type is None:
            strategy_type = cls.__registry__.get(
                _edge_enhancement_strategy_label(
                    request.backend_provider,
                    request.method,
                    EdgeDirection.ALL,
                )
            )
        if strategy_type is None:
            raise NotImplementedError(
                "No CellProfiler edge enhancement backend is registered for "
                f"provider {request.backend_provider.value!r}, method "
                f"{request.method.value!r}, direction {request.direction.value!r}."
            )
        return strategy_type()

    @abstractmethod
    def enhance(self, request: EdgeEnhancementRequest) -> np.ndarray:
        """Return edge-enhanced pixels for this strategy."""


class NumbaSobelAllStrategy(EdgeEnhancementStrategy):
    strategy_key = EdgeEnhancementStrategyKey(
        CellProfilerBackendProvider.NUMBA,
        EdgeMethod.SOBEL,
        EdgeDirection.ALL,
    )
    strategy_label = _edge_enhancement_strategy_label(
        strategy_key.backend_provider,
        strategy_key.method,
        strategy_key.direction,
    )

    def enhance(self, request: EdgeEnhancementRequest) -> np.ndarray:
        return _sobel_numba(request.image, request.mask, EdgeDirection.ALL)


class NumbaSobelHorizontalStrategy(EdgeEnhancementStrategy):
    strategy_key = EdgeEnhancementStrategyKey(
        CellProfilerBackendProvider.NUMBA,
        EdgeMethod.SOBEL,
        EdgeDirection.HORIZONTAL,
    )
    strategy_label = _edge_enhancement_strategy_label(
        strategy_key.backend_provider,
        strategy_key.method,
        strategy_key.direction,
    )

    def enhance(self, request: EdgeEnhancementRequest) -> np.ndarray:
        return _sobel_numba(request.image, request.mask, EdgeDirection.HORIZONTAL)


class NumbaSobelVerticalStrategy(EdgeEnhancementStrategy):
    strategy_key = EdgeEnhancementStrategyKey(
        CellProfilerBackendProvider.NUMBA,
        EdgeMethod.SOBEL,
        EdgeDirection.VERTICAL,
    )
    strategy_label = _edge_enhancement_strategy_label(
        strategy_key.backend_provider,
        strategy_key.method,
        strategy_key.direction,
    )

    def enhance(self, request: EdgeEnhancementRequest) -> np.ndarray:
        return _sobel_numba(request.image, request.mask, EdgeDirection.VERTICAL)


class NumpySobelAllStrategy(EdgeEnhancementStrategy):
    strategy_key = EdgeEnhancementStrategyKey(
        CellProfilerBackendProvider.NATIVE,
        EdgeMethod.SOBEL,
        EdgeDirection.ALL,
    )
    strategy_label = _edge_enhancement_strategy_label(
        strategy_key.backend_provider,
        strategy_key.method,
        strategy_key.direction,
    )

    def enhance(self, request: EdgeEnhancementRequest) -> np.ndarray:
        import centrosome.filter

        return centrosome.filter.sobel(request.image, request.mask)


class NumpySobelHorizontalStrategy(EdgeEnhancementStrategy):
    strategy_key = EdgeEnhancementStrategyKey(
        CellProfilerBackendProvider.NATIVE,
        EdgeMethod.SOBEL,
        EdgeDirection.HORIZONTAL,
    )
    strategy_label = _edge_enhancement_strategy_label(
        strategy_key.backend_provider,
        strategy_key.method,
        strategy_key.direction,
    )

    def enhance(self, request: EdgeEnhancementRequest) -> np.ndarray:
        import centrosome.filter

        return centrosome.filter.hsobel(request.image, request.mask)


class NumpySobelVerticalStrategy(EdgeEnhancementStrategy):
    strategy_key = EdgeEnhancementStrategyKey(
        CellProfilerBackendProvider.NATIVE,
        EdgeMethod.SOBEL,
        EdgeDirection.VERTICAL,
    )
    strategy_label = _edge_enhancement_strategy_label(
        strategy_key.backend_provider,
        strategy_key.method,
        strategy_key.direction,
    )

    def enhance(self, request: EdgeEnhancementRequest) -> np.ndarray:
        import centrosome.filter

        return centrosome.filter.vsobel(request.image, request.mask)


class NumpyPrewittAllStrategy(EdgeEnhancementStrategy):
    strategy_key = EdgeEnhancementStrategyKey(
        CellProfilerBackendProvider.NATIVE,
        EdgeMethod.PREWITT,
        EdgeDirection.ALL,
    )
    strategy_label = _edge_enhancement_strategy_label(
        strategy_key.backend_provider,
        strategy_key.method,
        strategy_key.direction,
    )

    def enhance(self, request: EdgeEnhancementRequest) -> np.ndarray:
        import centrosome.filter

        return centrosome.filter.prewitt(request.image, request.mask)


class NumpyPrewittHorizontalStrategy(EdgeEnhancementStrategy):
    strategy_key = EdgeEnhancementStrategyKey(
        CellProfilerBackendProvider.NATIVE,
        EdgeMethod.PREWITT,
        EdgeDirection.HORIZONTAL,
    )
    strategy_label = _edge_enhancement_strategy_label(
        strategy_key.backend_provider,
        strategy_key.method,
        strategy_key.direction,
    )

    def enhance(self, request: EdgeEnhancementRequest) -> np.ndarray:
        import centrosome.filter

        return centrosome.filter.hprewitt(request.image, request.mask)


class NumpyPrewittVerticalStrategy(EdgeEnhancementStrategy):
    strategy_key = EdgeEnhancementStrategyKey(
        CellProfilerBackendProvider.NATIVE,
        EdgeMethod.PREWITT,
        EdgeDirection.VERTICAL,
    )
    strategy_label = _edge_enhancement_strategy_label(
        strategy_key.backend_provider,
        strategy_key.method,
        strategy_key.direction,
    )

    def enhance(self, request: EdgeEnhancementRequest) -> np.ndarray:
        import centrosome.filter

        return centrosome.filter.vprewitt(request.image, request.mask)


class NumpyLaplacianOfGaussianStrategy(EdgeEnhancementStrategy):
    strategy_key = EdgeEnhancementStrategyKey(
        CellProfilerBackendProvider.NATIVE,
        EdgeMethod.LOG,
        EdgeDirection.ALL,
    )
    strategy_label = _edge_enhancement_strategy_label(
        strategy_key.backend_provider,
        strategy_key.method,
        strategy_key.direction,
    )

    def enhance(self, request: EdgeEnhancementRequest) -> np.ndarray:
        import centrosome.filter

        size = int(request.sigma * 4) + 1
        return centrosome.filter.laplacian_of_gaussian(
            request.image,
            request.mask,
            size,
            request.sigma,
        )


class NumpyCannyStrategy(EdgeEnhancementStrategy):
    strategy_key = EdgeEnhancementStrategyKey(
        CellProfilerBackendProvider.NATIVE,
        EdgeMethod.CANNY,
        EdgeDirection.ALL,
    )
    strategy_label = _edge_enhancement_strategy_label(
        strategy_key.backend_provider,
        strategy_key.method,
        strategy_key.direction,
    )

    def enhance(self, request: EdgeEnhancementRequest) -> np.ndarray:
        import centrosome.filter
        import centrosome.otsu

        low_threshold = request.low_threshold
        high_threshold = request.manual_threshold
        if request.automatic_threshold or request.automatic_low_threshold:
            sobel_image = centrosome.filter.sobel(request.image)
            low, high = centrosome.otsu.otsu3(sobel_image[request.mask])
            if request.automatic_threshold:
                high_threshold = high * request.threshold_adjustment_factor
            if request.automatic_low_threshold:
                low_threshold = low * request.threshold_adjustment_factor

        return centrosome.filter.canny(
            request.image,
            request.mask,
            request.sigma,
            low_threshold,
            high_threshold,
        )


class NumpyRobertsStrategy(EdgeEnhancementStrategy):
    strategy_key = EdgeEnhancementStrategyKey(
        CellProfilerBackendProvider.NATIVE,
        EdgeMethod.ROBERTS,
        EdgeDirection.ALL,
    )
    strategy_label = _edge_enhancement_strategy_label(
        strategy_key.backend_provider,
        strategy_key.method,
        strategy_key.direction,
    )

    def enhance(self, request: EdgeEnhancementRequest) -> np.ndarray:
        import centrosome.filter

        return centrosome.filter.roberts(request.image, request.mask)


class NumpyKirschStrategy(EdgeEnhancementStrategy):
    strategy_key = EdgeEnhancementStrategyKey(
        CellProfilerBackendProvider.NATIVE,
        EdgeMethod.KIRSCH,
        EdgeDirection.ALL,
    )
    strategy_label = _edge_enhancement_strategy_label(
        strategy_key.backend_provider,
        strategy_key.method,
        strategy_key.direction,
    )

    def enhance(self, request: EdgeEnhancementRequest) -> np.ndarray:
        import centrosome.kirsch

        return centrosome.kirsch.kirsch(request.image)


def _sobel_numba(
    image: np.ndarray,
    mask: np.ndarray,
    direction: EdgeDirection,
) -> np.ndarray:
    mode = {
        EdgeDirection.ALL: 0,
        EdgeDirection.HORIZONTAL: 1,
        EdgeDirection.VERTICAL: 2,
    }[direction]
    return _sobel_numba_kernel(
        np.ascontiguousarray(image, dtype=np.float32),
        np.ascontiguousarray(mask, dtype=np.bool_),
        mode,
    )


@njit(cache=True, parallel=True)
def _sobel_numba_kernel(
    image: np.ndarray,
    mask: np.ndarray,
    mode: int,
) -> np.ndarray:
    height, width = image.shape
    output = np.zeros((height, width), dtype=np.float32)
    if height < 3 or width < 3:
        return output

    for row in prange(1, height - 1):
        for col in range(1, width - 1):
            if not _full_sobel_neighborhood_is_valid(mask, row, col):
                continue

            horizontal = abs(
                (
                    image[row - 1, col - 1]
                    + 2.0 * image[row - 1, col]
                    + image[row - 1, col + 1]
                    - image[row + 1, col - 1]
                    - 2.0 * image[row + 1, col]
                    - image[row + 1, col + 1]
                )
                * 0.25
            )
            vertical = abs(
                (
                    image[row - 1, col - 1]
                    + 2.0 * image[row, col - 1]
                    + image[row + 1, col - 1]
                    - image[row - 1, col + 1]
                    - 2.0 * image[row, col + 1]
                    - image[row + 1, col + 1]
                )
                * 0.25
            )

            if mode == 1:
                output[row, col] = horizontal
            elif mode == 2:
                output[row, col] = vertical
            else:
                output[row, col] = np.sqrt(horizontal * horizontal + vertical * vertical)
    return output


@njit(cache=True)
def _full_sobel_neighborhood_is_valid(
    mask: np.ndarray,
    row: int,
    col: int,
) -> bool:
    for mask_row in range(row - 1, row + 2):
        for mask_col in range(col - 1, col + 2):
            if not mask[mask_row, mask_col]:
                return False
    return True


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

    method = _coerce_function_enum(EdgeMethod, method)
    direction = _coerce_function_enum(EdgeDirection, direction)
    backend_provider = _default_edge_backend_provider(method, edge_backend_provider)
    
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
    request = EdgeEnhancementRequest(
        image=pixel_data,
        mask=operation_mask,
        backend_provider=backend_provider,
        method=method,
        direction=direction,
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
