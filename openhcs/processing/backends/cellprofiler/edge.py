"""CellProfiler-compatible edge enhancement backend semantics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit, prange

from openhcs.processing.backends.cellprofiler._backend import (
    CellProfilerBackendProvider,
    normalize_cellprofiler_backend_provider,
)


class EdgeMethod(Enum):
    SOBEL = "sobel"
    LOG = "log"
    PREWITT = "prewitt"
    CANNY = "canny"
    ROBERTS = "roberts"
    KIRSCH = "kirsch"

    @property
    def default_backend_provider(self) -> CellProfilerBackendProvider:
        if self is EdgeMethod.SOBEL:
            return CellProfilerBackendProvider.NUMBA
        return CellProfilerBackendProvider.NATIVE


class EdgeDirection(Enum):
    ALL = ("all", True, True)
    HORIZONTAL = ("horizontal", True, False)
    VERTICAL = ("vertical", False, True)

    def __init__(
        self,
        label: str,
        includes_horizontal_response: bool,
        includes_vertical_response: bool,
    ) -> None:
        self._value_ = label
        self._includes_horizontal_response = includes_horizontal_response
        self._includes_vertical_response = includes_vertical_response

    @property
    def includes_horizontal_response(self) -> bool:
        return self._includes_horizontal_response

    @property
    def includes_vertical_response(self) -> bool:
        return self._includes_vertical_response


@dataclass(frozen=True, slots=True)
class EdgeEnhancementStrategyKey:
    backend_provider: CellProfilerBackendProvider
    method: EdgeMethod
    direction: EdgeDirection

    @property
    def label(self) -> str:
        return (
            f"{self.backend_provider.value}:"
            f"{self.method.value}:"
            f"{self.direction.value}"
        )


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

    @classmethod
    def build(
        cls,
        *,
        image: np.ndarray,
        mask: np.ndarray,
        method: EdgeMethod,
        direction: EdgeDirection,
        backend_provider: CellProfilerBackendProvider | None,
        automatic_threshold: bool,
        automatic_low_threshold: bool,
        sigma: float,
        low_threshold: float,
        manual_threshold: float,
        threshold_adjustment_factor: float,
    ) -> "EdgeEnhancementRequest":
        resolved_provider = (
            method.default_backend_provider
            if backend_provider is None
            else normalize_cellprofiler_backend_provider(backend_provider)
        )
        return cls(
            image=image,
            mask=mask,
            backend_provider=resolved_provider,
            method=method,
            direction=direction,
            automatic_threshold=automatic_threshold,
            automatic_low_threshold=automatic_low_threshold,
            sigma=sigma,
            low_threshold=low_threshold,
            manual_threshold=manual_threshold,
            threshold_adjustment_factor=threshold_adjustment_factor,
        )

    @property
    def strategy_key(self) -> EdgeEnhancementStrategyKey:
        return EdgeEnhancementStrategyKey(
            self.backend_provider,
            self.method,
            self.direction,
        )

    @property
    def fallback_strategy_key(self) -> EdgeEnhancementStrategyKey:
        return EdgeEnhancementStrategyKey(
            self.backend_provider,
            self.method,
            EdgeDirection.ALL,
        )


class EdgeEnhancementStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal dispatch point for one backend/method/direction edge algorithm."""

    __registry_key__ = "strategy_label"
    __skip_if_no_key__ = True
    strategy_label: ClassVar[str | None] = None
    strategy_key: ClassVar[EdgeEnhancementStrategyKey | None] = None

    @classmethod
    def for_request(cls, request: EdgeEnhancementRequest) -> "EdgeEnhancementStrategy":
        strategy_type = cls.__registry__.get(request.strategy_key.label)
        if strategy_type is None:
            strategy_type = cls.__registry__.get(request.fallback_strategy_key.label)
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


class EdgeEnhancementStrategyLeaf(EdgeEnhancementStrategy):
    """Declarative base for concrete edge enhancement leaves."""

    backend_provider: ClassVar[CellProfilerBackendProvider | None] = None
    method: ClassVar[EdgeMethod | None] = None
    direction: ClassVar[EdgeDirection | None] = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls.backend_provider is None or cls.method is None or cls.direction is None:
            return
        cls.strategy_key = EdgeEnhancementStrategyKey(
            cls.backend_provider,
            cls.method,
            cls.direction,
        )
        cls.strategy_label = cls.strategy_key.label


class NumbaSobelStrategy(EdgeEnhancementStrategyLeaf):
    """Shared Numba Sobel implementation for direction-specific leaves."""

    backend_provider = CellProfilerBackendProvider.NUMBA
    method = EdgeMethod.SOBEL

    def enhance(self, request: EdgeEnhancementRequest) -> np.ndarray:
        return _sobel_numba_kernel(
            np.ascontiguousarray(request.image, dtype=np.float32),
            np.ascontiguousarray(request.mask, dtype=np.bool_),
            request.direction.includes_horizontal_response,
            request.direction.includes_vertical_response,
        )


class NumbaSobelAllStrategy(NumbaSobelStrategy):
    direction = EdgeDirection.ALL


class NumbaSobelHorizontalStrategy(NumbaSobelStrategy):
    direction = EdgeDirection.HORIZONTAL


class NumbaSobelVerticalStrategy(NumbaSobelStrategy):
    direction = EdgeDirection.VERTICAL


class NumpySobelAllStrategy(EdgeEnhancementStrategyLeaf):
    backend_provider = CellProfilerBackendProvider.NATIVE
    method = EdgeMethod.SOBEL
    direction = EdgeDirection.ALL

    def enhance(self, request: EdgeEnhancementRequest) -> np.ndarray:
        import centrosome.filter

        return centrosome.filter.sobel(request.image, request.mask)


class NumpySobelHorizontalStrategy(EdgeEnhancementStrategyLeaf):
    backend_provider = CellProfilerBackendProvider.NATIVE
    method = EdgeMethod.SOBEL
    direction = EdgeDirection.HORIZONTAL

    def enhance(self, request: EdgeEnhancementRequest) -> np.ndarray:
        import centrosome.filter

        return centrosome.filter.hsobel(request.image, request.mask)


class NumpySobelVerticalStrategy(EdgeEnhancementStrategyLeaf):
    backend_provider = CellProfilerBackendProvider.NATIVE
    method = EdgeMethod.SOBEL
    direction = EdgeDirection.VERTICAL

    def enhance(self, request: EdgeEnhancementRequest) -> np.ndarray:
        import centrosome.filter

        return centrosome.filter.vsobel(request.image, request.mask)


class NumpyPrewittAllStrategy(EdgeEnhancementStrategyLeaf):
    backend_provider = CellProfilerBackendProvider.NATIVE
    method = EdgeMethod.PREWITT
    direction = EdgeDirection.ALL

    def enhance(self, request: EdgeEnhancementRequest) -> np.ndarray:
        import centrosome.filter

        return centrosome.filter.prewitt(request.image, request.mask)


class NumpyPrewittHorizontalStrategy(EdgeEnhancementStrategyLeaf):
    backend_provider = CellProfilerBackendProvider.NATIVE
    method = EdgeMethod.PREWITT
    direction = EdgeDirection.HORIZONTAL

    def enhance(self, request: EdgeEnhancementRequest) -> np.ndarray:
        import centrosome.filter

        return centrosome.filter.hprewitt(request.image, request.mask)


class NumpyPrewittVerticalStrategy(EdgeEnhancementStrategyLeaf):
    backend_provider = CellProfilerBackendProvider.NATIVE
    method = EdgeMethod.PREWITT
    direction = EdgeDirection.VERTICAL

    def enhance(self, request: EdgeEnhancementRequest) -> np.ndarray:
        import centrosome.filter

        return centrosome.filter.vprewitt(request.image, request.mask)


class NumpyLaplacianOfGaussianStrategy(EdgeEnhancementStrategyLeaf):
    backend_provider = CellProfilerBackendProvider.NATIVE
    method = EdgeMethod.LOG
    direction = EdgeDirection.ALL

    def enhance(self, request: EdgeEnhancementRequest) -> np.ndarray:
        import centrosome.filter

        size = int(request.sigma * 4) + 1
        return centrosome.filter.laplacian_of_gaussian(
            request.image,
            request.mask,
            size,
            request.sigma,
        )


class NumpyCannyStrategy(EdgeEnhancementStrategyLeaf):
    backend_provider = CellProfilerBackendProvider.NATIVE
    method = EdgeMethod.CANNY
    direction = EdgeDirection.ALL

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


class NumpyRobertsStrategy(EdgeEnhancementStrategyLeaf):
    backend_provider = CellProfilerBackendProvider.NATIVE
    method = EdgeMethod.ROBERTS
    direction = EdgeDirection.ALL

    def enhance(self, request: EdgeEnhancementRequest) -> np.ndarray:
        import centrosome.filter

        return centrosome.filter.roberts(request.image, request.mask)


class NumpyKirschStrategy(EdgeEnhancementStrategyLeaf):
    backend_provider = CellProfilerBackendProvider.NATIVE
    method = EdgeMethod.KIRSCH
    direction = EdgeDirection.ALL

    def enhance(self, request: EdgeEnhancementRequest) -> np.ndarray:
        import centrosome.kirsch

        return centrosome.kirsch.kirsch(request.image)


@njit(cache=True, parallel=True)
def _sobel_numba_kernel(
    image: np.ndarray,
    mask: np.ndarray,
    include_horizontal: bool,
    include_vertical: bool,
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

            if include_horizontal and include_vertical:
                output[row, col] = np.sqrt(horizontal * horizontal + vertical * vertical)
            elif include_horizontal:
                output[row, col] = horizontal
            elif include_vertical:
                output[row, col] = vertical
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
