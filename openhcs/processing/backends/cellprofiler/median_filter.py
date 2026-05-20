"""Median-filter backends for CellProfiler-compatible processing."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from metaclass_registry import AutoRegisterMeta

from openhcs.constants.constants import MemoryType
from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.callable_contract import runtime_image_execution_mode
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import (
    RuntimePure2DSliceBatchRequest,
    pure_2d_batch_executor,
)
from openhcs.core.public_api import public_names_from_objects
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    CellProfilerBackendAuthority,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract

CONSTANT_PADDING_MODE = "constant"
REFLECT_PADDING_MODE = "reflect"


class MedianFilterBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Median filtering operations keyed by OpenHCS memory type/provider."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def filter(
        self,
        image: np.ndarray,
        *,
        window_size: int,
        mode: str,
    ) -> np.ndarray:
        """Return a CellProfiler-compatible median-filtered image."""

    @abstractmethod
    def filter_batch(self, request: RuntimePure2DSliceBatchRequest) -> list[np.ndarray]:
        """Return median-filtered 2-D slices for a runtime batch."""

    @staticmethod
    def normalized_window_size(window_size: int) -> int:
        """Return CellProfiler's odd positive median-filter window size."""
        normalized = int(window_size)
        if normalized % 2 == 0:
            normalized += 1
        return normalized


class NumpyMedianFilterBackendStrategy(MedianFilterBackendStrategy):
    """NumPy/SciPy median filtering with exact accelerated rank paths."""

    max_vectorized_window_bytes = 1024**3
    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NATIVE,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NATIVE
    is_default_backend = True

    def filter(
        self,
        image: np.ndarray,
        *,
        window_size: int,
        mode: str,
    ) -> np.ndarray:
        data = np.asarray(image)
        normalized_window = self.normalized_window_size(window_size)
        if normalized_window <= 1:
            return data
        accelerated = self.rank_order_filter(data, normalized_window, mode)
        if accelerated is not None:
            return accelerated
        accelerated = self.vectorized_window_filter(data, normalized_window, mode)
        if accelerated is not None:
            return accelerated
        if data.ndim == 2:
            accelerated_2d = self.opencv_filter_2d(data, normalized_window, mode)
            if accelerated_2d is not None:
                return accelerated_2d
        return self.scipy_filter(data, normalized_window, mode)

    def filter_batch(self, request: RuntimePure2DSliceBatchRequest) -> list[np.ndarray]:
        slices_2d = request.slices_2d
        kwargs = request.kwargs
        normalized_window = self.normalized_window_size(int(kwargs.get("window_size", 3)))
        if normalized_window <= 1:
            return list(slices_2d)
        mode = kwargs.get("mode", CONSTANT_PADDING_MODE)
        outputs = [
            self.filter(np.asarray(slice_2d), window_size=normalized_window, mode=mode)
            for slice_2d in slices_2d
        ]
        return outputs

    def vectorized_window_filter(
        self,
        image: np.ndarray,
        window_size: int,
        mode: str,
    ) -> np.ndarray | None:
        """Return an exact constant-mode median using NumPy's vectorized partition."""
        if image.ndim != 3 or mode != CONSTANT_PADDING_MODE:
            return None
        if not np.issubdtype(image.dtype, np.number):
            return None
        if np.issubdtype(image.dtype, np.floating) and not np.all(np.isfinite(image)):
            return None

        window_shape = (int(window_size),) * image.ndim
        window_volume = int(np.prod(window_shape))
        working_set_bytes = int(image.size) * window_volume * image.dtype.itemsize
        if working_set_bytes > self.max_vectorized_window_bytes:
            return None

        from numpy.lib.stride_tricks import sliding_window_view

        pad_width = int(window_size) // 2
        padded = np.pad(image, pad_width, mode=CONSTANT_PADDING_MODE, constant_values=0)
        windows = sliding_window_view(padded, window_shape)
        flattened_windows = windows.reshape(image.shape + (window_volume,))
        median_rank = window_volume // 2
        filtered = np.partition(flattened_windows, median_rank, axis=-1)[..., median_rank]
        return filtered.astype(image.dtype, copy=False)

    def scipy_filter(
        self,
        image: np.ndarray,
        window_size: int,
        mode: str,
    ) -> np.ndarray:
        """Return SciPy's median filter result for the requested domain."""
        from scipy.ndimage import median_filter as scipy_median_filter

        filtered = scipy_median_filter(image, size=int(window_size), mode=mode)
        return filtered.astype(image.dtype, copy=False)

    def opencv_filter_2d(
        self,
        image: np.ndarray,
        window_size: int,
        mode: str,
    ) -> np.ndarray | None:
        """Return OpenCV's exact 2-D median result when its border mode matches."""
        if mode not in {CONSTANT_PADDING_MODE, REFLECT_PADDING_MODE}:
            return None
        if image.dtype not in (np.uint8, np.uint16, np.float32, np.float64):
            return None
        try:
            import cv2
        except ImportError:
            return None

        cv2_input_dtype = np.float32 if image.dtype == np.float64 else image.dtype
        cv2_input = np.ascontiguousarray(image, dtype=cv2_input_dtype)
        if mode == REFLECT_PADDING_MODE:
            filtered = cv2.medianBlur(cv2_input, int(window_size))
            return filtered.astype(image.dtype, copy=False)

        pad_width = int(window_size) // 2
        padded = np.pad(
            cv2_input,
            pad_width,
            mode=CONSTANT_PADDING_MODE,
            constant_values=0,
        )
        filtered = cv2.medianBlur(padded, int(window_size))[
            pad_width:-pad_width,
            pad_width:-pad_width,
        ]
        return filtered.astype(image.dtype, copy=False)

    def rank_order_filter(
        self,
        image: np.ndarray,
        window_size: int,
        mode: str,
    ) -> np.ndarray | None:
        """Return an exact rank-median result for finite constant-mode volumes."""
        if image.ndim != 3 or mode != CONSTANT_PADDING_MODE:
            return None
        if not np.issubdtype(image.dtype, np.integer) and not np.issubdtype(
            image.dtype,
            np.floating,
        ):
            return None
        if np.issubdtype(image.dtype, np.floating) and not np.all(np.isfinite(image)):
            return None
        try:
            from skimage.filters import rank
        except ImportError:
            return None

        zero = np.array([0], dtype=image.dtype)
        levels = np.unique(np.concatenate((zero, image.reshape(-1))))
        if levels.size > np.iinfo(np.uint16).max + 1:
            return None
        codes = np.searchsorted(levels, image).astype(np.uint16)
        pad_width = int(window_size) // 2
        padded_codes = np.pad(
            codes,
            pad_width,
            mode=CONSTANT_PADDING_MODE,
            constant_values=0,
        )
        filtered_codes = rank.median(
            padded_codes,
            footprint=np.ones((window_size, window_size, window_size), dtype=bool),
        )
        cropped_codes = filtered_codes[
            pad_width:-pad_width,
            pad_width:-pad_width,
            pad_width:-pad_width,
        ]
        return levels[cropped_codes].astype(image.dtype, copy=False)


def median_filter_backend(
    *,
    backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> MedianFilterBackendStrategy:
    """Return the selected median-filter backend."""
    return MedianFilterBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=backend_provider,
    )


@runtime_image_execution_mode(ImagePayloadExecutionMode.FULL_STACK)
@numpy(contract=ProcessingContract.PURE_2D)
def medianfilter(
    image: np.ndarray,
    window_size: int = 3,
    mode: str = CONSTANT_PADDING_MODE,
) -> np.ndarray:
    """Apply CellProfiler-compatible median filtering."""
    return median_filter_backend().filter(
        np.asarray(image),
        window_size=int(window_size),
        mode=str(mode),
    )


pure_2d_batch_executor(median_filter_backend().filter_batch)(medianfilter)


__all__ = public_names_from_objects(
    MedianFilterBackendStrategy,
    NumpyMedianFilterBackendStrategy,
    median_filter_backend,
    medianfilter,
)
