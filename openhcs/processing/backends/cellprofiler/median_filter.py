"""Median-filter backends for CellProfiler-compatible processing."""

from __future__ import annotations
from typing import Any, Callable
from openhcs.interop.cellprofiler.semantic_defaults import (
    SourceCallKeyword,
    SourceCallKeywordDefaultContract,
    SourceVolumetricPixelDataExecutionContract,
)
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    parse_cellprofiler_int,
)
from openhcs.interop.cellprofiler.module_declarations import (
    ProcessingContract,
    BinderSettingsSourceModule,
    BoundModuleSettings,
    CellProfilerModule,
    ImageArtifactInputModule,
    ImageArtifactOutputModule,
    ModuleSettingsSourceModule,
    ScopedMeasurementModule,
    StructuringElementSettingsModule,
)
from openhcs.interop.cellprofiler.setting_names import (
    optional_setting_value,
    required_setting_value,
    setting_values,
    split_symbol_names,
)
from openhcs.interop.cellprofiler.cellprofiler_literals import (
    cellprofiler_enum_from_literal,
)
from openhcs.processing.backends.cellprofiler.thresholding import (
    ThresholdSettingsModule,
)


class MedianFilterSemanticDefaultContract(SourceCallKeywordDefaultContract):
    contract_key = "MedianFilter.semantic_defaults"
    source_filename = "medianfilter.py"

    def source_call_keywords(self) -> tuple[SourceCallKeyword, ...]:
        return (
            SourceCallKeyword(
                callable_name="medianfilter",
                keyword_name="mode",
                absorbed_callable=medianfilter,
            ),
        )


class MedianFilterExecutionDomainContract(SourceVolumetricPixelDataExecutionContract):
    contract_key = "MedianFilter.execution_domain"
    source_filename = "medianfilter.py"
    callable_name = "medianfilter"

    @property
    def absorbed_callable(self) -> Callable[..., Any]:
        return medianfilter


class MedianfilterModule(
    ImageArtifactInputModule, ImageArtifactOutputModule, CellProfilerModule
):
    module_name = "Medianfilter"
    function_name = "medianfilter"
    validated = True
    confidence = 1.0
    image_input_settings = ("Select the input image",)
    image_output_settings = ("Name the output image",)
    semantic_default_contract_types = (
        MedianFilterSemanticDefaultContract,
        MedianFilterExecutionDomainContract,
    )
    semantic_default_contract_module_name = "MedianFilter"
    setting_bindings = (
        SettingToKeywordBinding("Window", "window_size", parse_cellprofiler_int),
    )


from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar
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
from openhcs.core.runtime_values import image_payload_data, with_image_payload_data
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    CellProfilerBackendAuthority,
)
from openhcs.processing.backends.cellprofiler.perf_fixtures import (
    capture_array_fixture,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract

CONSTANT_PADDING_MODE = "constant"
REFLECT_PADDING_MODE = "reflect"


@dataclass(frozen=True, slots=True)
class VectorizedMedianFilterPlan:
    """Execution plan for exact vectorized 3-D constant-mode median filtering."""

    image_shape: tuple[int, int, int]
    window_shape: tuple[int, int, int]
    window_volume: int
    median_rank: int
    chunk_plane_capacity: int


@dataclass(frozen=True, slots=True)
class VectorizedMedianFilterMemoryPolicy:
    """Memory policy for exact vectorized median windows."""

    max_window_bytes: int
    max_chunk_bytes: int

    def __post_init__(self) -> None:
        if self.max_window_bytes < 1:
            raise ValueError("max_window_bytes must be positive.")
        if self.max_chunk_bytes < 1:
            raise ValueError("max_chunk_bytes must be positive.")

    def plan(
        self, image: np.ndarray, *, window_size: int, mode: str
    ) -> VectorizedMedianFilterPlan | None:
        """Return an exact vectorized plan when the image fits this policy."""
        if image.ndim != 3 or mode != CONSTANT_PADDING_MODE:
            return None
        if not np.issubdtype(image.dtype, np.number):
            return None
        if np.issubdtype(image.dtype, np.floating) and (not np.all(np.isfinite(image))):
            return None
        image_shape = self.image_shape_3d(image)
        window_shape = (int(window_size),) * image.ndim
        window_volume = int(np.prod(window_shape))
        working_set_bytes = int(image.size) * window_volume * image.dtype.itemsize
        if working_set_bytes > self.max_window_bytes:
            return None
        plane_window_bytes = (
            image_shape[1] * image_shape[2] * window_volume * image.dtype.itemsize
        )
        return VectorizedMedianFilterPlan(
            image_shape=image_shape,
            window_shape=window_shape,
            window_volume=window_volume,
            median_rank=window_volume // 2,
            chunk_plane_capacity=max(
                1, int(self.max_chunk_bytes // plane_window_bytes)
            ),
        )

    @staticmethod
    def image_shape_3d(image: np.ndarray) -> tuple[int, int, int]:
        """Return a validated 3-D image shape for vectorized planning."""
        shape = tuple((int(axis_size) for axis_size in image.shape))
        if len(shape) != 3:
            raise ValueError(
                f"Vectorized median filtering requires a 3-D image shape, got {shape!r}."
            )
        if min(shape) < 1:
            raise ValueError(
                f"Vectorized median filtering requires non-empty axes, got {shape!r}."
            )
        return shape


class MedianFilterBackendStrategy(
    CellProfilerBackendStrategyMixin, ABC, metaclass=AutoRegisterMeta
):
    """Median filtering operations keyed by OpenHCS memory type/provider."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def filter(self, image: np.ndarray, *, window_size: int, mode: str) -> np.ndarray:
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

    vectorized_memory_policy: ClassVar[VectorizedMedianFilterMemoryPolicy] = (
        VectorizedMedianFilterMemoryPolicy(
            max_window_bytes=1024**3, max_chunk_bytes=16 * 1024**2
        )
    )
    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY, CellProfilerBackendProvider.NATIVE
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NATIVE
    is_default_backend = True

    def prepare_backend(self) -> None:
        """Warm this backend's exact vectorized 3-D median path."""
        image = np.linspace(0.0, 1.0, 8 * 16 * 16, dtype=np.float32).reshape(
            (8, 16, 16)
        )
        self.filter(image, window_size=5, mode=CONSTANT_PADDING_MODE)

    def filter(self, image: np.ndarray, *, window_size: int, mode: str) -> np.ndarray:
        data = np.asarray(image)
        normalized_window = self.normalized_window_size(window_size)
        capture_array_fixture(
            "medianfilter_input",
            image=data,
            window_size=np.asarray(normalized_window, dtype=np.int64),
            mode=np.asarray(str(mode)),
        )
        if normalized_window <= 1:
            return data
        if np.issubdtype(data.dtype, np.floating):
            large_constant_volume = (
                data.ndim == 3
                and mode == CONSTANT_PADDING_MODE
                and data.size >= np.iinfo(np.uint16).max
            )
            if large_constant_volume:
                accelerated = self.rank_order_filter(data, normalized_window, mode)
                if accelerated is not None:
                    return accelerated
                return self.scipy_filter(data, normalized_window, mode)
            accelerated = self.vectorized_window_filter(data, normalized_window, mode)
            if accelerated is not None:
                return accelerated
            accelerated = self.rank_order_filter(data, normalized_window, mode)
            if accelerated is not None:
                return accelerated
        else:
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
        normalized_window = self.normalized_window_size(
            int(kwargs.get("window_size", 3))
        )
        if normalized_window <= 1:
            return list(slices_2d)
        mode = kwargs.get("mode", CONSTANT_PADDING_MODE)
        outputs = [
            self.filter(np.asarray(slice_2d), window_size=normalized_window, mode=mode)
            for slice_2d in slices_2d
        ]
        return outputs

    def vectorized_window_filter(
        self, image: np.ndarray, window_size: int, mode: str
    ) -> np.ndarray | None:
        """Return an exact constant-mode median using NumPy's vectorized partition."""
        plan = self.vectorized_memory_policy.plan(
            image, window_size=window_size, mode=mode
        )
        if plan is None:
            return None
        from numpy.lib.stride_tricks import sliding_window_view

        pad_width = int(window_size) // 2
        padded = np.pad(image, pad_width, mode=CONSTANT_PADDING_MODE, constant_values=0)
        if plan.image_shape[0] <= plan.chunk_plane_capacity:
            windows = sliding_window_view(padded, plan.window_shape)
            flattened_windows = windows.reshape(
                plan.image_shape + (plan.window_volume,)
            )
            filtered = np.partition(flattened_windows, plan.median_rank, axis=-1)[
                ..., plan.median_rank
            ]
            return filtered.astype(image.dtype, copy=False)
        filtered = np.empty_like(image)
        for z_start in range(0, plan.image_shape[0], plan.chunk_plane_capacity):
            z_stop = min(z_start + plan.chunk_plane_capacity, plan.image_shape[0])
            chunk = padded[z_start : z_stop + window_size - 1]
            windows = sliding_window_view(chunk, plan.window_shape)
            flattened_windows = windows.reshape(
                (
                    z_stop - z_start,
                    plan.image_shape[1],
                    plan.image_shape[2],
                    plan.window_volume,
                )
            )
            filtered[z_start:z_stop] = np.partition(
                flattened_windows, plan.median_rank, axis=-1
            )[..., plan.median_rank]
        return filtered.astype(image.dtype, copy=False)

    def scipy_filter(
        self, image: np.ndarray, window_size: int, mode: str
    ) -> np.ndarray:
        """Return SciPy's median filter result for the requested domain."""
        from scipy.ndimage import median_filter as scipy_median_filter

        filtered = scipy_median_filter(image, size=int(window_size), mode=mode)
        return filtered.astype(image.dtype, copy=False)

    def opencv_filter_2d(
        self, image: np.ndarray, window_size: int, mode: str
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
            cv2_input, pad_width, mode=CONSTANT_PADDING_MODE, constant_values=0
        )
        filtered = cv2.medianBlur(padded, int(window_size))[
            pad_width:-pad_width, pad_width:-pad_width
        ]
        return filtered.astype(image.dtype, copy=False)

    def rank_order_filter(
        self, image: np.ndarray, window_size: int, mode: str
    ) -> np.ndarray | None:
        """Return an exact rank-median result for finite constant-mode volumes."""
        if image.ndim != 3 or mode != CONSTANT_PADDING_MODE:
            return None
        if not np.issubdtype(image.dtype, np.integer) and (
            not np.issubdtype(image.dtype, np.floating)
        ):
            return None
        if np.issubdtype(image.dtype, np.floating) and (not np.all(np.isfinite(image))):
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
            codes, pad_width, mode=CONSTANT_PADDING_MODE, constant_values=0
        )
        filtered_codes = rank.median(
            padded_codes,
            footprint=np.ones((window_size, window_size, window_size), dtype=bool),
        )
        cropped_codes = filtered_codes[
            pad_width:-pad_width, pad_width:-pad_width, pad_width:-pad_width
        ]
        return levels[cropped_codes].astype(image.dtype, copy=False)


def median_filter_backend(
    *, backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION
) -> MedianFilterBackendStrategy:
    """Return the selected median-filter backend."""
    return MedianFilterBackendStrategy.for_memory_type(
        MemoryType.NUMPY, backend_provider=backend_provider
    )


@runtime_image_execution_mode(ImagePayloadExecutionMode.FULL_STACK)
@numpy(contract=ProcessingContract.FLEXIBLE)
def medianfilter(
    image: np.ndarray, window_size: int = 3, mode: str = CONSTANT_PADDING_MODE
) -> np.ndarray:
    """Apply CellProfiler-compatible median filtering."""
    pixel_data = image_payload_data(image)
    filtered = median_filter_backend().filter(
        np.asarray(pixel_data), window_size=int(window_size), mode=str(mode)
    )
    return with_image_payload_data(image, filtered)


def prepare_medianfilter() -> None:
    """Warm the median-filter module path before timed execution."""
    MedianFilterBackendStrategy.prepare_registered_family()


pure_2d_batch_executor(median_filter_backend().filter_batch)(medianfilter)
medianfilter.__openhcs_prepare__ = prepare_medianfilter
__all__ = public_names_from_objects(
    MedianFilterBackendStrategy,
    NumpyMedianFilterBackendStrategy,
    median_filter_backend,
    medianfilter,
)
