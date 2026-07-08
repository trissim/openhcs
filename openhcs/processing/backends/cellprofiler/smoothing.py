"""Shared CellProfiler mask-normalized smoothing primitives."""

from __future__ import annotations
from collections.abc import Callable
from typing import Any
from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.callable_contract import runtime_image_execution_mode
from openhcs.core.runtime_values import image_payload_data, with_image_payload_data
from openhcs.interop.cellprofiler.semantic_defaults import (
    SourceVolumetricPixelDataExecutionContract,
)
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    parse_cellprofiler_float,
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


class ReduceNoiseExecutionDomainContract(SourceVolumetricPixelDataExecutionContract):
    contract_key = "ReduceNoise.execution_domain"
    source_filename = "reducenoise.py"
    callable_name = "reducenoise"

    @property
    def absorbed_callable(self) -> Callable[..., Any]:
        return reducenoise


class ReducenoiseModule(
    ImageArtifactInputModule, ImageArtifactOutputModule, CellProfilerModule
):
    module_name = "Reducenoise"
    aliases = ("ReduceNoise",)
    function_name = "reducenoise"
    validated = True
    confidence = 1.0
    image_input_settings = ("Select the input image",)
    image_output_settings = ("Name the output image",)
    semantic_default_contract_types = (ReduceNoiseExecutionDomainContract,)
    semantic_default_contract_module_name = "ReduceNoise"
    setting_bindings = (
        SettingToKeywordBinding("Size", "patch_size", parse_cellprofiler_int),
        SettingToKeywordBinding("Distance", "patch_distance", parse_cellprofiler_int),
        SettingToKeywordBinding(
            "Cut-off distance", "cutoff_distance", parse_cellprofiler_float
        ),
    )


class SmoothModule(
    ImageArtifactInputModule, ImageArtifactOutputModule, CellProfilerModule
):
    module_name = "Smooth"
    function_name = "smooth"
    validated = True
    confidence = 1.0
    image_input_settings = ("Select the input image",)
    image_output_settings = ("Name the output image",)
    setting_bindings = (
        SettingToKeywordBinding("Select smoothing method", "smoothing_method"),
        SettingToKeywordBinding(
            "Calculate artifact diameter automatically?", "auto_object_size"
        ),
        SettingToKeywordBinding("Typical artifact diameter", "object_size"),
        SettingToKeywordBinding(
            "Edge intensity difference", "edge_intensity_difference"
        ),
        SettingToKeywordBinding("Clip intensities to 0 and 1?", "clip_polynomial"),
    )


from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar
import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit
from openhcs.core.callable_contract import processing_prepare
from openhcs.core.memory.decorators import numpy as numpy_decorator
from openhcs.core.pipeline.function_contracts import (
    RuntimePure2DSliceBatchRequest,
    pure_2d_batch_executor,
)
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.runtime_values import (
    RuntimeImagePayloadContext,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
)
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
    CellProfilerBackendAuthority,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


class SmoothingMethod(Enum):
    """CellProfiler Smooth method names."""

    FIT_POLYNOMIAL = "fit_polynomial"
    GAUSSIAN_FILTER = "gaussian_filter"
    MEDIAN_FILTER = "median_filter"
    SMOOTH_KEEPING_EDGES = "smooth_keeping_edges"
    CIRCULAR_AVERAGE_FILTER = "circular_average_filter"
    SMOOTH_TO_AVERAGE = "smooth_to_average"


@dataclass(frozen=True, slots=True)
class SmoothingStrategyKey:
    """Provider/method key for CellProfiler Smooth strategy registration."""

    backend_provider: CellProfilerBackendProvider
    method: SmoothingMethod

    @property
    def label(self) -> str:
        return f"{self.backend_provider.value}:{self.method.value}"


@dataclass(frozen=True, slots=True)
class SmoothingRequest:
    """Resolved CellProfiler Smooth execution request."""

    pixel_data: np.ndarray
    mask: np.ndarray | None
    backend_provider: CellProfilerBackendProvider
    method: SmoothingMethod
    object_size: float
    sigma: float
    edge_intensity_difference: float
    clip_polynomial: bool

    @property
    def strategy_key(self) -> SmoothingStrategyKey:
        return SmoothingStrategyKey(self.backend_provider, self.method)


@dataclass(frozen=True, slots=True)
class SmoothingBackendSelectionRequest:
    """CellProfiler Smooth settings needed to select an equivalent backend."""

    method: SmoothingMethod
    auto_object_size: bool
    object_size: float
    image_shape: tuple[int, int]

    @property
    def effective_object_size(self) -> float:
        if not self.auto_object_size:
            return self.object_size
        calculated_size = max(1.0, float(np.mean(self.image_shape)) / 40.0)
        return min(30.0, calculated_size)

    @property
    def sigma(self) -> float:
        return self.effective_object_size / 2.35


class SmoothingBackendProviderPolicy(
    EnumKeyedStrategyMixin[SmoothingMethod], ABC, metaclass=AutoRegisterMeta
):
    """Nominal SSOT for CellProfiler Smooth default backend semantics."""

    __registry_key__ = "strategy_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "method"
    method: ClassVar[SmoothingMethod]
    strategy_label: ClassVar[str | None] = None

    @classmethod
    def resolve(
        cls,
        method: SmoothingMethod,
        backend_provider: BackendProviderInput,
        selection_request: SmoothingBackendSelectionRequest | None = None,
    ) -> CellProfilerBackendProvider:
        selection = CellProfilerBackendAuthority.provider_selection(backend_provider)
        return selection.provider_or(
            cls.for_enum_member(method).default_provider(selection_request)
        )

    @abstractmethod
    def default_provider(
        self, selection_request: SmoothingBackendSelectionRequest | None
    ) -> CellProfilerBackendProvider:
        """Return CP-compatible default provider for this Smooth method."""


class NativeSmoothingBackendProviderPolicy(SmoothingBackendProviderPolicy):
    """Smooth methods whose CP reference implementation is the native Python path."""

    def default_provider(
        self, selection_request: SmoothingBackendSelectionRequest | None
    ) -> CellProfilerBackendProvider:
        del selection_request
        return CellProfilerBackendProvider.NATIVE


class GaussianSmoothingBackendProviderPolicy(SmoothingBackendProviderPolicy):
    """Gaussian Smooth backend selection by numerical equivalence class."""

    method = SmoothingMethod.GAUSSIAN_FILTER
    opencv_equivalent_min_sigma: ClassVar[float] = 4.0

    def default_provider(
        self, selection_request: SmoothingBackendSelectionRequest | None
    ) -> CellProfilerBackendProvider:
        if (
            selection_request is not None
            and selection_request.sigma >= self.opencv_equivalent_min_sigma
        ):
            return CellProfilerBackendProvider.OPENCV
        return CellProfilerBackendProvider.NATIVE


class MedianSmoothingBackendProviderPolicy(NativeSmoothingBackendProviderPolicy):
    method = SmoothingMethod.MEDIAN_FILTER


class EdgePreservingSmoothingBackendProviderPolicy(
    NativeSmoothingBackendProviderPolicy
):
    method = SmoothingMethod.SMOOTH_KEEPING_EDGES


class PolynomialSmoothingBackendProviderPolicy(NativeSmoothingBackendProviderPolicy):
    method = SmoothingMethod.FIT_POLYNOMIAL


class CircularAverageSmoothingBackendProviderPolicy(
    NativeSmoothingBackendProviderPolicy
):
    method = SmoothingMethod.CIRCULAR_AVERAGE_FILTER


class SmoothToAverageBackendProviderPolicy(NativeSmoothingBackendProviderPolicy):
    method = SmoothingMethod.SMOOTH_TO_AVERAGE


class SmoothingStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal dispatch for CellProfiler smoothing algorithms."""

    __registry_key__ = "strategy_label"
    __skip_if_no_key__ = True
    strategy_label: ClassVar[str | None] = None
    strategy_key: ClassVar[SmoothingStrategyKey | None] = None

    @classmethod
    def for_request(cls, request: SmoothingRequest) -> "SmoothingStrategy":
        return cls.for_key(request.strategy_key)

    @classmethod
    def for_key(cls, strategy_key: SmoothingStrategyKey) -> "SmoothingStrategy":
        strategy_type = cls.__registry__.get(strategy_key.label)
        if strategy_type is None:
            raise NotImplementedError(
                f"No CellProfiler smoothing backend is registered for provider {strategy_key.backend_provider.value!r} and method {strategy_key.method.value!r}."
            )
        return strategy_type()

    @property
    def supports_stack_batch(self) -> bool:
        return False

    def smooth_stack(
        self, pixel_stack: np.ndarray, mask_stack: np.ndarray | None, sigma: float
    ) -> np.ndarray:
        raise NotImplementedError(
            f"{type(self).__name__} does not declare stack-batch smoothing support."
        )

    @abstractmethod
    def smooth(self, request: SmoothingRequest) -> np.ndarray:
        """Return smoothed pixels for this strategy."""


class SmoothingStrategyLeaf(SmoothingStrategy):
    """Declarative base for concrete smoothing leaves."""

    backend_provider: ClassVar[CellProfilerBackendProvider | None] = None
    method: ClassVar[SmoothingMethod | None] = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls.backend_provider is None or cls.method is None:
            return
        cls.strategy_key = SmoothingStrategyKey(cls.backend_provider, cls.method)
        cls.strategy_label = cls.strategy_key.label


@dataclass(frozen=True, slots=True)
class GaussianKernel1D:
    """One-dimensional constant-boundary Gaussian kernel authority."""

    sigma: float

    @property
    def array(self) -> np.ndarray:
        sigma = float(self.sigma)
        if sigma <= 0:
            return np.ones((1,), dtype=np.float64)
        radius = max(1, int(round(4.0 * sigma)))
        coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (coordinates / sigma) ** 2)
        kernel /= np.sum(kernel)
        return kernel.astype(np.float64, copy=False)


@dataclass(frozen=True, slots=True)
class MaskedFilterRequest(ABC):
    """Shared pixel/mask state for mask-normalized filtering."""

    pixels: np.ndarray
    mask: np.ndarray | None

    @property
    def resolved_mask(self) -> np.ndarray:
        if self.mask is None:
            return np.ones(self.pixels.shape, dtype=bool)
        return np.asarray(self.mask, dtype=bool)


@dataclass(frozen=True, slots=True)
class MaskedLinearFilterRequest(MaskedFilterRequest):
    """Mask-normalized linear filtering request."""

    operation: Callable[[np.ndarray], np.ndarray]

    def apply(self) -> np.ndarray:
        mask = self.resolved_mask
        masked_image = np.zeros(self.pixels.shape, dtype=self.pixels.dtype)
        masked_image[mask] = self.pixels[mask]
        weights = self.operation(mask.astype(float))
        filtered = self.operation(masked_image)
        return filtered / (weights + np.finfo(float).eps)


@dataclass(frozen=True, slots=True)
class OpenCVMaskedGaussianFilterRequest(MaskedFilterRequest):
    """OpenCV implementation of mask-normalized Gaussian smoothing."""

    sigma: float

    @property
    def image_array(self) -> np.ndarray:
        return np.ascontiguousarray(self.pixels, dtype=np.float32)

    @property
    def mask_array(self) -> np.ndarray:
        image_array = self.image_array
        if self.mask is None:
            return np.ones(image_array.shape, dtype=np.float32)
        mask_bool = np.asarray(self.mask, dtype=bool)
        if mask_bool.shape != image_array.shape:
            raise ValueError(
                f"Smoothing mask must match image shape; got {mask_bool.shape!r} for image {image_array.shape!r}."
            )
        return np.ascontiguousarray(mask_bool.astype(np.float32))

    def apply(self) -> np.ndarray:
        import cv2

        image_array = self.image_array
        mask_array = self.mask_array
        kernel = GaussianKernel1D(self.sigma).array.astype(np.float32, copy=False)
        masked_image = np.zeros(image_array.shape, dtype=np.float32)
        np.copyto(masked_image, image_array, where=mask_array.astype(bool, copy=False))
        filtered = cv2.sepFilter2D(
            masked_image, cv2.CV_32F, kernel, kernel, borderType=cv2.BORDER_CONSTANT
        )
        weights = cv2.sepFilter2D(
            mask_array, cv2.CV_32F, kernel, kernel, borderType=cv2.BORDER_CONSTANT
        )
        return filtered / (weights + np.finfo(np.float32).eps)

    @classmethod
    def apply_stack(
        cls, pixel_stack: np.ndarray, mask_stack: np.ndarray | None, sigma: float
    ) -> np.ndarray:
        return np.stack(
            [
                cls(
                    pixel_stack[index],
                    None if mask_stack is None else mask_stack[index],
                    sigma,
                ).apply()
                for index in range(pixel_stack.shape[0])
            ],
            axis=0,
        )


class NumbaGaussianSmoothingStrategy(SmoothingStrategyLeaf):
    backend_provider = CellProfilerBackendProvider.NUMBA
    method = SmoothingMethod.GAUSSIAN_FILTER

    def smooth(self, request: SmoothingRequest) -> np.ndarray:
        return _gaussian_filter_numba(request.pixel_data, request.mask, request.sigma)


class NumpyGaussianSmoothingStrategy(SmoothingStrategyLeaf):
    backend_provider = CellProfilerBackendProvider.NATIVE
    method = SmoothingMethod.GAUSSIAN_FILTER

    @property
    def supports_stack_batch(self) -> bool:
        return True

    def smooth(self, request: SmoothingRequest) -> np.ndarray:
        from scipy.ndimage import gaussian_filter

        return MaskedLinearFilterRequest(
            pixels=request.pixel_data,
            mask=request.mask,
            operation=lambda image: gaussian_filter(
                image, request.sigma, mode="constant", cval=0
            ),
        ).apply()

    def smooth_stack(
        self, pixel_stack: np.ndarray, mask_stack: np.ndarray | None, sigma: float
    ) -> np.ndarray:
        from scipy.ndimage import gaussian_filter

        return MaskedLinearFilterRequest(
            pixels=pixel_stack,
            mask=mask_stack,
            operation=lambda image: gaussian_filter(
                image, (0.0, sigma, sigma), mode="constant", cval=0
            ),
        ).apply()


class OpenCVGaussianSmoothingStrategy(SmoothingStrategyLeaf):
    backend_provider = CellProfilerBackendProvider.OPENCV
    method = SmoothingMethod.GAUSSIAN_FILTER

    @property
    def supports_stack_batch(self) -> bool:
        return True

    def smooth(self, request: SmoothingRequest) -> np.ndarray:
        return OpenCVMaskedGaussianFilterRequest(
            request.pixel_data, request.mask, request.sigma
        ).apply()

    def smooth_stack(
        self, pixel_stack: np.ndarray, mask_stack: np.ndarray | None, sigma: float
    ) -> np.ndarray:
        return OpenCVMaskedGaussianFilterRequest.apply_stack(
            pixel_stack, mask_stack, sigma
        )


class MedianSmoothingStrategy(SmoothingStrategyLeaf):
    backend_provider = CellProfilerBackendProvider.NATIVE
    method = SmoothingMethod.MEDIAN_FILTER

    def smooth(self, request: SmoothingRequest) -> np.ndarray:
        import centrosome.filter

        return centrosome.filter.median_filter(
            request.pixel_data, request.mask, request.object_size / 2 + 1
        )


class EdgePreservingSmoothingStrategy(SmoothingStrategyLeaf):
    backend_provider = CellProfilerBackendProvider.NATIVE
    method = SmoothingMethod.SMOOTH_KEEPING_EDGES

    def smooth(self, request: SmoothingRequest) -> np.ndarray:
        from skimage.restoration import denoise_bilateral

        return denoise_bilateral(
            image=request.pixel_data.astype(float),
            channel_axis=None,
            sigma_color=request.edge_intensity_difference,
            sigma_spatial=request.sigma,
        )


class PolynomialSmoothingStrategy(SmoothingStrategyLeaf):
    backend_provider = CellProfilerBackendProvider.NATIVE
    method = SmoothingMethod.FIT_POLYNOMIAL

    def smooth(self, request: SmoothingRequest) -> np.ndarray:
        return _fit_polynomial(
            request.pixel_data, request.mask, request.clip_polynomial
        )


class CircularAverageSmoothingStrategy(SmoothingStrategyLeaf):
    backend_provider = CellProfilerBackendProvider.NATIVE
    method = SmoothingMethod.CIRCULAR_AVERAGE_FILTER

    def smooth(self, request: SmoothingRequest) -> np.ndarray:
        import centrosome.filter

        return centrosome.filter.circular_average_filter(
            request.pixel_data, request.object_size / 2 + 1, request.mask
        )


class SmoothToAverageStrategy(SmoothingStrategyLeaf):
    backend_provider = CellProfilerBackendProvider.NATIVE
    method = SmoothingMethod.SMOOTH_TO_AVERAGE

    def smooth(self, request: SmoothingRequest) -> np.ndarray:
        if request.mask is None:
            mean_value = np.mean(request.pixel_data)
        else:
            mean_value = np.mean(request.pixel_data[request.mask])
        return np.full(request.pixel_data.shape, mean_value, dtype=np.float32)


def _gaussian_filter_numba(
    image: np.ndarray, mask: np.ndarray | None, sigma: float
) -> np.ndarray:
    image_array = np.asarray(image, dtype=np.float32)
    if image_array.ndim != 2:
        raise NotImplementedError(
            "Numba smoothing backend currently supports 2-D Gaussian planes."
        )
    kernel = GaussianKernel1D(sigma).array
    contiguous_image = np.ascontiguousarray(image_array)
    if mask is None:
        return _separable_gaussian_normalized_constant_2d_numba(
            contiguous_image, kernel
        )
    mask_array = np.asarray(mask, dtype=np.bool_)
    if mask_array.shape != image_array.shape:
        raise ValueError(
            f"Smoothing mask must match image shape; got {mask_array.shape!r} for image {image_array.shape!r}."
        )
    return _masked_separable_gaussian_constant_2d_numba(
        contiguous_image, np.ascontiguousarray(mask_array), kernel
    )


@njit(cache=True)
def _separable_gaussian_normalized_constant_2d_numba(
    image: np.ndarray, kernel: np.ndarray
) -> np.ndarray:
    height, width = image.shape
    radius = kernel.size // 2
    temp = np.zeros((height, width), dtype=np.float64)
    x_weights = np.zeros(width, dtype=np.float64)
    y_weights = np.zeros(height, dtype=np.float64)
    output = np.zeros((height, width), dtype=np.float32)
    for row in range(height):
        for col in range(width):
            value = 0.0
            weight = 0.0
            for offset in range(kernel.size):
                source_col = col + offset - radius
                if 0 <= source_col < width:
                    kernel_value = kernel[offset]
                    value += float(image[row, source_col]) * kernel_value
                    weight += kernel_value
            temp[row, col] = value
            if row == 0:
                x_weights[col] = weight
    for row in range(height):
        y_weight = 0.0
        for offset in range(kernel.size):
            source_row = row + offset - radius
            if 0 <= source_row < height:
                y_weight += kernel[offset]
        y_weights[row] = y_weight
        for col in range(width):
            value = 0.0
            for offset in range(kernel.size):
                source_row = row + offset - radius
                if 0 <= source_row < height:
                    value += temp[source_row, col] * kernel[offset]
            output[row, col] = value / (
                x_weights[col] * y_weights[row] + np.finfo(np.float64).eps
            )
    return output


@njit(cache=True)
def _masked_separable_gaussian_constant_2d_numba(
    image: np.ndarray, mask: np.ndarray, kernel: np.ndarray
) -> np.ndarray:
    height, width = image.shape
    radius = kernel.size // 2
    temp_values = np.zeros((height, width), dtype=np.float64)
    temp_weights = np.zeros((height, width), dtype=np.float64)
    output = np.zeros((height, width), dtype=np.float32)
    eps = np.finfo(np.float64).eps
    for row in range(height):
        for col in range(width):
            weighted_value = 0.0
            weight = 0.0
            for offset in range(kernel.size):
                source_col = col + offset - radius
                if 0 <= source_col < width and mask[row, source_col]:
                    kernel_value = kernel[offset]
                    weighted_value += float(image[row, source_col]) * kernel_value
                    weight += kernel_value
            temp_values[row, col] = weighted_value
            temp_weights[row, col] = weight
    for row in range(height):
        for col in range(width):
            weighted_value = 0.0
            weight = 0.0
            for offset in range(kernel.size):
                source_row = row + offset - radius
                if 0 <= source_row < height:
                    kernel_value = kernel[offset]
                    weighted_value += temp_values[source_row, col] * kernel_value
                    weight += temp_weights[source_row, col] * kernel_value
            output[row, col] = weighted_value / (weight + eps)
    return output


def _fit_polynomial(
    image: np.ndarray, mask: np.ndarray | None, clip: bool
) -> np.ndarray:
    if mask is None:
        mask = np.ones(image.shape, dtype=bool)
    valid = np.asarray(mask, dtype=bool) & (image > 0)
    if not np.any(valid):
        return image
    x, y = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    terms = (x, y, x * x, y * y, x * y, np.ones(image.shape))
    design = np.column_stack([term[valid].ravel() for term in terms])
    coeffs, *_ = np.linalg.lstsq(design, image[valid].ravel(), rcond=None)
    output = np.sum([coeff * term for coeff, term in zip(coeffs, terms)], axis=0)
    if clip:
        output = np.clip(output, 0, 1)
    return output


def smooth_image(
    image: np.ndarray,
    smoothing_method: SmoothingMethod | str = SmoothingMethod.GAUSSIAN_FILTER,
    auto_object_size: bool = True,
    object_size: float = 16.0,
    edge_intensity_difference: float = 0.1,
    clip_polynomial: bool = True,
    smoothing_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> np.ndarray:
    """Smooth one image payload using CellProfiler-compatible semantics."""
    smoothing_method = coerce_cellprofiler_enum(SmoothingMethod, smoothing_method)
    pixel_data = np.asarray(image_payload_data(image), dtype=np.float32)
    backend_provider = SmoothingBackendProviderPolicy.resolve(
        smoothing_method,
        smoothing_backend_provider,
        SmoothingBackendSelectionRequest(
            method=smoothing_method,
            auto_object_size=auto_object_size,
            object_size=float(object_size),
            image_shape=tuple((int(axis) for axis in pixel_data.shape)),
        ),
    )
    mask = image_payload_mask(image)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
    selection = SmoothingBackendSelectionRequest(
        method=smoothing_method,
        auto_object_size=auto_object_size,
        object_size=float(object_size),
        image_shape=tuple((int(axis) for axis in pixel_data.shape)),
    )
    request = SmoothingRequest(
        pixel_data=pixel_data,
        mask=mask,
        backend_provider=backend_provider,
        method=smoothing_method,
        object_size=float(selection.effective_object_size),
        sigma=float(selection.sigma),
        edge_intensity_difference=float(edge_intensity_difference),
        clip_polynomial=bool(clip_polynomial),
    )
    output = SmoothingStrategy.for_request(request).smooth(request)
    return RuntimeImagePayloadContext(
        np.asarray(output, dtype=np.float32),
        mask=mask,
        metadata=image_payload_metadata(image).without_unit_interval_intensity_scale(),
    ).payload()


@numpy_decorator(contract=ProcessingContract.PURE_2D)
def smooth(
    image: np.ndarray,
    smoothing_method: SmoothingMethod = SmoothingMethod.GAUSSIAN_FILTER,
    auto_object_size: bool = True,
    object_size: float = 16.0,
    edge_intensity_difference: float = 0.1,
    clip_polynomial: bool = True,
    smoothing_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> np.ndarray:
    """Smooth an image using CellProfiler-compatible filtering methods."""
    return smooth_image(
        image=image,
        smoothing_method=smoothing_method,
        auto_object_size=auto_object_size,
        object_size=object_size,
        edge_intensity_difference=edge_intensity_difference,
        clip_polynomial=clip_polynomial,
        smoothing_backend_provider=smoothing_backend_provider,
    )


@runtime_image_execution_mode(ImagePayloadExecutionMode.FULL_STACK)
@numpy_decorator(contract=ProcessingContract.FLEXIBLE)
def reducenoise(
    image: np.ndarray,
    patch_size: int = 5,
    patch_distance: int = 6,
    cutoff_distance: float = 0.1,
) -> np.ndarray:
    """Reduce image noise using CellProfiler-compatible non-local means."""
    from skimage.restoration import denoise_nl_means

    image_data = image_payload_data(image)
    if image_data.dtype != np.float32 and image_data.dtype != np.float64:
        image_data = image_data.astype(np.float32)
    denoised = denoise_nl_means(
        image_data,
        h=cutoff_distance,
        patch_size=patch_size,
        patch_distance=patch_distance,
        fast_mode=True,
        channel_axis=None,
    )
    return with_image_payload_data(image, denoised.astype(np.float32))


def smooth_batch(request: RuntimePure2DSliceBatchRequest) -> list[Any]:
    slices_2d = request.slices_2d
    kwargs = request.kwargs
    smoothing_method = coerce_cellprofiler_enum(
        SmoothingMethod, kwargs.get("smoothing_method", SmoothingMethod.GAUSSIAN_FILTER)
    )
    pixel_stack = np.ascontiguousarray(
        np.stack(
            [
                np.asarray(image_payload_data(slice_2d), dtype=np.float32)
                for slice_2d in slices_2d
            ],
            axis=0,
        )
    )
    selection_request = SmoothingBackendSelectionRequest(
        method=smoothing_method,
        auto_object_size=bool(kwargs.get("auto_object_size", True)),
        object_size=float(kwargs.get("object_size", 16.0)),
        image_shape=tuple((int(axis) for axis in pixel_stack.shape[1:])),
    )
    backend_provider = SmoothingBackendProviderPolicy.resolve(
        smoothing_method,
        kwargs.get(
            "smoothing_backend_provider", DEFAULT_CELLPROFILER_BACKEND_SELECTION
        ),
        selection_request,
    )
    strategy = SmoothingStrategy.for_key(
        SmoothingStrategyKey(backend_provider, smoothing_method)
    )
    if not strategy.supports_stack_batch:
        return [
            request.execute_one(slice_index)
            for slice_index in range(request.slice_count)
        ]
    masks = tuple((image_payload_mask(slice_2d) for slice_2d in slices_2d))
    mask_stack = None
    if any((mask is not None for mask in masks)):
        mask_stack = np.stack(
            [
                (
                    np.ones(pixel_stack.shape[1:], dtype=bool)
                    if mask is None
                    else np.asarray(mask, dtype=bool)
                )
                for mask in masks
            ],
            axis=0,
        )
    output_stack = strategy.smooth_stack(
        pixel_stack, mask_stack, float(selection_request.sigma)
    ).astype(np.float32, copy=False)
    return [
        RuntimeImagePayloadContext(
            output_stack[slice_index],
            mask=masks[slice_index],
            metadata=image_payload_metadata(
                slice_2d
            ).without_unit_interval_intensity_scale(),
        ).payload()
        for slice_index, slice_2d in enumerate(slices_2d)
    ]


@processing_prepare(smooth)
def prepare_smooth() -> None:
    """Compile default Gaussian smoothing before timed execution."""
    image = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape((64, 64))
    smooth.__wrapped__(image)


pure_2d_batch_executor(smooth_batch)(smooth)
__all__ = public_names_from_objects(
    CircularAverageSmoothingBackendProviderPolicy,
    EdgePreservingSmoothingBackendProviderPolicy,
    GaussianKernel1D,
    GaussianSmoothingBackendProviderPolicy,
    MaskedFilterRequest,
    MaskedLinearFilterRequest,
    MedianSmoothingBackendProviderPolicy,
    OpenCVMaskedGaussianFilterRequest,
    PolynomialSmoothingBackendProviderPolicy,
    SmoothToAverageBackendProviderPolicy,
    SmoothingBackendProviderPolicy,
    SmoothingBackendSelectionRequest,
    SmoothingMethod,
    SmoothingRequest,
    SmoothingStrategy,
    SmoothingStrategyKey,
    prepare_smooth,
    reducenoise,
    smooth,
    smooth_batch,
    smooth_image,
)
