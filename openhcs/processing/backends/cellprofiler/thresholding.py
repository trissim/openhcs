"""Threshold diagnostic backends for CellProfiler-compatible processing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from enum import Enum
from functools import lru_cache
import math
import time
from typing import Callable, ClassVar

import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit
import scipy.interpolate

from openhcs.constants.constants import MemoryType
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.runtime_values import (
    ImagePayloadMetadata,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    image_payload_with_context,
    image_intensity_scale_for_dtype,
    normalize_image_payload_intensity,
)
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    cellprofiler_backend_key,
)
from openhcs.processing.backends.cellprofiler.perf_fixtures import (
    capture_array_fixture,
    capture_enabled,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.materialization import csv_materializer


CELLPROFILER_BASIC_THRESHOLD_SMOOTHING_SCALE = 1.3488
THRESHOLD_BACKEND_REGISTRY_KEY = "backend_key"
SCIPY_CONSTANT_BOUNDARY_MODE = "constant"
CELLPROFILER_THRESHOLD_SMOOTHING_TRUNCATE_SIGMAS = 4.0
CELLPROFILER_THRESHOLD_SMOOTHING_HALF_MASS_FACTOR = 0.6744
CELLPROFILER_LI_TOLERANCE = 0.5 / 65536.0
CELLPROFILER_THRESHOLD_ENTROPY_DELTA = 2.0 ** -8
CELLPROFILER_THRESHOLD_ENTROPY_BINS = 256
CELLPROFILER_MULTI_OTSU_BINS = 128
CELLPROFILER_LOG_MULTI_OTSU_BINS = 128
CELLPROFILER_LOG_MULTI_OTSU_BIN_CENTER_OFFSET = 0.0


class CellProfilerThresholdAssignment(Enum):
    """Closed foreground/background assignment for multi-class CP thresholds."""

    FOREGROUND = "Foreground"
    BACKGROUND = "Background"


class CellProfilerAveragingMethod(Enum):
    """Closed CP robust-background center estimators."""

    MEAN = "Mean"
    MEDIAN = "Median"
    MODE = "Mode"


class CellProfilerThresholdMethod(Enum):
    """Closed CP threshold methods with global-threshold source semantics."""

    OTSU = ("Otsu", True, False)
    MINIMUM_CROSS_ENTROPY = ("Minimum Cross-Entropy", True, False)
    ROBUST_BACKGROUND = ("Robust Background", False, False)
    MULTI_OTSU = ("Multi-Otsu", False, True)
    SAUVOLA = ("Sauvola", False, False)
    MAX_INTENSITY_PERCENTAGE = ("Max Intensity Percentage", False, False)
    MANUAL = ("Manual", False, False)
    MEASUREMENT = ("Measurement", False, False)
    LI = ("Li", True, False)
    TRIANGLE = ("Triangle", False, False)
    ISODATA = ("Isodata", False, False)

    def __new__(
        cls,
        label: str,
        uses_raw_global_threshold_source: bool,
        uses_raw_global_threshold_source_when_log_transformed: bool,
    ) -> "CellProfilerThresholdMethod":
        member = object.__new__(cls)
        member._value_ = label
        member._uses_raw_global_threshold_source = uses_raw_global_threshold_source
        member._uses_raw_global_threshold_source_when_log_transformed = (
            uses_raw_global_threshold_source_when_log_transformed
        )
        return member

    def global_threshold_selection(
        self,
        *,
        log_transform: bool,
        image: np.ndarray,
        threshold_image: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Return the source image and kwargs for global threshold estimation."""
        if self._uses_raw_global_threshold_source:
            return np.asarray(image), {}
        if self._uses_raw_global_threshold_source_when_log_transformed and log_transform:
            return np.asarray(image), {"nbins": CELLPROFILER_LOG_MULTI_OTSU_BINS}
        return threshold_image, {}


class CellProfilerOtsuMethod(Enum):
    """Closed CP Otsu class-count selector."""

    TWO_CLASS = "Two classes"
    THREE_CLASS = "Three classes"


class CellProfilerThresholdScope(Enum):
    """Closed CP global/adaptive threshold scope."""

    GLOBAL = "Global"
    ADAPTIVE = "Adaptive"


class CellProfilerVarianceMethod(Enum):
    """Closed CP robust-background spread estimators."""

    STANDARD_DEVIATION = "Standard deviation"
    MEDIAN_ABSOLUTE_DEVIATION = "Median absolute deviation"


class RobustBackgroundCenterStrategy(
    EnumKeyedStrategyMixin[CellProfilerAveragingMethod],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal center estimator for CP robust-background thresholding."""

    __registry_key__ = "averaging_method_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "averaging_method"
    __enum_label_attr__ = "averaging_method_label"

    averaging_method: ClassVar[CellProfilerAveragingMethod | None] = None
    averaging_method_label: ClassVar[str | None] = None

    @classmethod
    def for_averaging_method(
        cls,
        averaging_method: CellProfilerAveragingMethod | str,
    ) -> "RobustBackgroundCenterStrategy":
        resolved = coerce_cellprofiler_enum(
            CellProfilerAveragingMethod,
            averaging_method,
        )
        return cls.for_enum_member(resolved)

    def center(self, values: np.ndarray) -> float:
        """Return the robust-background center for trimmed values."""
        return float(type(self)._center(values))

    @staticmethod
    @abstractmethod
    def _center(values: np.ndarray) -> float:
        """Return the strategy-specific center estimate."""


@dataclass(frozen=True)
class CellProfilerThresholdProfiler:
    """Bound profiler for the CellProfiler threshold execution timeline."""

    sink: Callable[..., None]
    function_name: str = "cellprofiler_threshold"

    def record(self, phase_name: str, phase_started_at: float, **metadata: object) -> None:
        self.sink(
            phase_name,
            time.perf_counter() - phase_started_at,
            function=self.function_name,
            **metadata,
        )

    def record_method(
        self,
        phase_name: str,
        phase_started_at: float,
        threshold_method: CellProfilerThresholdMethod,
    ) -> None:
        self.record(phase_name, phase_started_at, method=threshold_method.value)

    def record_global_raw(
        self,
        phase_started_at: float,
        threshold_method: CellProfilerThresholdMethod,
        selection_image: np.ndarray,
    ) -> None:
        self.record(
            "threshold_global_raw",
            phase_started_at,
            method=threshold_method.value,
            pixels=np.asarray(selection_image).size,
        )

    def record_apply(self, phase_started_at: float, smoothing: float) -> None:
        self.record(
            "threshold_apply",
            phase_started_at,
            smoothing=float(smoothing),
        )


class MeanRobustBackgroundCenterStrategy(RobustBackgroundCenterStrategy):
    averaging_method = CellProfilerAveragingMethod.MEAN

    @staticmethod
    def _center(values: np.ndarray) -> float:
        return float(np.mean(values))


class MedianRobustBackgroundCenterStrategy(RobustBackgroundCenterStrategy):
    averaging_method = CellProfilerAveragingMethod.MEDIAN

    @staticmethod
    def _center(values: np.ndarray) -> float:
        return float(np.median(values))


class ModeRobustBackgroundCenterStrategy(RobustBackgroundCenterStrategy):
    averaging_method = CellProfilerAveragingMethod.MODE

    @staticmethod
    def _center(values: np.ndarray) -> float:
        return float(threshold_primitives().binned_mode(values))


class RobustBackgroundSpreadStrategy(
    EnumKeyedStrategyMixin[CellProfilerVarianceMethod],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal spread estimator for CP robust-background thresholding."""

    __registry_key__ = "variance_method_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "variance_method"
    __enum_label_attr__ = "variance_method_label"

    variance_method: ClassVar[CellProfilerVarianceMethod | None] = None
    variance_method_label: ClassVar[str | None] = None

    @classmethod
    def for_variance_method(
        cls,
        variance_method: CellProfilerVarianceMethod | str,
    ) -> "RobustBackgroundSpreadStrategy":
        resolved = coerce_cellprofiler_enum(
            CellProfilerVarianceMethod,
            variance_method,
        )
        return cls.for_enum_member(resolved)

    @abstractmethod
    def spread(self, values: np.ndarray) -> float:
        """Return the robust-background spread for trimmed values."""


class StandardDeviationRobustBackgroundSpreadStrategy(RobustBackgroundSpreadStrategy):
    variance_method = CellProfilerVarianceMethod.STANDARD_DEVIATION

    def spread(self, values: np.ndarray) -> float:
        return float(np.std(values))


class MedianAbsoluteDeviationRobustBackgroundSpreadStrategy(
    RobustBackgroundSpreadStrategy
):
    variance_method = CellProfilerVarianceMethod.MEDIAN_ABSOLUTE_DEVIATION

    def spread(self, values: np.ndarray) -> float:
        return float(threshold_primitives().mad(values))


@dataclass(frozen=True, slots=True)
class CellProfilerThresholdDiagnostics:
    """CellProfiler threshold measurements emitted as runtime facts."""

    final_threshold: float
    original_threshold: float
    weighted_variance: float
    sum_of_entropies: float


@dataclass(frozen=True, slots=True)
class RobustBackgroundThresholdSettings:
    """Settings meaningful to CP robust-background thresholding."""

    lower_outlier_fraction: float
    upper_outlier_fraction: float
    averaging_method: CellProfilerAveragingMethod
    variance_method: CellProfilerVarianceMethod
    number_of_deviations: float

    def as_kwargs(self) -> dict[str, object]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


def normalize_cellprofiler_image(image: np.ndarray) -> np.ndarray:
    """Return an image in CellProfiler's normalized pixel-data convention."""
    return image_payload_data(normalize_image_payload_intensity(image, dtype=np.float32))


def unit_interval_scale_for_threshold_diagnostics(
    image_data: np.ndarray,
    metadata: ImagePayloadMetadata,
) -> int | None:
    """Return a proof scale for exact unit-interval threshold diagnostics."""
    metadata_scale = metadata.unit_interval_intensity_scale_for_channel(0)
    if metadata_scale is not None and metadata_scale > 1:
        return int(metadata_scale)
    image_array = np.asarray(image_data)
    if not np.issubdtype(image_array.dtype, np.integer):
        return None
    scale = image_intensity_scale_for_dtype(image_array.dtype)
    if scale is None or scale <= 1:
        return None
    return int(scale)


@dataclass(frozen=True, slots=True)
class QuantizedThresholdLogTables:
    """Log lookup tables for CellProfiler threshold diagnostics."""

    values: np.ndarray
    weighted_log_values: np.ndarray
    entropy_log_values: np.ndarray
    entropy_log_delta_values: np.ndarray


@dataclass(frozen=True, slots=True)
class RectangularMaskDomain:
    """The true region of a mask that is exactly one filled 2D rectangle."""

    y: slice
    x: slice

    @property
    def slices(self) -> tuple[slice, slice]:
        return self.y, self.x


@dataclass(frozen=True, slots=True)
class ThresholdApplicationSmoothing:
    """CellProfiler threshold-application smoothing policy."""

    smoothing: float

    @property
    def sigma(self) -> float:
        return float(self.smoothing) / CELLPROFILER_BASIC_THRESHOLD_SMOOTHING_SCALE

    @property
    def enabled(self) -> bool:
        return self.sigma > 0.0

    def gaussian_filter(self, array: np.ndarray) -> np.ndarray:
        from scipy import ndimage as ndi

        return ndi.gaussian_filter(
            array,
            sigma=self.sigma,
            mode=SCIPY_CONSTANT_BOUNDARY_MODE,
            cval=0,
            truncate=4.0,
        )

    @staticmethod
    @lru_cache(maxsize=32)
    def full_mask_weight(shape: tuple[int, int], sigma: float) -> np.ndarray:
        from scipy import ndimage as ndi

        return ndi.gaussian_filter(
            np.ones(shape, dtype=np.float64),
            sigma=sigma,
            mode=SCIPY_CONSTANT_BOUNDARY_MODE,
            cval=0,
            truncate=4.0,
        )

    def smooth(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
        """Return the image CellProfiler thresholds against after estimation."""
        if not self.enabled:
            return np.asarray(image), 0.0

        image_array = np.asarray(image, dtype=np.float64)
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != image_array.shape:
            raise ValueError(
                "Threshold application mask must match the image shape; got "
                f"mask {mask_array.shape!r} for image {image_array.shape!r}."
            )

        full_mask = bool(np.all(mask_array))
        capture_array_fixture(
            "threshold_application",
            image=image_array,
            mask=mask_array,
            smoothing=np.asarray(self.smoothing, dtype=np.float64),
        )
        masked_image = image_array if full_mask else np.where(mask_array, image_array, 0.0)
        smoothed_image = self.gaussian_filter(masked_image)
        mask_weight = (
            self.full_mask_weight(image_array.shape, self.sigma)
            if full_mask
            else self.gaussian_filter(mask_array.astype(np.float64))
        )
        if full_mask:
            smoothed_image /= mask_weight
            return smoothed_image, self.sigma

        output = np.zeros_like(image_array)
        valid = mask_weight != 0
        output[valid] = smoothed_image[valid] / mask_weight[valid]
        return output, self.sigma


@dataclass(frozen=True, slots=True)
class ThresholdApplicationRequest:
    """Executable CellProfiler threshold application request."""

    image: np.ndarray
    threshold: float | np.ndarray
    mask: np.ndarray | None = None
    smoothing: float = 0.0

    @property
    def resolved_mask(self) -> np.ndarray:
        return (
            np.full(np.asarray(self.image).shape, True)
            if self.mask is None
            else np.asarray(self.mask, dtype=bool)
        )

    def apply(self) -> tuple[np.ndarray, float]:
        if self.smoothing == 0:
            thresholded = np.asarray(self.image) >= self.threshold
            if self.mask is None:
                return thresholded, 0.0
            return thresholded & self.resolved_mask, 0.0

        blurred_image, sigma = ThresholdApplicationSmoothing(self.smoothing).smooth(
            self.image,
            self.resolved_mask,
        )
        return (blurred_image >= self.threshold) & self.resolved_mask, sigma


class ThresholdSmoothingBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Memory-backend-specific threshold smoothing."""

    __registry_key__ = THRESHOLD_BACKEND_REGISTRY_KEY
    __skip_if_no_key__ = True

    @abstractmethod
    def smooth_threshold_image(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        smoothing: float,
        threshold_method: object | None = None,
        log_transform: bool = False,
    ) -> tuple[np.ndarray, float]:
        """Return the smoothed image and effective Gaussian sigma."""


class NumbaNumpyThresholdSmoothingBackendStrategy(ThresholdSmoothingBackendStrategy):
    """NumPy-memory threshold smoothing with Numba convolution."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def prepare_backend(self) -> None:
        image = np.linspace(0.0, 1.0, 16, dtype=np.float64).reshape((4, 4))
        mask = np.ones(image.shape, dtype=np.bool_)
        self.smooth_threshold_image(image, mask, 1.0)

    def smooth_threshold_image(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        smoothing: float,
        threshold_method: object | None = None,
        log_transform: bool = False,
    ) -> tuple[np.ndarray, float]:
        image_array = np.asarray(image, dtype=np.float64)
        mask_array = np.asarray(mask, dtype=np.bool_)
        if image_array.ndim != 2:
            raise NotImplementedError(
                "CellProfiler threshold smoothing currently supports 2-D "
                f"NumPy planes, got shape {image_array.shape!r}."
            )
        if mask_array.shape != image_array.shape:
            raise ValueError(
                "Threshold smoothing mask must match the image shape; got "
                f"mask {mask_array.shape!r} for image {image_array.shape!r}."
            )
        sigma, kernel = _threshold_smoothing_kernel(
            smoothing,
            threshold_method,
            log_transform=log_transform,
        )
        return _masked_kernel_convolution_2d_numba(
            image_array,
            mask_array,
            kernel,
        ), sigma


def _threshold_smoothing_kernel(
    smoothing: float,
    threshold_method: object | None,
    *,
    log_transform: bool = False,
) -> tuple[float, np.ndarray]:
    sigma, radius = _threshold_smoothing_kernel_parameters(smoothing)
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    y, x = np.meshgrid(coordinates, coordinates, indexing="ij")
    radius_squared = float(radius * radius)
    distance_squared = x * x + y * y
    # CellProfiler's public Centrosome primitive names this parameter ``sd``,
    # but its circular Gaussian uses twice that value in the exponent.
    effective_sigma = 2.0 * sigma
    kernel = np.exp(-0.5 * distance_squared / (effective_sigma * effective_sigma))
    kernel[distance_squared > radius_squared] = 0.0
    kernel /= np.sum(kernel)
    return sigma, kernel.astype(np.float64, copy=False)


def _threshold_smoothing_kernel_parameters(smoothing: float) -> tuple[float, int]:
    sigma = float(smoothing) / CELLPROFILER_THRESHOLD_SMOOTHING_HALF_MASS_FACTOR
    radius = max(
        1,
        int(math.ceil(sigma * CELLPROFILER_THRESHOLD_SMOOTHING_TRUNCATE_SIGMAS)),
    )
    return sigma, radius


@njit(cache=True)
def _masked_kernel_convolution_2d_numba(
    image: np.ndarray,
    mask: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    height, width = image.shape
    kernel_height, kernel_width = kernel.shape
    center_y = kernel_height // 2
    center_x = kernel_width // 2
    output = np.empty((height, width), dtype=np.float64)
    eps = np.finfo(np.float64).eps

    for y in range(height):
        for x in range(width):
            weighted_sum = 0.0
            weight = 0.0
            for ky in range(kernel_height):
                iy = y + ky - center_y
                if iy < 0 or iy >= height:
                    continue
                for kx in range(kernel_width):
                    ix = x + kx - center_x
                    if ix < 0 or ix >= width or not mask[iy, ix]:
                        continue
                    kernel_value = kernel[ky, kx]
                    weight += kernel_value
                    weighted_sum += image[iy, ix] * kernel_value
            output[y, x] = weighted_sum / (weight + eps)
    return output


class ThresholdDiagnosticsBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Memory-backend-specific threshold diagnostic measurements."""

    __registry_key__ = THRESHOLD_BACKEND_REGISTRY_KEY
    __skip_if_no_key__ = True

    def diagnostics(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
        binary_image: np.ndarray,
        *,
        proven_unit_interval_scale: int | None = None,
    ) -> tuple[float, float]:
        """Return weighted variance and sum of entropies."""
        return (
            self.weighted_variance(image, mask, binary_image),
            self.sum_of_entropies(image, mask, binary_image),
        )

    @abstractmethod
    def weighted_variance(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        binary_image: np.ndarray,
    ) -> float:
        """Compute weighted foreground/background log variance."""

    @abstractmethod
    def sum_of_entropies(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        binary_image: np.ndarray,
    ) -> float:
        """Compute foreground plus background log-histogram entropy."""


class NumpyThresholdDiagnosticsBackendStrategy(ThresholdDiagnosticsBackendStrategy):
    """Independent NumPy implementation of CellProfiler threshold diagnostics."""

    backend_key = cellprofiler_backend_key(MemoryType.NUMPY)
    memory_type = MemoryType.NUMPY
    is_default_backend = False

    def weighted_variance(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        binary_image: np.ndarray,
    ) -> float:
        return _numpy_threshold_weighted_variance(image, mask, binary_image)

    def sum_of_entropies(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        binary_image: np.ndarray,
    ) -> float:
        return _numpy_threshold_sum_of_entropies(image, mask, binary_image)


class ThresholdDiagnosticDomain(Enum):
    """Nominal image domains for CellProfiler threshold diagnostics."""

    PLANAR_IMAGE = "planar_image"
    ND_IMAGE = "nd_image"


@dataclass(frozen=True, slots=True)
class ThresholdDiagnosticRequest:
    """Typed inputs for one CellProfiler threshold diagnostic measurement."""

    backend: "NumbaNumpyThresholdDiagnosticsBackendStrategy"
    domain: ThresholdDiagnosticDomain
    image: np.ndarray
    mask: np.ndarray | None
    binary_image: np.ndarray
    proven_unit_interval_scale: int | None

    @classmethod
    def from_inputs(
        cls,
        *,
        backend: "NumbaNumpyThresholdDiagnosticsBackendStrategy",
        image: np.ndarray,
        mask: np.ndarray | None,
        binary_image: np.ndarray,
        proven_unit_interval_scale: int | None,
    ) -> "ThresholdDiagnosticRequest":
        image_array = np.asarray(image, dtype=np.float64)
        binary_array = np.asarray(binary_image, dtype=np.bool_)
        if binary_array.shape != image_array.shape:
            raise ValueError(
                "Threshold diagnostics binary image must match the image shape; got "
                f"binary {binary_array.shape!r} for image {image_array.shape!r}."
            )
        mask_array = None if mask is None else np.asarray(mask, dtype=np.bool_)
        if mask_array is not None and mask_array.shape != image_array.shape:
            raise ValueError(
                "Threshold diagnostics mask must match the image shape; got "
                f"mask {mask_array.shape!r} for image {image_array.shape!r}."
            )
        return cls(
            backend=backend,
            domain=(
                ThresholdDiagnosticDomain.PLANAR_IMAGE
                if image_array.ndim == 2
                else ThresholdDiagnosticDomain.ND_IMAGE
            ),
            image=image_array,
            mask=mask_array,
            binary_image=binary_array,
            proven_unit_interval_scale=proven_unit_interval_scale,
        )

    def full_mask(self) -> np.ndarray:
        """Return an explicit mask in the same domain as the diagnostic image."""
        if self.mask is None:
            return np.ones(self.image.shape, dtype=np.bool_)
        return self.mask


class ThresholdDiagnosticDomainStrategy(ABC, metaclass=AutoRegisterMeta):
    """Apply CellProfiler threshold diagnostics in the correct image domain."""

    __registry_key__ = "domain_key"
    __skip_if_no_key__ = True
    domain: ClassVar[ThresholdDiagnosticDomain | None] = None
    domain_key: ClassVar[str | None] = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        domain = cls.__dict__.get("domain")
        if isinstance(domain, ThresholdDiagnosticDomain):
            cls.domain_key = domain.value

    @classmethod
    def evaluate(cls, request: ThresholdDiagnosticRequest) -> tuple[float, float]:
        strategy_type = cls.__registry__[request.domain.value]
        return strategy_type().diagnostics(request)

    @abstractmethod
    def diagnostics(self, request: ThresholdDiagnosticRequest) -> tuple[float, float]:
        """Return weighted variance and sum of entropies for ``request``."""


class PlanarThresholdDiagnosticDomainStrategy(ThresholdDiagnosticDomainStrategy):
    """Use the optimized 2-D backend path for planar CellProfiler images."""

    domain = ThresholdDiagnosticDomain.PLANAR_IMAGE

    def diagnostics(self, request: ThresholdDiagnosticRequest) -> tuple[float, float]:
        return request.backend.diagnostics_planar(request)


class WholeImageThresholdDiagnosticDomainStrategy(ThresholdDiagnosticDomainStrategy):
    """Measure ND images as one CellProfiler image domain, not per-plane averages."""

    domain = ThresholdDiagnosticDomain.ND_IMAGE

    def diagnostics(self, request: ThresholdDiagnosticRequest) -> tuple[float, float]:
        return request.backend.diagnostics_whole_image(request)


class NumbaNumpyThresholdDiagnosticsBackendStrategy(
    ThresholdDiagnosticsBackendStrategy
):
    """Numba-accelerated NumPy implementation of threshold diagnostics."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def prepare_backend(self) -> None:
        image = np.linspace(0.0, 1.0, 16, dtype=np.float64).reshape((4, 4))
        mask = np.ones(image.shape, dtype=np.bool_)
        binary = image > 0.5
        self.diagnostics(image, mask, binary)
        self.diagnostics(image[None, ...], mask[None, ...], binary[None, ...])

    def diagnostics(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
        binary_image: np.ndarray,
        *,
        proven_unit_interval_scale: int | None = None,
    ) -> tuple[float, float]:
        return ThresholdDiagnosticDomainStrategy.evaluate(
            ThresholdDiagnosticRequest.from_inputs(
                backend=self,
                image=image,
                mask=mask,
                binary_image=binary_image,
                proven_unit_interval_scale=proven_unit_interval_scale,
            )
        )

    def diagnostics_planar(
        self,
        request: ThresholdDiagnosticRequest,
    ) -> tuple[float, float]:
        image_array = request.image
        binary_array = request.binary_image
        proven_unit_interval_scale = request.proven_unit_interval_scale
        if request.mask is None:
            self._validate_unmasked_inputs(image_array, binary_array)
            full_mask = True
        else:
            mask_array = request.mask
            self._validate_inputs(image_array, mask_array, binary_array)
            full_mask = bool(np.all(mask_array))
            if not full_mask:
                mask_domain = _rectangular_mask_domain(mask_array)
                if mask_domain is not None:
                    cropped_image = image_array[mask_domain.slices]
                    if bool(np.all(np.isfinite(cropped_image))):
                        if proven_unit_interval_scale is not None:
                            scale = int(proven_unit_interval_scale)
                            log_tables = _quantized_log_tables(scale)
                            y_slice, x_slice = mask_domain.slices
                            weighted_variance, sum_of_entropies = (
                                _threshold_diagnostics_rectangular_mask_quantized_numba(
                                    np.ascontiguousarray(
                                        np.rint(image_array * scale).astype(np.int64),
                                    ),
                                    np.ascontiguousarray(binary_array),
                                    _deterministic_normal_noise(image_array.shape),
                                    log_tables.values,
                                    log_tables.weighted_log_values,
                                    log_tables.entropy_log_values,
                                    log_tables.entropy_log_delta_values,
                                    int(y_slice.start),
                                    int(y_slice.stop),
                                    int(x_slice.start),
                                    int(x_slice.stop),
                                )
                            )
                            return float(weighted_variance), float(sum_of_entropies)
        if full_mask and bool(np.all(np.isfinite(image_array))):
            if proven_unit_interval_scale is not None:
                scale = int(proven_unit_interval_scale)
                log_tables = _quantized_log_tables(scale)
                weighted_variance, sum_of_entropies = (
                    _threshold_diagnostics_unmasked_finite_quantized_numba(
                        np.ascontiguousarray(
                            np.rint(image_array * scale).astype(np.int64),
                        ),
                        np.ascontiguousarray(binary_array),
                        _deterministic_normal_noise(image_array.shape),
                        log_tables.values,
                        log_tables.weighted_log_values,
                        log_tables.entropy_log_values,
                        log_tables.entropy_log_delta_values,
                    )
                )
                return float(weighted_variance), float(sum_of_entropies)
            weighted_variance, sum_of_entropies = (
                _threshold_diagnostics_unmasked_finite_numba(
                    np.ascontiguousarray(image_array),
                    np.ascontiguousarray(binary_array),
                    _deterministic_normal_noise(image_array.shape),
                )
            )
            return float(weighted_variance), float(sum_of_entropies)
        if request.mask is None:
            mask_array = np.ones(image_array.shape, dtype=np.bool_)
            weighted_variance, sum_of_entropies = (
                _threshold_diagnostics_numba(
                    np.ascontiguousarray(image_array),
                    np.ascontiguousarray(mask_array),
                    np.ascontiguousarray(binary_array),
                    _deterministic_normal_noise(image_array.shape),
                )
            )
            return float(weighted_variance), float(sum_of_entropies)
        weighted_variance, sum_of_entropies = _threshold_diagnostics_numba(
            np.ascontiguousarray(image_array),
            np.ascontiguousarray(mask_array),
            np.ascontiguousarray(binary_array),
            _deterministic_normal_noise(image_array.shape),
        )
        return float(weighted_variance), float(sum_of_entropies)

    def diagnostics_whole_image(
        self,
        request: ThresholdDiagnosticRequest,
    ) -> tuple[float, float]:
        """Evaluate an ND CellProfiler image as one flattened measurement domain."""
        image_array = request.image
        binary_array = request.binary_image
        mask_array = request.full_mask()
        flat_image = np.ascontiguousarray(image_array.reshape(-1, 1))
        flat_binary = np.ascontiguousarray(binary_array.reshape(-1, 1))
        flat_mask = np.ascontiguousarray(mask_array.reshape(-1, 1))
        noise = _deterministic_normal_noise(image_array.shape).reshape(-1, 1)

        if bool(np.all(flat_mask)) and bool(np.all(np.isfinite(flat_image))):
            if request.proven_unit_interval_scale is not None:
                scale = int(request.proven_unit_interval_scale)
                log_tables = _quantized_log_tables(scale)
                weighted_variance, sum_of_entropies = (
                    _threshold_diagnostics_unmasked_finite_quantized_numba(
                        np.ascontiguousarray(
                            np.rint(flat_image * scale).astype(np.int64),
                        ),
                        flat_binary,
                        noise,
                        log_tables.values,
                        log_tables.weighted_log_values,
                        log_tables.entropy_log_values,
                        log_tables.entropy_log_delta_values,
                    )
                )
                return float(weighted_variance), float(sum_of_entropies)
            weighted_variance, sum_of_entropies = (
                _threshold_diagnostics_unmasked_finite_numba(
                    flat_image,
                    flat_binary,
                    noise,
                )
            )
            return float(weighted_variance), float(sum_of_entropies)

        weighted_variance, sum_of_entropies = _threshold_diagnostics_numba(
            flat_image,
            flat_mask,
            flat_binary,
            noise,
        )
        return float(weighted_variance), float(sum_of_entropies)

    def _validate_unmasked_inputs(
        self,
        image_array: np.ndarray,
        binary_array: np.ndarray,
    ) -> None:
        if image_array.ndim != 2:
            raise NotImplementedError(
                "CellProfiler threshold diagnostics currently support 2-D "
                f"NumPy planes, got shape {image_array.shape!r}."
            )
        if binary_array.shape != image_array.shape:
            raise ValueError(
                "Threshold diagnostics binary image must match the image shape; got "
                f"binary {binary_array.shape!r} for image {image_array.shape!r}."
            )

    def weighted_variance(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        binary_image: np.ndarray,
    ) -> float:
        image_array = np.asarray(image, dtype=np.float64)
        mask_array = np.asarray(mask, dtype=np.bool_)
        binary_array = np.asarray(binary_image, dtype=np.bool_)
        self._validate_inputs(image_array, mask_array, binary_array)
        return float(
            _threshold_weighted_variance_numba(
                np.ascontiguousarray(image_array),
                np.ascontiguousarray(mask_array),
                np.ascontiguousarray(binary_array),
            )
        )

    def sum_of_entropies(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        binary_image: np.ndarray,
    ) -> float:
        image_array = np.asarray(image, dtype=np.float64)
        mask_array = np.asarray(mask, dtype=np.bool_)
        binary_array = np.asarray(binary_image, dtype=np.bool_)
        self._validate_inputs(image_array, mask_array, binary_array)
        return float(
            _threshold_sum_of_entropies_numba(
                np.ascontiguousarray(image_array),
                np.ascontiguousarray(mask_array),
                np.ascontiguousarray(binary_array),
                np.ascontiguousarray(_deterministic_normal_noise(image_array.shape)),
            )
        )

    def _validate_inputs(
        self,
        image_array: np.ndarray,
        mask_array: np.ndarray,
        binary_array: np.ndarray,
    ) -> None:
        if image_array.ndim != 2:
            raise NotImplementedError(
                "CellProfiler threshold diagnostics currently support 2-D "
                f"NumPy planes, got shape {image_array.shape!r}."
            )
        if mask_array.shape != image_array.shape:
            raise ValueError(
                "Threshold diagnostics mask must match the image shape; got "
                f"mask {mask_array.shape!r} for image {image_array.shape!r}."
            )
        if binary_array.shape != image_array.shape:
            raise ValueError(
                "Threshold diagnostics binary image must match the image shape; got "
                f"binary {binary_array.shape!r} for image {image_array.shape!r}."
            )


class ThresholdPrimitiveBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Small threshold helper primitives supplied by an explicit provider."""

    __registry_key__ = THRESHOLD_BACKEND_REGISTRY_KEY
    __skip_if_no_key__ = True

    @abstractmethod
    def log_transform(self, values: np.ndarray) -> tuple[np.ndarray, object]:
        """Return CP-compatible log-transformed values and conversion state."""

    @abstractmethod
    def inverse_log_transform(
        self,
        values: float | np.ndarray,
        conversion: object,
    ) -> float | np.ndarray:
        """Map CP-compatible log-threshold values back to image space."""

    @abstractmethod
    def binned_mode(self, values: np.ndarray) -> float:
        """Return CP-compatible binned mode."""

    @abstractmethod
    def mad(self, values: np.ndarray) -> float:
        """Return CP-compatible median absolute deviation."""

    @abstractmethod
    def otsu_threshold(self, values: np.ndarray) -> float:
        """Return CP-compatible Otsu threshold."""

    @abstractmethod
    def weighted_otsu_threshold(self, values: np.ndarray) -> float:
        """Return CellProfiler's two-class weighted-variance Otsu threshold."""

    @abstractmethod
    def li_threshold(self, values: np.ndarray) -> float:
        """Return Li's minimum cross-entropy threshold."""

    @abstractmethod
    def triangle_threshold(self, values: np.ndarray) -> float:
        """Return triangle-method threshold."""

    @abstractmethod
    def isodata_threshold(self, values: np.ndarray) -> float:
        """Return iterative intermeans threshold."""

    @abstractmethod
    def mean_threshold(self, values: np.ndarray) -> float:
        """Return arithmetic mean threshold."""

    @abstractmethod
    def yen_threshold(self, values: np.ndarray) -> float:
        """Return Yen threshold."""

    @abstractmethod
    def minimum_threshold(self, values: np.ndarray) -> float:
        """Return histogram minimum threshold."""

    @abstractmethod
    def multiotsu_thresholds(self, values: np.ndarray, *, nbins: int) -> np.ndarray:
        """Return two thresholds for CP-compatible three-class Otsu."""

    @abstractmethod
    def sauvola_threshold_image(
        self,
        image: np.ndarray,
        *,
        window_size: int,
    ) -> np.ndarray:
        """Return per-pixel Sauvola thresholds."""

    @abstractmethod
    def minimum_cross_entropy_threshold(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> float:
        """Return CP-compatible minimum cross-entropy threshold."""


@dataclass(frozen=True, slots=True)
class NumbaLogTransformConversion:
    """State needed to invert the CP-style log normalization."""

    noise_min: float
    log_min: float
    log_max: float


class NumbaNumpyThresholdPrimitiveBackendStrategy(ThresholdPrimitiveBackendStrategy):
    """Numba-backed threshold primitives for NumPy-memory images."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def prepare_backend(self) -> None:
        values = np.linspace(0.01, 1.0, 32, dtype=np.float64)
        image = values[:25].reshape((5, 5))
        transformed, conversion = self.log_transform(values.astype(np.float32))
        self.inverse_log_transform(transformed, conversion)
        self.binned_mode(values)
        self.mad(values)
        self.otsu_threshold(values)
        self.weighted_otsu_threshold(values.astype(np.float32))
        self.li_threshold(values)
        self.triangle_threshold(values)
        self.isodata_threshold(values)
        self.mean_threshold(values)
        self.yen_threshold(values)
        self.multiotsu_thresholds(values, nbins=16)
        self.sauvola_threshold_image(image, window_size=3)
        self.minimum_cross_entropy_threshold(image)

    def log_transform(self, values: np.ndarray) -> tuple[np.ndarray, object]:
        values_array = np.asarray(values, dtype=np.float32)
        transformed, noise_min, log_min, log_max = _log_transform_numba(
            np.ascontiguousarray(values_array.ravel()),
        )
        return (
            transformed.reshape(values_array.shape),
            NumbaLogTransformConversion(
                noise_min=float(noise_min),
                log_min=float(log_min),
                log_max=float(log_max),
            ),
        )

    def inverse_log_transform(
        self,
        values: float | np.ndarray,
        conversion: object,
    ) -> float | np.ndarray:
        if not isinstance(conversion, NumbaLogTransformConversion):
            raise TypeError(
                "Numba threshold primitive inverse_log_transform requires "
                "NumbaLogTransformConversion state."
            )
        values_array = np.asarray(values, dtype=np.float64)
        inverted = _inverse_log_transform_numba(
            np.ascontiguousarray(values_array.ravel()),
            conversion.log_min,
            conversion.log_max,
        ).reshape(values_array.shape)
        if np.isscalar(values):
            return float(inverted.reshape(-1)[0])
        return inverted.astype(np.float32, copy=False)

    def binned_mode(self, values: np.ndarray) -> float:
        return float(_binned_mode_numba(_finite_flat_float64(values)))

    def mad(self, values: np.ndarray) -> float:
        return float(_mad_numba(_finite_flat_float64(values)))

    def otsu_threshold(self, values: np.ndarray) -> float:
        return float(_otsu_threshold_numba(_finite_flat_float64(values), 256))

    def weighted_otsu_threshold(self, values: np.ndarray) -> float:
        values_array = np.asarray(values, dtype=np.float32)
        transformed, _noise_min, log_min, log_max = _log_transform_numba(
            np.ascontiguousarray(values_array.ravel()),
        )
        threshold = _weighted_otsu_threshold_numba_compatible(transformed, 256)
        return float(
            _inverse_log_transform_numba(
                np.asarray([threshold], dtype=np.float64),
                log_min,
                log_max,
            )[0]
        )

    def li_threshold(self, values: np.ndarray) -> float:
        values_array = np.asarray(values)
        if values_array.dtype == np.float32:
            finite_values32 = _finite_flat_float32(values_array)
            finite_values = np.ascontiguousarray(finite_values32, dtype=np.float64)
            return float(
                _li_threshold_numba(finite_values, _li_tolerance_numba(finite_values))
            )
        finite_values = _finite_flat_float64(values_array)
        return float(
            _li_threshold_numba(finite_values, _li_tolerance_numba(finite_values))
        )

    def triangle_threshold(self, values: np.ndarray) -> float:
        return float(_triangle_threshold_numba(_finite_flat_float64(values), 256))

    def isodata_threshold(self, values: np.ndarray) -> float:
        return float(_isodata_threshold_numba(_finite_flat_float64(values)))

    def mean_threshold(self, values: np.ndarray) -> float:
        return float(_mean_threshold_numba(_finite_flat_float64(values)))

    def yen_threshold(self, values: np.ndarray) -> float:
        return float(_yen_threshold_numba(_finite_flat_float64(values), 256))

    def minimum_threshold(self, values: np.ndarray) -> float:
        threshold = float(_minimum_threshold_numba(_finite_flat_float64(values), 256))
        if not np.isfinite(threshold):
            raise ValueError(
                "Histogram minimum threshold requires a bimodal histogram."
            )
        return threshold

    def multiotsu_thresholds(self, values: np.ndarray, *, nbins: int) -> np.ndarray:
        return _multiotsu_three_class_thresholds_numba(
            _finite_flat_float64(values),
            int(nbins),
        )

    def sauvola_threshold_image(
        self,
        image: np.ndarray,
        *,
        window_size: int,
    ) -> np.ndarray:
        image_array = np.asarray(image, dtype=np.float64)
        if image_array.ndim != 2:
            raise NotImplementedError(
                "CellProfiler Sauvola thresholding currently supports 2-D "
                f"NumPy planes, got shape {image_array.shape!r}."
            )
        return _sauvola_threshold_image_numba(
            np.ascontiguousarray(image_array),
            int(window_size),
            0.2,
            1.0,
        )

    def minimum_cross_entropy_threshold(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> float:
        image_array = np.asarray(image)
        if mask is None:
            if image_array.dtype == np.float32:
                values32 = _finite_flat_float32(image_array)
                return _li_threshold_float32_numpy(values32)
            values = _finite_flat_float64(image_array)
        else:
            mask_array = np.asarray(mask, dtype=bool)
            if mask_array.shape != image_array.shape:
                raise ValueError(
                    "Minimum cross-entropy mask must match the image shape; got "
                    f"mask {mask_array.shape!r} for image {image_array.shape!r}."
            )
            if image_array.dtype == np.float32:
                values32 = _finite_flat_float32(image_array[mask_array])
                return _li_threshold_float32_numpy(values32)
            values = _finite_flat_float64(image_array[mask_array])
        return float(_li_threshold_numba(values, _li_tolerance_numba(values)))


class CentrosomeNumpyThresholdPrimitiveBackendStrategy(
    ThresholdPrimitiveBackendStrategy
):
    """Centrosome-backed threshold primitives exposed as a backend provider."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.CENTROSOME,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.CENTROSOME
    is_default_backend = False

    def log_transform(self, values: np.ndarray) -> tuple[np.ndarray, object]:
        import centrosome.threshold

        return centrosome.threshold.log_transform(values)

    def inverse_log_transform(
        self,
        values: float | np.ndarray,
        conversion: object,
    ) -> float | np.ndarray:
        import centrosome.threshold

        return centrosome.threshold.inverse_log_transform(values, conversion)

    def binned_mode(self, values: np.ndarray) -> float:
        import centrosome.threshold

        return float(centrosome.threshold.binned_mode(values))

    def mad(self, values: np.ndarray) -> float:
        import centrosome.threshold

        return float(centrosome.threshold.mad(values))

    def otsu_threshold(self, values: np.ndarray) -> float:
        import centrosome.threshold

        _, global_threshold = centrosome.threshold.get_threshold(
            centrosome.threshold.TM_OTSU,
            centrosome.threshold.TM_GLOBAL,
            np.asarray(values, dtype=np.float32),
            two_class_otsu=True,
            use_weighted_variance=True,
            assign_middle_to_foreground=True,
        )
        return float(global_threshold)

    def weighted_otsu_threshold(self, values: np.ndarray) -> float:
        return self.otsu_threshold(values)

    def li_threshold(self, values: np.ndarray) -> float:
        raise NotImplementedError(
            "Centrosome threshold primitive backend does not provide Li "
            "thresholding. Select the Numba backend explicitly."
        )

    def triangle_threshold(self, values: np.ndarray) -> float:
        raise NotImplementedError(
            "Centrosome threshold primitive backend does not provide Triangle "
            "thresholding. Select the Numba backend explicitly."
        )

    def isodata_threshold(self, values: np.ndarray) -> float:
        raise NotImplementedError(
            "Centrosome threshold primitive backend does not provide Isodata "
            "thresholding. Select the Numba backend explicitly."
        )

    def mean_threshold(self, values: np.ndarray) -> float:
        raise NotImplementedError(
            "Centrosome threshold primitive backend does not provide Mean "
            "thresholding. Select the Numba backend explicitly."
        )

    def yen_threshold(self, values: np.ndarray) -> float:
        raise NotImplementedError(
            "Centrosome threshold primitive backend does not provide Yen "
            "thresholding. Select the Numba backend explicitly."
        )

    def minimum_threshold(self, values: np.ndarray) -> float:
        raise NotImplementedError(
            "Centrosome threshold primitive backend does not provide histogram "
            "Minimum thresholding. Select the Numba backend explicitly."
        )

    def multiotsu_thresholds(self, values: np.ndarray, *, nbins: int) -> np.ndarray:
        raise NotImplementedError(
            "Centrosome threshold primitive backend does not provide Multi-Otsu "
            "thresholding. Select the Numba backend explicitly."
        )

    def sauvola_threshold_image(
        self,
        image: np.ndarray,
        *,
        window_size: int,
    ) -> np.ndarray:
        raise NotImplementedError(
            "Centrosome threshold primitive backend does not provide Sauvola "
            "thresholding. Select the Numba backend explicitly."
        )

    def minimum_cross_entropy_threshold(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> float:
        raise NotImplementedError(
            "Centrosome threshold primitive backend does not provide CP-style "
            "minimum cross-entropy thresholding. Select the Numba backend "
            "explicitly."
        )


def threshold_primitives(
    *,
    backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> ThresholdPrimitiveBackendStrategy:
    """Return the selected threshold primitive backend."""
    return ThresholdPrimitiveBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=backend_provider,
    )


@dataclass(frozen=True, slots=True)
class GlobalThresholdRequest:
    """Context needed by one CellProfiler global threshold method."""

    primitives: ThresholdPrimitiveBackendStrategy
    threshold_image: np.ndarray
    threshold_mask: np.ndarray | None
    values: np.ndarray
    assignment: CellProfilerThresholdAssignment
    log_transform: bool
    kwargs: dict[str, object]


class GlobalThresholdMethodStrategy(
    EnumKeyedStrategyMixin[CellProfilerThresholdMethod],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal implementation for one CellProfiler global threshold method."""

    __registry_key__ = "method_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "method"

    method: ClassVar[CellProfilerThresholdMethod]
    method_label: ClassVar[str | None] = None

    @classmethod
    def for_method(
        cls,
        method: CellProfilerThresholdMethod,
    ) -> "GlobalThresholdMethodStrategy":
        try:
            return cls.for_enum_member(method)
        except KeyError as exc:
            raise NotImplementedError(
                f"Threshold method {method} not supported."
            ) from exc

    @abstractmethod
    def compute(self, request: GlobalThresholdRequest) -> float:
        """Compute the unclipped global threshold."""


class HelperBackedGlobalThresholdStrategy(GlobalThresholdMethodStrategy):
    """Global-threshold strategy whose behavior is declared as a request helper."""

    def compute(self, request: GlobalThresholdRequest) -> float:
        return float(type(self)._threshold_helper(request))

    @staticmethod
    @abstractmethod
    def _threshold_helper(request: GlobalThresholdRequest) -> float:
        """Compute the method-specific raw threshold."""


class MinimumCrossEntropyGlobalThresholdStrategy(GlobalThresholdMethodStrategy):
    method = CellProfilerThresholdMethod.MINIMUM_CROSS_ENTROPY
    method_label = method.value

    def compute(self, request: GlobalThresholdRequest) -> float:
        return request.primitives.minimum_cross_entropy_threshold(
            request.threshold_image,
            request.threshold_mask,
        )


class LiGlobalThresholdStrategy(GlobalThresholdMethodStrategy):
    method = CellProfilerThresholdMethod.LI
    method_label = method.value

    def compute(self, request: GlobalThresholdRequest) -> float:
        return request.primitives.li_threshold(request.values)


class RobustBackgroundGlobalThresholdStrategy(HelperBackedGlobalThresholdStrategy):
    method = CellProfilerThresholdMethod.ROBUST_BACKGROUND
    method_label = method.value

    @staticmethod
    def _threshold_helper(request: GlobalThresholdRequest) -> float:
        return get_threshold_robust_background(
            request.values,
            **request.kwargs,
        )


class OtsuGlobalThresholdStrategy(GlobalThresholdMethodStrategy):
    method = CellProfilerThresholdMethod.OTSU
    method_label = method.value

    def compute(self, request: GlobalThresholdRequest) -> float:
        return request.primitives.otsu_threshold(request.values)


class MultiOtsuGlobalThresholdStrategy(GlobalThresholdMethodStrategy):
    method = CellProfilerThresholdMethod.MULTI_OTSU
    method_label = method.value

    def compute(self, request: GlobalThresholdRequest) -> float:
        bin_wanted = (
            0
            if request.assignment is CellProfilerThresholdAssignment.FOREGROUND
            else 1
        )
        nbins = int(request.kwargs.get("nbins", CELLPROFILER_MULTI_OTSU_BINS))
        thresholds = request.primitives.multiotsu_thresholds(
            request.values,
            nbins=nbins,
        )
        threshold = float(thresholds[bin_wanted])
        if request.log_transform:
            threshold += (
                threshold_histogram_bin_width(request.values, nbins)
                * CELLPROFILER_LOG_MULTI_OTSU_BIN_CENTER_OFFSET
            )
        return threshold


class SauvolaGlobalThresholdStrategy(HelperBackedGlobalThresholdStrategy):
    method = CellProfilerThresholdMethod.SAUVOLA
    method_label = method.value

    @staticmethod
    def _threshold_helper(request: GlobalThresholdRequest) -> float:
        return float(
            np.mean(
                request.primitives.sauvola_threshold_image(
                    request.values.reshape(1, -1),
                    window_size=int(request.kwargs.get("window_size", 15)),
                )
            )
        )


class TriangleGlobalThresholdStrategy(GlobalThresholdMethodStrategy):
    method = CellProfilerThresholdMethod.TRIANGLE
    method_label = method.value

    def compute(self, request: GlobalThresholdRequest) -> float:
        return request.primitives.triangle_threshold(request.values)


class IsodataGlobalThresholdStrategy(GlobalThresholdMethodStrategy):
    method = CellProfilerThresholdMethod.ISODATA
    method_label = method.value

    def compute(self, request: GlobalThresholdRequest) -> float:
        return request.primitives.isodata_threshold(request.values)


class MaxIntensityPercentageGlobalThresholdStrategy(HelperBackedGlobalThresholdStrategy):
    method = CellProfilerThresholdMethod.MAX_INTENSITY_PERCENTAGE
    method_label = method.value

    @staticmethod
    def _threshold_helper(request: GlobalThresholdRequest) -> float:
        return float(
            np.max(request.values) * float(request.kwargs.get("fraction", 0.75))
        )


def cellprofiler_get_global_threshold(
    image: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    threshold_method: CellProfilerThresholdMethod | str = CellProfilerThresholdMethod.OTSU,
    threshold_min: float = 0,
    threshold_max: float = 1,
    threshold_correction_factor: float = 1,
    assign_middle_to_foreground: (
        CellProfilerThresholdAssignment | str
    ) = CellProfilerThresholdAssignment.FOREGROUND,
    log_transform: bool = False,
    **kwargs: object,
) -> float:
    """Compute one global threshold using independent CP-compatible semantics."""
    primitives = threshold_primitives()
    method = coerce_cellprofiler_enum(CellProfilerThresholdMethod, threshold_method)
    assignment = coerce_cellprofiler_enum(
        CellProfilerThresholdAssignment,
        assign_middle_to_foreground,
    )
    threshold_image = np.asarray(image, dtype=np.float32)
    if log_transform:
        threshold_image, conversion = primitives.log_transform(threshold_image)
    else:
        conversion = None
    threshold_mask = None if mask is None else np.asarray(mask, dtype=bool)
    values = threshold_image[threshold_mask] if mask is not None else threshold_image.ravel()
    values = values[np.isfinite(values)]

    if values.size == 0:
        threshold = 0.0
    elif np.all(values == values.ravel()[0]):
        threshold = float(values.ravel()[0])
    else:
        threshold = GlobalThresholdMethodStrategy.for_method(method).compute(
            GlobalThresholdRequest(
                primitives=primitives,
                threshold_image=threshold_image,
                threshold_mask=threshold_mask,
                values=values,
                assignment=assignment,
                log_transform=log_transform,
                kwargs=dict(kwargs),
            )
        )

    if conversion is not None:
        threshold = float(primitives.inverse_log_transform(threshold, conversion))
    threshold *= threshold_correction_factor
    return clip_threshold(threshold, threshold_min, threshold_max)


def cellprofiler_get_adaptive_threshold(
    image: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    threshold_method: CellProfilerThresholdMethod | str = CellProfilerThresholdMethod.OTSU,
    window_size: int = 50,
    threshold_min: float = 0,
    threshold_max: float = 1,
    threshold_correction_factor: float = 1,
    assign_middle_to_foreground: (
        CellProfilerThresholdAssignment | str
    ) = CellProfilerThresholdAssignment.FOREGROUND,
    global_limits: tuple[float, float] = (0.7, 1.5),
    log_transform: bool = False,
    global_threshold_function: object | None = None,
    **kwargs: object,
) -> np.ndarray:
    """Compute CP-style adaptive thresholds without depending on CP packages."""
    primitives = threshold_primitives()
    global_threshold = (
        cellprofiler_get_global_threshold
        if global_threshold_function is None
        else global_threshold_function
    )
    method = coerce_cellprofiler_enum(CellProfilerThresholdMethod, threshold_method)
    assignment = coerce_cellprofiler_enum(
        CellProfilerThresholdAssignment,
        assign_middle_to_foreground,
    )
    data = np.asarray(image, dtype=np.float32)
    if mask is not None:
        data = np.where(np.asarray(mask, dtype=bool), data, False)

    if log_transform:
        transformed, conversion = primitives.log_transform(data)
    else:
        transformed = data
        conversion = None

    if transformed.size == 0 or np.all(np.isnan(transformed)):
        thresholds = np.zeros_like(transformed)
    elif np.all(transformed == transformed.ravel()[0]):
        thresholds = np.full_like(transformed, transformed.ravel()[0])
    elif method is CellProfilerThresholdMethod.SAUVOLA:
        if window_size % 2 == 0:
            window_size += 1
        thresholds = primitives.sauvola_threshold_image(
            transformed,
            window_size=window_size,
        )
    else:
        thresholds = adaptive_threshold_blocks(
            transformed,
            window_size=window_size,
            threshold_method=method,
            assign_middle_to_foreground=assignment,
            global_threshold_function=global_threshold,
            **kwargs,
        )

    global_value = global_threshold(
        transformed,
        mask=None,
        threshold_method=method,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        threshold_correction_factor=threshold_correction_factor,
        assign_middle_to_foreground=assignment,
        log_transform=False,
        **kwargs,
    )
    if conversion is not None:
        thresholds = primitives.inverse_log_transform(thresholds, conversion)
        global_value = float(primitives.inverse_log_transform(global_value, conversion))

    thresholds = thresholds * threshold_correction_factor
    t_min = max(threshold_min, global_value * global_limits[0])
    t_max = min(threshold_max, global_value * global_limits[1])
    thresholds[thresholds < t_min] = t_min
    thresholds[thresholds > t_max] = t_max
    return thresholds


def threshold_multiotsu(values: np.ndarray, *, nbins: int) -> np.ndarray:
    """Compute CP-compatible multi-Otsu thresholds for the observed value range."""
    if values.size == 0:
        return np.zeros((2,), dtype=float)
    return threshold_primitives().multiotsu_thresholds(values, nbins=nbins)


def threshold_histogram_bin_width(values: np.ndarray, nbins: int) -> float:
    values_array = np.asarray(values, dtype=np.float64)
    finite_values = values_array[np.isfinite(values_array)]
    if finite_values.size == 0 or nbins <= 0:
        return 0.0
    value_min = float(np.min(finite_values))
    value_max = float(np.max(finite_values))
    if value_max == value_min:
        return 0.0
    return (value_max - value_min) / float(nbins)


def get_threshold_robust_background(
    image: np.ndarray,
    *,
    lower_outlier_fraction: float = 0.05,
    upper_outlier_fraction: float = 0.05,
    averaging_method: CellProfilerAveragingMethod | str = CellProfilerAveragingMethod.MEAN,
    variance_method: (
        CellProfilerVarianceMethod | str
    ) = CellProfilerVarianceMethod.STANDARD_DEVIATION,
    number_of_deviations: float = 2,
    **_ignored: object,
) -> float:
    averaging_method = coerce_cellprofiler_enum(
        CellProfilerAveragingMethod,
        averaging_method,
    )
    variance_method = coerce_cellprofiler_enum(
        CellProfilerVarianceMethod,
        variance_method,
    )
    flat = np.asarray(image).flatten()
    if flat.size < 3:
        return 0.0
    flat.sort()
    if flat[0] == flat[-1]:
        return float(flat[0])
    low_chop = int(round(flat.size * lower_outlier_fraction))
    high_chop = flat.size - int(round(flat.size * upper_outlier_fraction))
    trimmed = flat if low_chop == 0 else flat[low_chop:high_chop]
    center = RobustBackgroundCenterStrategy.for_averaging_method(
        averaging_method,
    ).center(trimmed)
    spread = RobustBackgroundSpreadStrategy.for_variance_method(
        variance_method,
    ).spread(trimmed)
    return float(center + spread * number_of_deviations)


def adaptive_threshold_blocks(
    image: np.ndarray,
    *,
    window_size: int,
    threshold_method: CellProfilerThresholdMethod,
    assign_middle_to_foreground: CellProfilerThresholdAssignment,
    global_threshold_function: object | None = None,
    **kwargs: object,
) -> np.ndarray:
    image_size = np.array(image.shape[:2], dtype=int)
    nblocks = image_size // window_size
    if any(count < 2 for count in nblocks):
        raise ValueError(
            "Adaptive window cannot exceed 50% of an image dimension.\n"
            f"Window of {window_size}px is too large for a "
            f"{image_size[1]}x{image_size[0]} image"
        )

    increment = np.array(image_size, dtype=float) / np.array(nblocks, dtype=float)
    block_threshold = np.zeros([nblocks[0], nblocks[1]])
    for row in range(nblocks[0]):
        row_start = int(row * increment[0])
        row_stop = int((row + 1) * increment[0])
        for column in range(nblocks[1]):
            column_start = int(column * increment[1])
            column_stop = int((column + 1) * increment[1])
            block = image[row_start:row_stop, column_start:column_stop]
            block = block[~np.logical_not(block)]
            block_threshold[row, column] = block_threshold_value(
                block,
                threshold_method=threshold_method,
                assign_middle_to_foreground=assign_middle_to_foreground,
                global_threshold_function=global_threshold_function,
                **kwargs,
            )

    spline_order = min(3, int(np.min(nblocks)) - 1)
    row_start = int(increment[0] / 2)
    row_end = int((nblocks[0] - 0.5) * increment[0])
    column_start = int(increment[1] / 2)
    column_end = int((nblocks[1] - 0.5) * increment[1])
    interpolation = scipy.interpolate.RectBivariateSpline(
        np.linspace(row_start, row_end, nblocks[0]),
        np.linspace(column_start, column_end, nblocks[1]),
        block_threshold,
        bbox=(0.5, image.shape[0] - 0.5, 0.5, image.shape[1] - 0.5),
        kx=spline_order,
        ky=spline_order,
    )
    return interpolation(
        np.linspace(0.5, int(nblocks[0] * increment[0]) - 0.5, image.shape[0]),
        np.linspace(0.5, int(nblocks[1] * increment[1]) - 0.5, image.shape[1]),
    )


def block_threshold_value(
    block: np.ndarray,
    *,
    threshold_method: CellProfilerThresholdMethod,
    assign_middle_to_foreground: CellProfilerThresholdAssignment,
    global_threshold_function: object | None = None,
    **kwargs: object,
) -> float:
    global_threshold = (
        cellprofiler_get_global_threshold
        if global_threshold_function is None
        else global_threshold_function
    )
    if block.size == 0:
        return 0.0
    if np.all(block == block[0]):
        return float(block[0])
    if (
        threshold_method is CellProfilerThresholdMethod.MULTI_OTSU
        and np.unique(block).size < 3
    ):
        return threshold_primitives().otsu_threshold(block)
    return global_threshold(
        block,
        threshold_method=threshold_method,
        assign_middle_to_foreground=assign_middle_to_foreground,
        threshold_min=0,
        threshold_max=1,
        threshold_correction_factor=1,
        log_transform=False,
        **kwargs,
    )


def threshold_method_for_class_count(
    threshold_method: CellProfilerThresholdMethod,
    otsu_class_count: CellProfilerOtsuMethod,
) -> CellProfilerThresholdMethod:
    if (
        threshold_method is CellProfilerThresholdMethod.OTSU
        and otsu_class_count is CellProfilerOtsuMethod.THREE_CLASS
    ):
        return CellProfilerThresholdMethod.MULTI_OTSU
    return threshold_method


def threshold_method_kwargs(
    threshold_method: CellProfilerThresholdMethod,
    *,
    lower_outlier_fraction: float,
    upper_outlier_fraction: float,
    averaging_method: CellProfilerAveragingMethod,
    variance_method: CellProfilerVarianceMethod,
    number_of_deviations: float,
) -> dict[str, object]:
    """Return kwargs that are meaningful for the selected threshold algorithm."""
    if threshold_method is not CellProfilerThresholdMethod.ROBUST_BACKGROUND:
        return {}
    return RobustBackgroundThresholdSettings(
        lower_outlier_fraction=lower_outlier_fraction,
        upper_outlier_fraction=upper_outlier_fraction,
        averaging_method=averaging_method,
        variance_method=variance_method,
        number_of_deviations=number_of_deviations,
    ).as_kwargs()


def clip_threshold(threshold: float, threshold_min: float, threshold_max: float) -> float:
    return float(min(max(float(threshold), threshold_min), threshold_max))


def cellprofiler_threshold(
    image: np.ndarray,
    *,
    use_advanced_settings: bool,
    threshold_scope: CellProfilerThresholdScope,
    threshold_method: CellProfilerThresholdMethod,
    otsu_class_count: CellProfilerOtsuMethod,
    assign_middle_to_foreground: CellProfilerThresholdAssignment,
    log_transform: bool,
    threshold_correction_factor: float,
    threshold_min: float,
    threshold_max: float,
    threshold_smoothing_scale: float,
    adaptive_window_size: int,
    lower_outlier_fraction: float,
    upper_outlier_fraction: float,
    averaging_method: CellProfilerAveragingMethod,
    variance_method: CellProfilerVarianceMethod,
    number_of_deviations: float,
    manual_threshold: float,
    mask: np.ndarray | None = None,
    smooth_threshold_application: bool = True,
    global_threshold_function: object | None = None,
    adaptive_threshold_function: object | None = None,
    apply_threshold_function: object | None = None,
    log_profile_function: object | None = None,
) -> tuple[np.ndarray, float, float]:
    """Apply CellProfiler threshold semantics without a CP workspace."""
    global_threshold = (
        cellprofiler_get_global_threshold
        if global_threshold_function is None
        else global_threshold_function
    )
    adaptive_threshold = (
        cellprofiler_get_adaptive_threshold
        if adaptive_threshold_function is None
        else adaptive_threshold_function
    )
    profiler = CellProfilerThresholdProfiler(
        (lambda *args, **kwargs: None)
        if log_profile_function is None
        else log_profile_function
    )

    total_started_at = time.perf_counter()
    phase_started_at = time.perf_counter()
    threshold_mask = None if mask is None else np.asarray(mask, dtype=bool)
    threshold_scope = coerce_cellprofiler_enum(CellProfilerThresholdScope, threshold_scope)
    threshold_method = coerce_cellprofiler_enum(CellProfilerThresholdMethod, threshold_method)
    otsu_class_count = coerce_cellprofiler_enum(CellProfilerOtsuMethod, otsu_class_count)
    assign_middle_to_foreground = coerce_cellprofiler_enum(
        CellProfilerThresholdAssignment,
        assign_middle_to_foreground,
    )
    averaging_method = coerce_cellprofiler_enum(CellProfilerAveragingMethod, averaging_method)
    variance_method = coerce_cellprofiler_enum(CellProfilerVarianceMethod, variance_method)
    profiler.record("threshold_coerce_settings", phase_started_at)

    if not use_advanced_settings:
        threshold_scope = CellProfilerThresholdScope.GLOBAL
        threshold_method = CellProfilerThresholdMethod.MINIMUM_CROSS_ENTROPY
        log_transform = False
        threshold_smoothing_scale = CELLPROFILER_BASIC_THRESHOLD_SMOOTHING_SCALE

    if threshold_method is CellProfilerThresholdMethod.MEASUREMENT:
        raise NotImplementedError(
            "Measurement-based thresholding requires a prior measurement source."
        )

    effective_method = threshold_method_for_class_count(threshold_method, otsu_class_count)
    threshold_image = np.asarray(image)
    if threshold_method is CellProfilerThresholdMethod.MANUAL:
        final_threshold: float | np.ndarray = float(manual_threshold)
        original_threshold = float(manual_threshold)
    else:
        phase_started_at = time.perf_counter()
        method_kwargs = threshold_method_kwargs(
            effective_method,
            lower_outlier_fraction=lower_outlier_fraction,
            upper_outlier_fraction=upper_outlier_fraction,
            averaging_method=averaging_method,
            variance_method=variance_method,
            number_of_deviations=number_of_deviations,
        )
        profiler.record_method(
            "threshold_method_kwargs",
            phase_started_at,
            effective_method,
        )
        if threshold_scope is CellProfilerThresholdScope.ADAPTIVE:
            phase_started_at = time.perf_counter()
            final_threshold = adaptive_threshold(
                threshold_image,
                mask=threshold_mask,
                threshold_method=effective_method,
                window_size=adaptive_window_size,
                threshold_min=threshold_min,
                threshold_max=threshold_max,
                threshold_correction_factor=threshold_correction_factor,
                assign_middle_to_foreground=assign_middle_to_foreground,
                log_transform=log_transform,
                global_threshold_function=global_threshold,
                **method_kwargs,
            )
            profiler.record_method(
                "threshold_adaptive_final",
                phase_started_at,
                effective_method,
            )
            phase_started_at = time.perf_counter()
            original_threshold = float(
                np.mean(
                    np.atleast_1d(
                        adaptive_threshold(
                            threshold_image,
                            mask=threshold_mask,
                            threshold_method=effective_method,
                            window_size=adaptive_window_size,
                            threshold_min=threshold_min if not use_advanced_settings else 0,
                            threshold_max=threshold_max if not use_advanced_settings else 1,
                            threshold_correction_factor=(
                                threshold_correction_factor
                                if not use_advanced_settings
                                else 1
                            ),
                            assign_middle_to_foreground=assign_middle_to_foreground,
                            log_transform=log_transform,
                            global_threshold_function=global_threshold,
                            **method_kwargs,
                        )
                    )
                )
            )
            profiler.record_method(
                "threshold_adaptive_original",
                phase_started_at,
                effective_method,
            )
        else:
            selection_image, selection_kwargs = effective_method.global_threshold_selection(
                log_transform=log_transform,
                image=image,
                threshold_image=threshold_image,
            )
            phase_started_at = time.perf_counter()
            raw_threshold = global_threshold(
                selection_image,
                mask=threshold_mask,
                threshold_method=effective_method,
                threshold_min=0,
                threshold_max=1,
                threshold_correction_factor=1,
                assign_middle_to_foreground=assign_middle_to_foreground,
                log_transform=log_transform,
                **method_kwargs,
                **selection_kwargs,
            )
            profiler.record_global_raw(
                phase_started_at,
                effective_method,
                selection_image,
            )
            phase_started_at = time.perf_counter()
            final_threshold = clip_threshold(
                raw_threshold * threshold_correction_factor,
                threshold_min,
                threshold_max,
            )
            original_threshold = (
                final_threshold
                if not use_advanced_settings
                else clip_threshold(raw_threshold, 0, 1)
            )
            profiler.record_method(
                "threshold_clip",
                phase_started_at,
                effective_method,
            )

    application_smoothing = threshold_smoothing_scale if smooth_threshold_application else 0.0
    phase_started_at = time.perf_counter()
    if apply_threshold_function is None:
        binary, _sigma = ThresholdApplicationRequest(
            image=image,
            threshold=final_threshold,
            mask=threshold_mask,
            smoothing=application_smoothing,
        ).apply()
    else:
        binary, _sigma = apply_threshold_function(
            image,
            threshold=final_threshold,
            mask=threshold_mask,
            smoothing=application_smoothing,
        )
    profiler.record_apply(phase_started_at, application_smoothing)
    phase_started_at = time.perf_counter()
    if threshold_mask is not None:
        binary = np.asarray(binary, dtype=bool) & threshold_mask
    result = (
        binary.astype(bool),
        float(np.mean(np.atleast_1d(final_threshold))),
        float(original_threshold),
    )
    profiler.record("threshold_finalize", phase_started_at)
    profiler.record("threshold_total", total_started_at)
    return result


def cellprofiler_threshold_diagnostics(
    image: np.ndarray,
    binary: np.ndarray,
    *,
    final_threshold: float,
    original_threshold: float,
    mask: np.ndarray | None = None,
    proven_unit_interval_scale: int | None = None,
    log_profile_function: object | None = None,
) -> CellProfilerThresholdDiagnostics:
    """Return CellProfiler's image-level threshold quality measurements."""
    log_profile = (lambda *args, **kwargs: None) if log_profile_function is None else log_profile_function
    total_started_at = time.perf_counter()
    phase_started_at = time.perf_counter()
    measurement_mask = None if mask is None else np.asarray(mask, dtype=bool)
    binary_image = np.asarray(binary, dtype=bool)
    if capture_enabled():
        capture_array_fixture(
            "threshold_diagnostics",
            image=np.asarray(image),
            binary=binary_image,
            mask=(
                np.ones_like(binary_image, dtype=bool)
                if measurement_mask is None
                else measurement_mask
            ),
            final_threshold=np.asarray(final_threshold, dtype=np.float64),
            original_threshold=np.asarray(original_threshold, dtype=np.float64),
        )
    log_profile(
        "threshold_diagnostics_prepare",
        time.perf_counter() - phase_started_at,
        function="cellprofiler_threshold_diagnostics",
    )
    phase_started_at = time.perf_counter()
    weighted_variance, sum_of_entropies = (
        ThresholdDiagnosticsBackendStrategy.for_memory_type().diagnostics(
            image,
            measurement_mask,
            binary_image,
            proven_unit_interval_scale=proven_unit_interval_scale,
        )
    )
    log_profile(
        "threshold_diagnostics_backend",
        time.perf_counter() - phase_started_at,
        function="cellprofiler_threshold_diagnostics",
    )
    phase_started_at = time.perf_counter()
    result = CellProfilerThresholdDiagnostics(
        final_threshold=float(final_threshold),
        original_threshold=float(original_threshold),
        weighted_variance=float(np.mean(np.atleast_1d(weighted_variance))),
        sum_of_entropies=float(np.mean(np.atleast_1d(sum_of_entropies))),
    )
    log_profile(
        "threshold_diagnostics_finalize",
        time.perf_counter() - phase_started_at,
        function="cellprofiler_threshold_diagnostics",
    )
    log_profile(
        "threshold_diagnostics_total",
        time.perf_counter() - total_started_at,
        function="cellprofiler_threshold_diagnostics",
    )
    return result


ThresholdScope = CellProfilerThresholdScope
ThresholdMethod = CellProfilerThresholdMethod
Assignment = CellProfilerThresholdAssignment
AveragingMethod = CellProfilerAveragingMethod
VarianceMethod = CellProfilerVarianceMethod


@dataclass
class ThresholdResult:
    """Threshold measurement row emitted by the CP-compatible Threshold module."""

    slice_index: int
    final_threshold: float
    original_threshold: float
    guide_threshold: float
    sigma: float
    weighted_variance: float = 0.0
    sum_of_entropies: float = 0.0


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(
    (
        "threshold_results",
        csv_materializer(
            fields=[
                "slice_index",
                "final_threshold",
                "original_threshold",
                "guide_threshold",
                "sigma",
            ],
            analysis_type="threshold",
        ),
    )
)
def threshold(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    threshold_scope: ThresholdScope | str = ThresholdScope.GLOBAL,
    threshold_method: ThresholdMethod | str = ThresholdMethod.OTSU,
    assign_middle_to_foreground: Assignment | str = Assignment.FOREGROUND,
    log_transform: bool = False,
    threshold_correction_factor: float = 1.0,
    threshold_min: float = 0.0,
    threshold_max: float = 1.0,
    window_size: int = 50,
    smoothing: float = 0.0,
    lower_outlier_fraction: float = 0.05,
    upper_outlier_fraction: float = 0.05,
    averaging_method: AveragingMethod | str = AveragingMethod.MEAN,
    variance_method: VarianceMethod | str = VarianceMethod.STANDARD_DEVIATION,
    number_of_deviations: float = 2.0,
    predefined_threshold: float | None = None,
    automatic: bool = False,
    otsu_class_count: CellProfilerOtsuMethod | str = CellProfilerOtsuMethod.TWO_CLASS,
    use_advanced_settings: bool = True,
) -> tuple[np.ndarray, ThresholdResult]:
    """Apply CP-compatible thresholding and emit the module measurement row."""
    source_payload = image
    image = np.asarray(image_payload_data(source_payload), dtype=np.float32)
    if mask is not None:
        mask = np.asarray(image_payload_data(mask))
    else:
        mask = image_payload_mask(source_payload)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)

    threshold_scope = coerce_cellprofiler_enum(ThresholdScope, threshold_scope)
    threshold_method = coerce_cellprofiler_enum(ThresholdMethod, threshold_method)
    assign_middle_to_foreground = coerce_cellprofiler_enum(
        Assignment,
        assign_middle_to_foreground,
    )
    averaging_method = coerce_cellprofiler_enum(AveragingMethod, averaging_method)
    variance_method = coerce_cellprofiler_enum(VarianceMethod, variance_method)
    otsu_class_count = coerce_cellprofiler_enum(CellProfilerOtsuMethod, otsu_class_count)

    guide_threshold = 0.0

    if automatic:
        smoothing = 1.0
        log_transform = False
        threshold_scope = ThresholdScope.GLOBAL
        threshold_method = ThresholdMethod.MINIMUM_CROSS_ENTROPY

    if threshold_method is ThresholdMethod.MANUAL and predefined_threshold is None:
        predefined_threshold = 0.0
    if predefined_threshold is not None:
        threshold_method = ThresholdMethod.MANUAL

    binary_image, final_threshold, original_threshold = cellprofiler_threshold(
        image,
        use_advanced_settings=use_advanced_settings,
        threshold_scope=threshold_scope,
        threshold_method=threshold_method,
        otsu_class_count=otsu_class_count,
        assign_middle_to_foreground=assign_middle_to_foreground,
        log_transform=log_transform,
        threshold_correction_factor=threshold_correction_factor,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        threshold_smoothing_scale=smoothing,
        adaptive_window_size=window_size,
        lower_outlier_fraction=lower_outlier_fraction,
        upper_outlier_fraction=upper_outlier_fraction,
        averaging_method=averaging_method,
        variance_method=variance_method,
        number_of_deviations=number_of_deviations,
        manual_threshold=(
            float(predefined_threshold)
            if predefined_threshold is not None
            else 0.0
        ),
        mask=mask,
    )
    diagnostics = cellprofiler_threshold_diagnostics(
        image,
        binary_image,
        final_threshold=final_threshold,
        original_threshold=original_threshold,
        mask=mask,
    )
    output_image = image_payload_with_context(
        binary_image.astype(np.float32),
        mask=mask,
        metadata=image_payload_metadata(
            source_payload
        ).without_unit_interval_intensity_scale(),
    )
    return output_image, ThresholdResult(
        slice_index=0,
        final_threshold=float(final_threshold),
        original_threshold=float(original_threshold),
        guide_threshold=guide_threshold,
        sigma=float(smoothing),
        weighted_variance=diagnostics.weighted_variance,
        sum_of_entropies=diagnostics.sum_of_entropies,
    )


def _finite_flat_float64(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).ravel()
    return np.ascontiguousarray(flat[np.isfinite(flat)], dtype=np.float64)


def _finite_flat_float32(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).ravel()
    return np.ascontiguousarray(flat[np.isfinite(flat)], dtype=np.float32)


def _weighted_otsu_threshold_numba_compatible(
    values: np.ndarray,
    bin_count: int,
) -> float:
    """Return weighted Otsu using the fastest exact sorted-rank representation."""
    values_array = np.ascontiguousarray(values, dtype=np.float64)
    unique_values, counts = np.unique(values_array, return_counts=True)
    if unique_values.size < values_array.size:
        return float(
            _counted_sorted_weighted_otsu_threshold_numba(
                np.ascontiguousarray(unique_values, dtype=np.float64),
                np.ascontiguousarray(counts, dtype=np.int64),
                int(values_array.size),
                int(bin_count),
            )
        )
    return float(_sorted_weighted_otsu_threshold_numba(values_array, int(bin_count)))


def _li_threshold_float32_numpy(values: np.ndarray) -> float:
    """Return Li threshold with NumPy float32 reduction semantics.

    CP-compatible pipelines normalize images to float32 before thresholding.
    NumPy's float32 reduction order is observable here: a small threshold drift
    is enough to move object boundaries. Keep this semantic path explicit rather
    than hiding it as a backend fallback.
    """
    image = np.asarray(values, dtype=np.float32)
    if image.size == 0:
        return 0.0
    if np.all(image == image.flat[0]):
        return float(image.flat[0])

    image = image.copy()
    image_min = np.min(image)
    image -= image_min
    tolerance = _li_tolerance_numpy(image)
    threshold_next = np.mean(image)
    threshold_current = -2.0 * tolerance

    iterations = 0
    while (
        abs(threshold_next - threshold_current) > tolerance
        and iterations < 1000
    ):
        threshold_current = threshold_next
        foreground = image > threshold_current
        mean_foreground = np.mean(image[foreground])
        mean_background = np.mean(image[~foreground])
        if mean_background == 0.0:
            break
        threshold_next = (mean_background - mean_foreground) / (
            np.log(mean_background) - np.log(mean_foreground)
        )
        iterations += 1

    return float(threshold_next + image_min)


def _li_tolerance_numpy(values: np.ndarray) -> float:
    if values.size < 2:
        return CELLPROFILER_LI_TOLERANCE
    unique_values = np.unique(np.asarray(values, dtype=np.float64).ravel())
    if unique_values.size < 2:
        return CELLPROFILER_LI_TOLERANCE
    positive_diffs = np.diff(unique_values)
    positive_diffs = positive_diffs[positive_diffs > 0]
    if positive_diffs.size == 0:
        return CELLPROFILER_LI_TOLERANCE
    return max(float(np.min(positive_diffs) / 2.0), CELLPROFILER_LI_TOLERANCE)


def _numpy_threshold_weighted_variance(
    image: np.ndarray,
    mask: np.ndarray,
    binary_image: np.ndarray,
) -> float:
    image_array = np.asarray(image)
    mask_array = np.asarray(mask, dtype=bool)
    if not np.any(mask_array):
        return 0.0
    minval = float(np.max(image_array[mask_array]) / 256)
    if minval == 0:
        return 0.0

    fg = np.log2(np.maximum(image_array[binary_image & mask_array], minval))
    bg = np.log2(np.maximum(image_array[(~binary_image) & mask_array], minval))
    nfg = fg.size
    nbg = bg.size
    if nfg == 0:
        return float(np.var(bg))
    if nbg == 0:
        return float(np.var(fg))
    return float((np.var(fg) * nfg + np.var(bg) * nbg) / (nfg + nbg))


def _numpy_threshold_sum_of_entropies(
    image: np.ndarray,
    mask: np.ndarray,
    binary_image: np.ndarray,
) -> float:
    image_array = np.asarray(image)
    mask_array = np.asarray(mask, dtype=bool).copy()
    mask_array[np.isnan(image_array)] = False
    if not np.any(mask_array):
        return 0.0

    minval = float(np.max(image_array[mask_array]) / 256)
    if minval == 0:
        return 0.0

    clamped_image = image_array.copy()
    clamped_image[clamped_image < minval] = minval
    smoothed_image = _smooth_with_deterministic_noise(clamped_image, bits=8)
    im_min = np.min(smoothed_image)
    im_max = np.max(smoothed_image)
    upper = np.log2(im_max)
    lower = np.log2(im_min)
    if upper == lower:
        return float(math.log(np.sum(mask_array), 2))

    fg = smoothed_image[binary_image & mask_array]
    bg = smoothed_image[(~binary_image) & mask_array]
    if len(fg) == 0 or len(bg) == 0:
        return 0.0

    hfg = np.histogram(np.log2(fg), 256, range=(lower, upper), weights=None)[0]
    hbg = np.histogram(np.log2(bg), 256, range=(lower, upper), weights=None)[0]
    hfg = hfg[hfg > 0]
    hbg = hbg[hbg > 0]
    if hfg.size == 0:
        hfg = np.ones((1,), int)
    if hbg.size == 0:
        hbg = np.ones((1,), int)

    hfg = hfg.astype(float) / float(np.sum(hfg))
    hbg = hbg.astype(float) / float(np.sum(hbg))
    return float(np.sum(hfg * np.log2(hfg)) + np.sum(hbg * np.log2(hbg)))


def _rectangular_mask_domain(mask: np.ndarray) -> RectangularMaskDomain | None:
    """Return the true rectangle for masks that are exactly one filled rectangle."""
    if mask.ndim != 2:
        return None
    row_indices = np.flatnonzero(np.any(mask, axis=1))
    if row_indices.size == 0:
        return None
    column_indices = np.flatnonzero(np.any(mask, axis=0))
    y0 = int(row_indices[0])
    y1 = int(row_indices[-1]) + 1
    x0 = int(column_indices[0])
    x1 = int(column_indices[-1]) + 1
    if int(np.sum(mask)) != (y1 - y0) * (x1 - x0):
        return None
    return RectangularMaskDomain(slice(y0, y1), slice(x0, x1))


@lru_cache(maxsize=8)
def _quantized_log_tables(
    scale: int,
) -> QuantizedThresholdLogTables:
    codes = np.arange(int(scale) + 1, dtype=np.float32)
    values = (codes / np.float32(scale)).astype(np.float64, copy=False)
    weighted_log_values = np.zeros_like(values)
    positive_values = values > 0.0
    weighted_log_values[positive_values] = np.log2(values[positive_values])
    entropy_values = np.clip(values, CELLPROFILER_THRESHOLD_ENTROPY_DELTA, 1.0)
    return QuantizedThresholdLogTables(
        values=values,
        weighted_log_values=weighted_log_values,
        entropy_log_values=np.log2(entropy_values),
        entropy_log_delta_values=np.log2(
            entropy_values + CELLPROFILER_THRESHOLD_ENTROPY_DELTA
        ),
    )


@njit(cache=True)
def _threshold_diagnostics_unmasked_finite_quantized_numba(
    codes: np.ndarray,
    binary_image: np.ndarray,
    noise: np.ndarray,
    values: np.ndarray,
    weighted_log_values: np.ndarray,
    entropy_log_values: np.ndarray,
    entropy_log_delta_values: np.ndarray,
) -> tuple[float, float]:
    height, width = codes.shape
    if height == 0 or width == 0:
        return 0.0, 0.0

    max_value = values[codes[0, 0]]
    for y in range(height):
        for x in range(width):
            value = values[codes[y, x]]
            if value > max_value:
                max_value = value

    weighted_variance = 0.0
    minval = max_value / 256.0
    minval_log = 0.0
    delta = CELLPROFILER_THRESHOLD_ENTROPY_DELTA
    lower = np.inf
    upper = -np.inf
    foreground_count = 0
    background_count = 0
    if minval > 0.0:
        minval_log = math.log2(minval)
        fg_count = 0
        bg_count = 0
        fg_sum = 0.0
        bg_sum = 0.0
        fg_sumsq = 0.0
        bg_sumsq = 0.0
        for y in range(height):
            for x in range(width):
                code = codes[y, x]
                value = values[code]
                if value < minval:
                    clipped = minval
                    if clipped < delta:
                        clipped = delta
                    elif clipped > 1.0:
                        clipped = 1.0
                    weighted_log_value = minval_log
                    entropy_log_value = math.log2(clipped)
                    log_delta_value = math.log2(clipped + delta)
                else:
                    weighted_log_value = weighted_log_values[code]
                    entropy_log_value = entropy_log_values[code]
                    log_delta_value = entropy_log_delta_values[code]
                noise_value = noise[y, x]
                log_smoothed_value = (
                    log_delta_value * noise_value
                    + (1.0 - noise_value) * entropy_log_value
                )
                if log_smoothed_value > 0.0:
                    log_smoothed_value = 0.0
                if log_smoothed_value < lower:
                    lower = log_smoothed_value
                if log_smoothed_value > upper:
                    upper = log_smoothed_value
                if binary_image[y, x]:
                    fg_count += 1
                    foreground_count += 1
                    fg_sum += weighted_log_value
                    fg_sumsq += weighted_log_value * weighted_log_value
                else:
                    bg_count += 1
                    background_count += 1
                    bg_sum += weighted_log_value
                    bg_sumsq += weighted_log_value * weighted_log_value

        if fg_count == 0 and bg_count == 0:
            weighted_variance = 0.0
        elif fg_count == 0:
            bg_mean = bg_sum / bg_count
            weighted_variance = bg_sumsq / bg_count - bg_mean * bg_mean
        elif bg_count == 0:
            fg_mean = fg_sum / fg_count
            weighted_variance = fg_sumsq / fg_count - fg_mean * fg_mean
        else:
            fg_mean = fg_sum / fg_count
            bg_mean = bg_sum / bg_count
            fg_variance = fg_sumsq / fg_count - fg_mean * fg_mean
            bg_variance = bg_sumsq / bg_count - bg_mean * bg_mean
            weighted_variance = (
                fg_variance * fg_count + bg_variance * bg_count
            ) / (fg_count + bg_count)

    if minval == 0.0:
        return weighted_variance, 0.0

    if upper == lower:
        return weighted_variance, math.log2(float(foreground_count + background_count))
    if foreground_count == 0 or background_count == 0:
        return weighted_variance, 0.0

    foreground_hist = np.zeros(CELLPROFILER_THRESHOLD_ENTROPY_BINS, dtype=np.int64)
    background_hist = np.zeros(CELLPROFILER_THRESHOLD_ENTROPY_BINS, dtype=np.int64)
    scale = float(CELLPROFILER_THRESHOLD_ENTROPY_BINS) / (upper - lower)
    for y in range(height):
        for x in range(width):
            code = codes[y, x]
            value = values[code]
            if value < minval:
                clipped = minval
                if clipped < delta:
                    clipped = delta
                elif clipped > 1.0:
                    clipped = 1.0
                entropy_log_value = math.log2(clipped)
                log_delta_value = math.log2(clipped + delta)
            else:
                entropy_log_value = entropy_log_values[code]
                log_delta_value = entropy_log_delta_values[code]
            noise_value = noise[y, x]
            log_smoothed_value = (
                log_delta_value * noise_value
                + (1.0 - noise_value) * entropy_log_value
            )
            if log_smoothed_value > 0.0:
                log_smoothed_value = 0.0
            bin_index = int((log_smoothed_value - lower) * scale)
            if bin_index < 0:
                continue
            if bin_index >= CELLPROFILER_THRESHOLD_ENTROPY_BINS:
                if bin_index == CELLPROFILER_THRESHOLD_ENTROPY_BINS:
                    bin_index = CELLPROFILER_THRESHOLD_ENTROPY_BINS - 1
                else:
                    continue
            if binary_image[y, x]:
                foreground_hist[bin_index] += 1
            else:
                background_hist[bin_index] += 1

    return weighted_variance, _histogram_entropy_numba(
        foreground_hist,
        foreground_count,
    ) + _histogram_entropy_numba(
        background_hist,
        background_count,
    )


@njit(cache=True)
def _threshold_diagnostics_rectangular_mask_quantized_numba(
    codes: np.ndarray,
    binary_image: np.ndarray,
    noise: np.ndarray,
    values: np.ndarray,
    weighted_log_values: np.ndarray,
    entropy_log_values: np.ndarray,
    entropy_log_delta_values: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> tuple[float, float]:
    height, width = codes.shape
    if height == 0 or width == 0 or y0 >= y1 or x0 >= x1:
        return 0.0, 0.0

    masked_max = values[codes[y0, x0]]
    for y in range(y0, y1):
        for x in range(x0, x1):
            value = values[codes[y, x]]
            if value > masked_max:
                masked_max = value

    weighted_variance = 0.0
    minval = masked_max / 256.0
    if minval > 0.0:
        minval_log = math.log2(minval)
        fg_count = 0
        bg_count = 0
        fg_sum = 0.0
        bg_sum = 0.0
        fg_sumsq = 0.0
        bg_sumsq = 0.0
        for y in range(y0, y1):
            for x in range(x0, x1):
                code = codes[y, x]
                value = values[code]
                log_value = minval_log if value < minval else weighted_log_values[code]
                if binary_image[y, x]:
                    fg_count += 1
                    fg_sum += log_value
                    fg_sumsq += log_value * log_value
                else:
                    bg_count += 1
                    bg_sum += log_value
                    bg_sumsq += log_value * log_value

        if fg_count == 0 and bg_count == 0:
            weighted_variance = 0.0
        elif fg_count == 0:
            bg_mean = bg_sum / bg_count
            weighted_variance = bg_sumsq / bg_count - bg_mean * bg_mean
        elif bg_count == 0:
            fg_mean = fg_sum / fg_count
            weighted_variance = fg_sumsq / fg_count - fg_mean * fg_mean
        else:
            fg_mean = fg_sum / fg_count
            bg_mean = bg_sum / bg_count
            fg_variance = fg_sumsq / fg_count - fg_mean * fg_mean
            bg_variance = bg_sumsq / bg_count - bg_mean * bg_mean
            weighted_variance = (
                fg_variance * fg_count + bg_variance * bg_count
            ) / (fg_count + bg_count)

    if minval == 0.0:
        return weighted_variance, 0.0

    delta = CELLPROFILER_THRESHOLD_ENTROPY_DELTA
    lower = np.inf
    upper = -np.inf
    for y in range(height):
        for x in range(width):
            code = codes[y, x]
            value = values[code]
            if value < minval:
                clipped = minval
                if clipped < delta:
                    clipped = delta
                elif clipped > 1.0:
                    clipped = 1.0
                log_value = math.log2(clipped)
                log_delta_value = math.log2(clipped + delta)
            else:
                log_value = entropy_log_values[code]
                log_delta_value = entropy_log_delta_values[code]
            noise_value = noise[y, x]
            log_smoothed_value = (
                log_delta_value * noise_value
                + (1.0 - noise_value) * log_value
            )
            if log_smoothed_value > 0.0:
                log_smoothed_value = 0.0
            if log_smoothed_value < lower:
                lower = log_smoothed_value
            if log_smoothed_value > upper:
                upper = log_smoothed_value

    foreground_count = 0
    background_count = 0
    for y in range(y0, y1):
        for x in range(x0, x1):
            if binary_image[y, x]:
                foreground_count += 1
            else:
                background_count += 1

    if upper == lower:
        return weighted_variance, math.log2(float(foreground_count + background_count))
    if foreground_count == 0 or background_count == 0:
        return weighted_variance, 0.0

    foreground_hist = np.zeros(CELLPROFILER_THRESHOLD_ENTROPY_BINS, dtype=np.int64)
    background_hist = np.zeros(CELLPROFILER_THRESHOLD_ENTROPY_BINS, dtype=np.int64)
    histogram_scale = float(CELLPROFILER_THRESHOLD_ENTROPY_BINS) / (upper - lower)
    for y in range(y0, y1):
        for x in range(x0, x1):
            code = codes[y, x]
            value = values[code]
            if value < minval:
                clipped = minval
                if clipped < delta:
                    clipped = delta
                elif clipped > 1.0:
                    clipped = 1.0
                log_value = math.log2(clipped)
                log_delta_value = math.log2(clipped + delta)
            else:
                log_value = entropy_log_values[code]
                log_delta_value = entropy_log_delta_values[code]
            noise_value = noise[y, x]
            log_smoothed_value = (
                log_delta_value * noise_value
                + (1.0 - noise_value) * log_value
            )
            if log_smoothed_value > 0.0:
                log_smoothed_value = 0.0
            bin_index = int((log_smoothed_value - lower) * histogram_scale)
            if bin_index < 0:
                continue
            if bin_index >= CELLPROFILER_THRESHOLD_ENTROPY_BINS:
                if bin_index == CELLPROFILER_THRESHOLD_ENTROPY_BINS:
                    bin_index = CELLPROFILER_THRESHOLD_ENTROPY_BINS - 1
                else:
                    continue
            if binary_image[y, x]:
                foreground_hist[bin_index] += 1
            else:
                background_hist[bin_index] += 1

    return weighted_variance, _histogram_entropy_numba(
        foreground_hist,
        foreground_count,
    ) + _histogram_entropy_numba(
        background_hist,
        background_count,
    )


@njit(cache=True)
def _threshold_diagnostics_unmasked_finite_numba(
    image: np.ndarray,
    binary_image: np.ndarray,
    noise: np.ndarray,
) -> tuple[float, float]:
    height, width = image.shape
    if height == 0 or width == 0:
        return 0.0, 0.0

    max_value = image[0, 0]
    for y in range(height):
        for x in range(width):
            value = image[y, x]
            if value > max_value:
                max_value = value

    weighted_variance = 0.0
    minval = max_value / 256.0
    if minval != 0.0:
        fg_count = 0
        bg_count = 0
        fg_sum = 0.0
        bg_sum = 0.0
        fg_sumsq = 0.0
        bg_sumsq = 0.0
        for y in range(height):
            for x in range(width):
                value = image[y, x]
                if value < minval:
                    value = minval
                log_value = math.log2(value)
                if binary_image[y, x]:
                    fg_count += 1
                    fg_sum += log_value
                    fg_sumsq += log_value * log_value
                else:
                    bg_count += 1
                    bg_sum += log_value
                    bg_sumsq += log_value * log_value

        if fg_count == 0 and bg_count == 0:
            weighted_variance = 0.0
        elif fg_count == 0:
            bg_mean = bg_sum / bg_count
            weighted_variance = bg_sumsq / bg_count - bg_mean * bg_mean
        elif bg_count == 0:
            fg_mean = fg_sum / fg_count
            weighted_variance = fg_sumsq / fg_count - fg_mean * fg_mean
        else:
            fg_mean = fg_sum / fg_count
            bg_mean = bg_sum / bg_count
            fg_variance = fg_sumsq / fg_count - fg_mean * fg_mean
            bg_variance = bg_sumsq / bg_count - bg_mean * bg_mean
            weighted_variance = (
                fg_variance * fg_count + bg_variance * bg_count
            ) / (fg_count + bg_count)

    if minval == 0.0:
        return weighted_variance, 0.0

    delta = CELLPROFILER_THRESHOLD_ENTROPY_DELTA
    lower = np.inf
    upper = -np.inf
    foreground_count = 0
    background_count = 0
    log_smoothed = np.empty((height, width), dtype=np.float64)
    for y in range(height):
        for x in range(width):
            value = image[y, x]
            if value < minval:
                value = minval
            if value < delta:
                clipped = delta
            elif value > 1.0:
                clipped = 1.0
            else:
                clipped = value

            noise_value = noise[y, x]
            log_smoothed_value = (
                math.log2(clipped + delta) * noise_value
                + (1.0 - noise_value) * math.log2(clipped)
            )
            if log_smoothed_value > 0.0:
                log_smoothed_value = 0.0
            log_smoothed[y, x] = log_smoothed_value
            if log_smoothed_value < lower:
                lower = log_smoothed_value
            if log_smoothed_value > upper:
                upper = log_smoothed_value
            if binary_image[y, x]:
                foreground_count += 1
            else:
                background_count += 1

    if upper == lower:
        return weighted_variance, math.log2(float(foreground_count + background_count))
    if foreground_count == 0 or background_count == 0:
        return weighted_variance, 0.0

    foreground_hist = np.zeros(CELLPROFILER_THRESHOLD_ENTROPY_BINS, dtype=np.int64)
    background_hist = np.zeros(CELLPROFILER_THRESHOLD_ENTROPY_BINS, dtype=np.int64)
    scale = float(CELLPROFILER_THRESHOLD_ENTROPY_BINS) / (upper - lower)
    for y in range(height):
        for x in range(width):
            bin_index = int((log_smoothed[y, x] - lower) * scale)
            if bin_index < 0:
                continue
            if bin_index >= CELLPROFILER_THRESHOLD_ENTROPY_BINS:
                if bin_index == CELLPROFILER_THRESHOLD_ENTROPY_BINS:
                    bin_index = CELLPROFILER_THRESHOLD_ENTROPY_BINS - 1
                else:
                    continue
            if binary_image[y, x]:
                foreground_hist[bin_index] += 1
            else:
                background_hist[bin_index] += 1

    return weighted_variance, _histogram_entropy_numba(
        foreground_hist,
        foreground_count,
    ) + _histogram_entropy_numba(
        background_hist,
        background_count,
    )


@njit(cache=True)
def _threshold_diagnostics_numba(
    image: np.ndarray,
    mask: np.ndarray,
    binary_image: np.ndarray,
    noise: np.ndarray,
) -> tuple[float, float]:
    height, width = image.shape
    any_weighted_masked = False
    any_entropy_masked = False
    weighted_max_value = -np.inf
    entropy_max_value = -np.inf
    for y in range(height):
        for x in range(width):
            if not mask[y, x]:
                continue
            value = image[y, x]
            any_weighted_masked = True
            if value > weighted_max_value:
                weighted_max_value = value
            if not np.isnan(value):
                any_entropy_masked = True
                if value > entropy_max_value:
                    entropy_max_value = value

    weighted_variance = 0.0
    weighted_minval = weighted_max_value / 256.0
    if any_weighted_masked and weighted_minval != 0.0:
        fg_count = 0
        bg_count = 0
        fg_sum = 0.0
        bg_sum = 0.0
        fg_sumsq = 0.0
        bg_sumsq = 0.0
        for y in range(height):
            for x in range(width):
                if not mask[y, x]:
                    continue
                value = image[y, x]
                if value < weighted_minval:
                    value = weighted_minval
                log_value = math.log2(value)
                if binary_image[y, x]:
                    fg_count += 1
                    fg_sum += log_value
                    fg_sumsq += log_value * log_value
                else:
                    bg_count += 1
                    bg_sum += log_value
                    bg_sumsq += log_value * log_value

        if fg_count == 0 and bg_count == 0:
            weighted_variance = 0.0
        elif fg_count == 0:
            bg_mean = bg_sum / bg_count
            weighted_variance = bg_sumsq / bg_count - bg_mean * bg_mean
        elif bg_count == 0:
            fg_mean = fg_sum / fg_count
            weighted_variance = fg_sumsq / fg_count - fg_mean * fg_mean
        else:
            fg_mean = fg_sum / fg_count
            bg_mean = bg_sum / bg_count
            fg_variance = fg_sumsq / fg_count - fg_mean * fg_mean
            bg_variance = bg_sumsq / bg_count - bg_mean * bg_mean
            weighted_variance = (
                fg_variance * fg_count + bg_variance * bg_count
            ) / (fg_count + bg_count)

    if not any_entropy_masked:
        return weighted_variance, 0.0
    entropy_minval = entropy_max_value / 256.0
    if entropy_minval == 0.0:
        return weighted_variance, 0.0

    delta = CELLPROFILER_THRESHOLD_ENTROPY_DELTA
    im_min = np.inf
    im_max = -np.inf
    foreground_count = 0
    background_count = 0
    smoothed = np.empty((height, width), dtype=np.float64)
    for y in range(height):
        for x in range(width):
            value = image[y, x]
            if value < entropy_minval:
                value = entropy_minval
            if value < delta:
                clipped = delta
            elif value > 1.0:
                clipped = 1.0
            else:
                clipped = value

            noise_value = noise[y, x]
            smoothed_value = 2.0 ** (
                math.log2(clipped + delta) * noise_value
                + (1.0 - noise_value) * math.log2(clipped)
            )
            if smoothed_value > 1.0:
                smoothed_value = 1.0
            elif smoothed_value < 0.0:
                smoothed_value = 0.0
            smoothed[y, x] = smoothed_value
            if smoothed_value < im_min:
                im_min = smoothed_value
            if smoothed_value > im_max:
                im_max = smoothed_value

            if mask[y, x] and not np.isnan(image[y, x]):
                if binary_image[y, x]:
                    foreground_count += 1
                else:
                    background_count += 1

    upper = math.log2(im_max)
    lower = math.log2(im_min)
    if upper == lower:
        return weighted_variance, math.log2(float(foreground_count + background_count))
    if foreground_count == 0 or background_count == 0:
        return weighted_variance, 0.0

    foreground_hist = np.zeros(CELLPROFILER_THRESHOLD_ENTROPY_BINS, dtype=np.int64)
    background_hist = np.zeros(CELLPROFILER_THRESHOLD_ENTROPY_BINS, dtype=np.int64)
    scale = float(CELLPROFILER_THRESHOLD_ENTROPY_BINS) / (upper - lower)
    for y in range(height):
        for x in range(width):
            if (not mask[y, x]) or np.isnan(image[y, x]):
                continue
            log_value = math.log2(smoothed[y, x])
            bin_index = int((log_value - lower) * scale)
            if bin_index < 0:
                continue
            if bin_index >= CELLPROFILER_THRESHOLD_ENTROPY_BINS:
                if bin_index == CELLPROFILER_THRESHOLD_ENTROPY_BINS:
                    bin_index = CELLPROFILER_THRESHOLD_ENTROPY_BINS - 1
                else:
                    continue
            if binary_image[y, x]:
                foreground_hist[bin_index] += 1
            else:
                background_hist[bin_index] += 1

    return weighted_variance, _histogram_entropy_numba(
        foreground_hist,
        foreground_count,
    ) + _histogram_entropy_numba(
        background_hist,
        background_count,
    )


@njit(cache=True)
def _threshold_weighted_variance_numba(
    image: np.ndarray,
    mask: np.ndarray,
    binary_image: np.ndarray,
) -> float:
    height, width = image.shape
    any_masked = False
    max_value = -np.inf
    for y in range(height):
        for x in range(width):
            if not mask[y, x]:
                continue
            any_masked = True
            value = image[y, x]
            if value > max_value:
                max_value = value

    if not any_masked:
        return 0.0
    minval = max_value / 256.0
    if minval == 0.0:
        return 0.0

    fg_count = 0
    bg_count = 0
    fg_sum = 0.0
    bg_sum = 0.0
    fg_sumsq = 0.0
    bg_sumsq = 0.0
    for y in range(height):
        for x in range(width):
            if not mask[y, x]:
                continue
            value = image[y, x]
            if value < minval:
                value = minval
            log_value = math.log2(value)
            if binary_image[y, x]:
                fg_count += 1
                fg_sum += log_value
                fg_sumsq += log_value * log_value
            else:
                bg_count += 1
                bg_sum += log_value
                bg_sumsq += log_value * log_value

    if fg_count == 0 and bg_count == 0:
        return 0.0
    if fg_count == 0:
        bg_mean = bg_sum / bg_count
        return bg_sumsq / bg_count - bg_mean * bg_mean
    if bg_count == 0:
        fg_mean = fg_sum / fg_count
        return fg_sumsq / fg_count - fg_mean * fg_mean

    fg_mean = fg_sum / fg_count
    bg_mean = bg_sum / bg_count
    fg_variance = fg_sumsq / fg_count - fg_mean * fg_mean
    bg_variance = bg_sumsq / bg_count - bg_mean * bg_mean
    return (
        fg_variance * fg_count + bg_variance * bg_count
    ) / (fg_count + bg_count)


@njit(cache=True)
def _threshold_sum_of_entropies_numba(
    image: np.ndarray,
    mask: np.ndarray,
    binary_image: np.ndarray,
    noise: np.ndarray,
) -> float:
    height, width = image.shape
    any_masked = False
    max_value = -np.inf
    for y in range(height):
        for x in range(width):
            if (not mask[y, x]) or np.isnan(image[y, x]):
                continue
            any_masked = True
            value = image[y, x]
            if value > max_value:
                max_value = value

    if not any_masked:
        return 0.0
    minval = max_value / 256.0
    if minval == 0.0:
        return 0.0

    delta = CELLPROFILER_THRESHOLD_ENTROPY_DELTA
    im_min = np.inf
    im_max = -np.inf
    foreground_count = 0
    background_count = 0
    smoothed = np.empty((height, width), dtype=np.float64)
    for y in range(height):
        for x in range(width):
            value = image[y, x]
            if value < minval:
                value = minval
            if value < delta:
                clipped = delta
            elif value > 1.0:
                clipped = 1.0
            else:
                clipped = value

            noise_value = noise[y, x]
            smoothed_value = 2.0 ** (
                math.log2(clipped + delta) * noise_value
                + (1.0 - noise_value) * math.log2(clipped)
            )
            if smoothed_value > 1.0:
                smoothed_value = 1.0
            elif smoothed_value < 0.0:
                smoothed_value = 0.0
            smoothed[y, x] = smoothed_value
            if smoothed_value < im_min:
                im_min = smoothed_value
            if smoothed_value > im_max:
                im_max = smoothed_value

            if mask[y, x] and not np.isnan(image[y, x]):
                if binary_image[y, x]:
                    foreground_count += 1
                else:
                    background_count += 1

    upper = math.log2(im_max)
    lower = math.log2(im_min)
    if upper == lower:
        return math.log2(float(foreground_count + background_count))
    if foreground_count == 0 or background_count == 0:
        return 0.0

    foreground_hist = np.zeros(CELLPROFILER_THRESHOLD_ENTROPY_BINS, dtype=np.int64)
    background_hist = np.zeros(CELLPROFILER_THRESHOLD_ENTROPY_BINS, dtype=np.int64)
    scale = float(CELLPROFILER_THRESHOLD_ENTROPY_BINS) / (upper - lower)
    for y in range(height):
        for x in range(width):
            if (not mask[y, x]) or np.isnan(image[y, x]):
                continue
            log_value = math.log2(smoothed[y, x])
            bin_index = int((log_value - lower) * scale)
            if bin_index < 0:
                continue
            if bin_index >= CELLPROFILER_THRESHOLD_ENTROPY_BINS:
                if bin_index == CELLPROFILER_THRESHOLD_ENTROPY_BINS:
                    bin_index = CELLPROFILER_THRESHOLD_ENTROPY_BINS - 1
                else:
                    continue
            if binary_image[y, x]:
                foreground_hist[bin_index] += 1
            else:
                background_hist[bin_index] += 1

    return _histogram_entropy_numba(
        foreground_hist,
        foreground_count,
    ) + _histogram_entropy_numba(
        background_hist,
        background_count,
    )


@njit(cache=True)
def _histogram_entropy_numba(histogram: np.ndarray, total_count: int) -> float:
    if total_count <= 0:
        return 0.0
    entropy = 0.0
    total = float(total_count)
    for index in range(histogram.size):
        count = histogram[index]
        if count <= 0:
            continue
        probability = float(count) / total
        entropy += probability * math.log2(probability)
    return entropy


def _smooth_with_deterministic_noise(image: np.ndarray, *, bits: int) -> np.ndarray:
    delta = pow(2.0, -bits)
    image_copy = np.clip(image, delta, 1)
    noise = _deterministic_normal_noise(image_copy.shape)
    result = np.exp2(
        np.log2(image_copy + delta) * noise
        + (1 - noise) * np.log2(image_copy)
    )
    result[result > 1] = 1
    result[result < 0] = 0
    return result


@lru_cache(maxsize=16)
def _deterministic_normal_noise(shape: tuple[int, ...]) -> np.ndarray:
    random_state = np.random.RandomState()
    random_state.seed(0)
    noise = random_state.normal(size=shape)
    noise.setflags(write=False)
    return noise


@njit(cache=True)
def _log_transform_numba(
    values: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    transformed = np.zeros(values.shape, dtype=np.float64)
    if values.size == 0:
        return transformed, 0.0, 0.0, 0.0

    min_value = np.inf
    max_value = -np.inf
    for index in range(values.size):
        value = float(values[index])
        if not np.isfinite(value):
            continue
        if value < min_value:
            min_value = value
        if value > max_value:
            max_value = value

    if not np.isfinite(max_value) or max_value <= 0.0:
        return transformed, 0.0, 0.0, 0.0
    if not np.isfinite(min_value):
        min_value = 0.0
    noise_min = float(
        np.float32(
            np.float32(min_value)
            + np.float32(max_value - min_value) / np.float32(256.0)
            + np.finfo(np.float32).eps
        )
    )

    log_min = float(np.float32(math.log(noise_min)))
    log_max = float(np.float32(math.log(np.float32(max_value))))
    denominator = log_max - log_min
    if denominator == 0.0:
        return transformed, noise_min, log_min, log_max

    for index in range(values.size):
        value = float(values[index])
        if not np.isfinite(value) or value < noise_min:
            value = noise_min
        transformed[index] = (math.log(value) - log_min) / denominator
    return transformed, noise_min, log_min, log_max


@njit(cache=True)
def _inverse_log_transform_numba(
    values: np.ndarray,
    log_min: float,
    log_max: float,
) -> np.ndarray:
    output = np.empty(values.shape, dtype=np.float64)
    scale = log_max - log_min
    for index in range(values.size):
        output[index] = np.float32(math.exp(log_min + float(values[index]) * scale))
    return output


@njit(cache=True)
def _binned_mode_numba(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    minimum = values[0]
    maximum = values[0]
    for index in range(1, values.size):
        value = values[index]
        if value < minimum:
            minimum = value
        if value > maximum:
            maximum = value
    if maximum == minimum:
        return float(minimum)

    bin_count = int(math.ceil(math.sqrt(float(values.size))))
    if bin_count < 2:
        bin_count = 2
    counts = np.zeros(bin_count, dtype=np.int64)
    scale = float(bin_count) / (maximum - minimum)
    for index in range(values.size):
        bin_index = int((values[index] - minimum) * scale)
        if bin_index < 0:
            bin_index = 0
        elif bin_index >= bin_count:
            bin_index = bin_count - 1
        counts[bin_index] += 1

    best_index = 0
    best_count = counts[0]
    for index in range(1, bin_count):
        if counts[index] > best_count:
            best_index = index
            best_count = counts[index]
    return minimum + (float(best_index) + 0.5) * (maximum - minimum) / bin_count


@njit(cache=True)
def _mad_numba(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    sorted_values = np.sort(values.copy())
    median = _median_sorted_numba(sorted_values)
    deviations = np.empty(values.size, dtype=np.float64)
    for index in range(values.size):
        deviations[index] = abs(values[index] - median)
    deviations = np.sort(deviations)
    return _median_sorted_numba(deviations)


@njit(cache=True)
def _median_sorted_numba(sorted_values: np.ndarray) -> float:
    size = sorted_values.size
    if size == 0:
        return 0.0
    middle = size // 2
    if size % 2 == 1:
        return float(sorted_values[middle])
    return (float(sorted_values[middle - 1]) + float(sorted_values[middle])) / 2.0


@njit(cache=True)
def _histogram_range_numba(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    minimum = values[0]
    maximum = values[0]
    for index in range(1, values.size):
        value = values[index]
        if value < minimum:
            minimum = value
        if value > maximum:
            maximum = value
    return minimum, maximum


@njit(cache=True)
def _histogram_counts_numba(
    values: np.ndarray,
    bin_count: int,
    minimum: float,
    maximum: float,
) -> np.ndarray:
    counts = np.zeros(bin_count, dtype=np.int64)
    if values.size == 0 or maximum == minimum:
        return counts
    scale = float(bin_count) / (maximum - minimum)
    for index in range(values.size):
        bin_index = int((values[index] - minimum) * scale)
        if bin_index < 0:
            bin_index = 0
        elif bin_index >= bin_count:
            bin_index = bin_count - 1
        counts[bin_index] += 1
    return counts


@njit(cache=True)
def _otsu_threshold_numba(values: np.ndarray, bin_count: int) -> float:
    if values.size == 0:
        return 0.0
    minimum, maximum = _histogram_range_numba(values)
    if maximum == minimum:
        return float(minimum)
    counts = _histogram_counts_numba(values, bin_count, minimum, maximum)
    width = (maximum - minimum) / float(bin_count)
    total = 0
    total_weighted = 0.0
    for index in range(bin_count):
        count = counts[index]
        center = minimum + (float(index) + 0.5) * width
        total += count
        total_weighted += float(count) * center
    if total == 0:
        return 0.0

    background_count = 0
    background_weighted = 0.0
    best_index = 0
    best_variance = -1.0
    for index in range(bin_count - 1):
        count = counts[index]
        center = minimum + (float(index) + 0.5) * width
        background_count += count
        background_weighted += float(count) * center
        foreground_count = total - background_count
        if background_count <= 0 or foreground_count <= 0:
            continue
        background_mean = background_weighted / float(background_count)
        foreground_mean = (
            total_weighted - background_weighted
        ) / float(foreground_count)
        mean_delta = background_mean - foreground_mean
        variance = (
            float(background_count)
            * float(foreground_count)
            * mean_delta
            * mean_delta
        )
        if variance > best_variance:
            best_variance = variance
            best_index = index
    return minimum + (float(best_index) + 0.5) * width


@njit(cache=True)
def _sorted_weighted_otsu_threshold_numba(
    values: np.ndarray,
    bin_count: int,
) -> float:
    if values.size == 0:
        return 0.0
    sorted_values = np.sort(values.copy())
    size = sorted_values.size
    if size == 1:
        return float(sorted_values[0])
    if bin_count > size:
        bin_count = size
    step = size // bin_count
    if step < 1:
        step = 1

    variance = _running_variance_numba(sorted_values)
    reversed_values = np.empty(size, dtype=np.float64)
    for index in range(size):
        reversed_values[index] = sorted_values[size - 1 - index]
    reversed_variance = _running_variance_numba(reversed_values)

    best_score = np.inf
    best_candidate = 0
    candidate_count = 0
    for candidate_index in range(0, size - 1, step):
        high_index = candidate_index + 1
        score = (
            variance[candidate_index] * float(candidate_index)
            + reversed_variance[size - 1 - high_index]
            * float(size - high_index)
        )
        if score < best_score:
            best_score = score
            best_candidate = candidate_count
        candidate_count += 1

    if candidate_count == 0:
        return float(sorted_values[1])

    best_index = 1 + best_candidate * step
    low_candidate = best_candidate - 1
    high_candidate = best_candidate + 1
    if low_candidate < 0:
        low_candidate = 0
    if high_candidate >= candidate_count:
        high_candidate = candidate_count - 1
    low_index = 1 + low_candidate * step
    high_index = 1 + high_candidate * step
    if low_index >= size:
        low_index = size - 1
    if high_index >= size:
        high_index = size - 1
    return (float(sorted_values[low_index]) + float(sorted_values[high_index])) / 2.0


@njit(cache=True)
def _counted_sorted_weighted_otsu_threshold_numba(
    unique_values: np.ndarray,
    counts: np.ndarray,
    size: int,
    bin_count: int,
) -> float:
    if size == 0:
        return 0.0
    if size == 1:
        return float(unique_values[0])
    if bin_count > size:
        bin_count = size
    step = size // bin_count
    if step < 1:
        step = 1

    unique_count = unique_values.size
    cumulative_counts = np.empty(unique_count, dtype=np.int64)
    cumulative_sums = np.empty(unique_count, dtype=np.float64)
    cumulative_squares = np.empty(unique_count, dtype=np.float64)

    running_count = 0
    running_sum = 0.0
    running_square = 0.0
    for index in range(unique_count):
        count = counts[index]
        value = unique_values[index]
        running_count += count
        running_sum += value * float(count)
        running_square += value * value * float(count)
        cumulative_counts[index] = running_count
        cumulative_sums[index] = running_sum
        cumulative_squares[index] = running_square

    total_sum = cumulative_sums[unique_count - 1]
    total_square = cumulative_squares[unique_count - 1]

    best_score = np.inf
    best_candidate = 0
    candidate_count = 0
    for candidate_index in range(0, size - 1, step):
        high_index = candidate_index + 1
        foreground_count = candidate_index + 1
        background_count = size - high_index
        foreground_variance = _counted_prefix_variance_at_rank_numba(
            unique_values,
            cumulative_counts,
            cumulative_sums,
            cumulative_squares,
            candidate_index,
        )
        background_sum = total_sum - _counted_prefix_sum_at_rank_numba(
            unique_values,
            cumulative_counts,
            cumulative_sums,
            high_index - 1,
        )
        background_square = total_square - _counted_prefix_square_at_rank_numba(
            unique_values,
            cumulative_counts,
            cumulative_squares,
            high_index - 1,
        )
        background_variance = _sample_variance_numba(
            background_count,
            background_sum,
            background_square,
        )
        score = (
            foreground_variance * float(candidate_index)
            + background_variance * float(background_count)
        )
        if score < best_score:
            best_score = score
            best_candidate = candidate_count
        candidate_count += 1

    if candidate_count == 0:
        return _counted_value_at_rank_numba(unique_values, cumulative_counts, 1)

    low_candidate = best_candidate - 1
    high_candidate = best_candidate + 1
    if low_candidate < 0:
        low_candidate = 0
    if high_candidate >= candidate_count:
        high_candidate = candidate_count - 1
    low_index = 1 + low_candidate * step
    high_index = 1 + high_candidate * step
    if low_index >= size:
        low_index = size - 1
    if high_index >= size:
        high_index = size - 1
    return (
        _counted_value_at_rank_numba(unique_values, cumulative_counts, low_index)
        + _counted_value_at_rank_numba(unique_values, cumulative_counts, high_index)
    ) / 2.0


@njit(cache=True)
def _counted_prefix_variance_at_rank_numba(
    unique_values: np.ndarray,
    cumulative_counts: np.ndarray,
    cumulative_sums: np.ndarray,
    cumulative_squares: np.ndarray,
    rank: int,
) -> float:
    prefix_count = rank + 1
    prefix_sum = _counted_prefix_sum_at_rank_numba(
        unique_values,
        cumulative_counts,
        cumulative_sums,
        rank,
    )
    prefix_square = _counted_prefix_square_at_rank_numba(
        unique_values,
        cumulative_counts,
        cumulative_squares,
        rank,
    )
    return _sample_variance_numba(prefix_count, prefix_sum, prefix_square)


@njit(cache=True)
def _counted_prefix_sum_at_rank_numba(
    unique_values: np.ndarray,
    cumulative_counts: np.ndarray,
    cumulative_sums: np.ndarray,
    rank: int,
) -> float:
    bucket = _counted_bucket_for_rank_numba(cumulative_counts, rank)
    previous_count = 0 if bucket == 0 else cumulative_counts[bucket - 1]
    previous_sum = 0.0 if bucket == 0 else cumulative_sums[bucket - 1]
    partial_count = rank - previous_count + 1
    return previous_sum + unique_values[bucket] * float(partial_count)


@njit(cache=True)
def _counted_prefix_square_at_rank_numba(
    unique_values: np.ndarray,
    cumulative_counts: np.ndarray,
    cumulative_squares: np.ndarray,
    rank: int,
) -> float:
    bucket = _counted_bucket_for_rank_numba(cumulative_counts, rank)
    previous_count = 0 if bucket == 0 else cumulative_counts[bucket - 1]
    previous_square = 0.0 if bucket == 0 else cumulative_squares[bucket - 1]
    partial_count = rank - previous_count + 1
    value = unique_values[bucket]
    return previous_square + value * value * float(partial_count)


@njit(cache=True)
def _counted_value_at_rank_numba(
    unique_values: np.ndarray,
    cumulative_counts: np.ndarray,
    rank: int,
) -> float:
    return float(unique_values[_counted_bucket_for_rank_numba(cumulative_counts, rank)])


@njit(cache=True)
def _counted_bucket_for_rank_numba(cumulative_counts: np.ndarray, rank: int) -> int:
    low = 0
    high = cumulative_counts.size - 1
    while low < high:
        middle = (low + high) // 2
        if rank < cumulative_counts[middle]:
            high = middle
        else:
            low = middle + 1
    return low


@njit(cache=True)
def _sample_variance_numba(count: int, total: float, square_total: float) -> float:
    if count <= 1:
        return 0.0
    variance = (square_total - total * total / float(count)) / float(count - 1)
    if variance > 0.0:
        return variance
    return 0.0


@njit(cache=True)
def _running_variance_numba(values: np.ndarray) -> np.ndarray:
    size = values.size
    output = np.zeros(size, dtype=np.float64)
    if size < 2:
        return output

    running_sum = float(values[0])
    previous_mean = running_sum
    accumulator = 0.0
    for index in range(1, size):
        value = float(values[index])
        running_sum += value
        mean = running_sum / float(index + 1)
        accumulator += (value - previous_mean) * (value - mean)
        output[index] = accumulator / float(index)
        previous_mean = mean
    return output


@njit(cache=True)
def _li_tolerance_numba(values: np.ndarray) -> float:
    tolerance = CELLPROFILER_LI_TOLERANCE
    if values.size < 2:
        return tolerance
    sorted_values = np.sort(values.copy())
    min_diff = np.inf
    previous = sorted_values[0]
    for index in range(1, sorted_values.size):
        current = sorted_values[index]
        difference = current - previous
        if difference > 0.0 and difference < min_diff:
            min_diff = difference
        previous = current
    if min_diff == np.inf:
        return tolerance
    half_diff = min_diff / 2.0
    if half_diff > tolerance:
        return half_diff
    return tolerance


@njit(cache=True)
def _li_threshold_numba(values: np.ndarray, tolerance: float) -> float:
    if values.size == 0:
        return 0.0
    minimum, maximum = _histogram_range_numba(values)
    if maximum == minimum:
        return float(minimum)

    threshold = 0.0
    for index in range(values.size):
        threshold += values[index] - minimum
    threshold /= float(values.size)
    previous_threshold = -2.0 * tolerance
    tiny = np.finfo(np.float64).tiny
    iterations = 0
    while abs(threshold - previous_threshold) > tolerance and iterations < 1000:
        previous_threshold = threshold
        background_count = 0
        foreground_count = 0
        background_sum = 0.0
        foreground_sum = 0.0
        for index in range(values.size):
            value = values[index] - minimum
            if value <= previous_threshold:
                background_count += 1
                background_sum += value
            else:
                foreground_count += 1
                foreground_sum += value

        if background_count == 0 or foreground_count == 0:
            return previous_threshold + minimum

        background_mean = background_sum / float(background_count)
        foreground_mean = foreground_sum / float(foreground_count)
        if background_mean <= tiny:
            background_mean = tiny
        if foreground_mean <= tiny:
            foreground_mean = tiny
        if background_mean == foreground_mean:
            return background_mean + minimum

        threshold = (
            background_mean - foreground_mean
        ) / (
            math.log(background_mean) - math.log(foreground_mean)
        )
        iterations += 1
    return threshold + minimum


@njit(cache=True)
def _triangle_threshold_numba(values: np.ndarray, bin_count: int) -> float:
    if values.size == 0:
        return 0.0
    minimum, maximum = _histogram_range_numba(values)
    if maximum == minimum:
        return float(minimum)

    counts = _histogram_counts_numba(values, bin_count, minimum, maximum)
    first = 0
    while first < bin_count and counts[first] == 0:
        first += 1
    last = bin_count - 1
    while last >= 0 and counts[last] == 0:
        last -= 1
    if first >= last:
        return minimum + (float(first) + 0.5) * (maximum - minimum) / bin_count

    peak = first
    peak_count = counts[first]
    for index in range(first + 1, last + 1):
        count = counts[index]
        if count > peak_count:
            peak = index
            peak_count = count

    if peak - first < last - peak:
        original_first = first
        first = bin_count - 1 - last
        last = bin_count - 1 - original_first
        peak = bin_count - 1 - peak
        reversed_counts = np.empty(bin_count, dtype=np.int64)
        for index in range(bin_count):
            reversed_counts[index] = counts[bin_count - 1 - index]
        counts = reversed_counts
        is_reversed = True
    else:
        is_reversed = False

    x1 = float(first)
    y1 = float(counts[first])
    x2 = float(last)
    y2 = float(counts[last])
    dx = x2 - x1
    dy = y2 - y1
    normalizer = math.sqrt(dx * dx + dy * dy)
    if normalizer == 0.0:
        threshold_index = peak
    else:
        threshold_index = first
        max_distance = -1.0
        for index in range(first, last + 1):
            distance = abs(
                dy * float(index)
                - dx * float(counts[index])
                + x2 * y1
                - y2 * x1
            ) / normalizer
            if distance > max_distance:
                max_distance = distance
                threshold_index = index

    if is_reversed:
        threshold_index = bin_count - 1 - threshold_index
    return minimum + (float(threshold_index) + 0.5) * (maximum - minimum) / bin_count


@njit(cache=True)
def _isodata_threshold_numba(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    minimum, maximum = _histogram_range_numba(values)
    if maximum == minimum:
        return float(minimum)

    threshold = _mean_threshold_numba(values)
    for _ in range(1000):
        lower_count = 0
        upper_count = 0
        lower_sum = 0.0
        upper_sum = 0.0
        for index in range(values.size):
            value = values[index]
            if value <= threshold:
                lower_count += 1
                lower_sum += value
            else:
                upper_count += 1
                upper_sum += value
        if lower_count == 0 or upper_count == 0:
            return threshold
        next_threshold = (
            lower_sum / float(lower_count) + upper_sum / float(upper_count)
        ) / 2.0
        if next_threshold == threshold:
            return threshold
        if abs(next_threshold - threshold) <= 0.5 / 65536.0:
            return next_threshold
        threshold = next_threshold
    return threshold


@njit(cache=True)
def _mean_threshold_numba(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    total = 0.0
    for index in range(values.size):
        total += values[index]
    return total / float(values.size)


@njit(cache=True)
def _yen_threshold_numba(values: np.ndarray, bin_count: int) -> float:
    if values.size == 0:
        return 0.0
    minimum, maximum = _histogram_range_numba(values)
    if maximum == minimum:
        return float(minimum)

    counts = _histogram_counts_numba(values, bin_count, minimum, maximum)
    total_count = 0.0
    for index in range(bin_count):
        total_count += float(counts[index])
    if total_count == 0.0:
        return 0.0

    p1 = np.zeros(bin_count, dtype=np.float64)
    p1_sq = np.zeros(bin_count, dtype=np.float64)
    p2_sq = np.zeros(bin_count, dtype=np.float64)
    running_probability = 0.0
    running_square = 0.0
    for index in range(bin_count):
        probability = float(counts[index]) / total_count
        running_probability += probability
        running_square += probability * probability
        p1[index] = running_probability
        p1_sq[index] = running_square

    running_square = 0.0
    for index in range(bin_count - 1, -1, -1):
        probability = float(counts[index]) / total_count
        running_square += probability * probability
        p2_sq[index] = running_square

    best_index = 0
    best_criterion = -np.inf
    for index in range(bin_count - 1):
        foreground_probability = p1[index]
        background_probability = 1.0 - foreground_probability
        square_product = p1_sq[index] * p2_sq[index + 1]
        probability_product = foreground_probability * background_probability
        if square_product <= 0.0 or probability_product <= 0.0:
            continue
        criterion = -math.log(square_product) + 2.0 * math.log(
            probability_product
        )
        if criterion > best_criterion:
            best_criterion = criterion
            best_index = index

    return minimum + (float(best_index) + 0.5) * (maximum - minimum) / bin_count


@njit(cache=True)
def _minimum_threshold_numba(values: np.ndarray, bin_count: int) -> float:
    if values.size == 0:
        return 0.0
    minimum, maximum = _histogram_range_numba(values)
    if maximum == minimum:
        return float(minimum)

    histogram = _histogram_counts_numba(values, bin_count, minimum, maximum).astype(
        np.float64
    )
    maxima = np.empty(bin_count, dtype=np.int64)
    maxima_count = 0
    for _ in range(10000):
        maxima_count = _histogram_local_maxima_numba(histogram, maxima)
        if maxima_count <= 2:
            break
        previous = histogram.copy()
        for index in range(bin_count):
            left = previous[index - 1] if index > 0 else previous[index]
            center = previous[index]
            right = previous[index + 1] if index < bin_count - 1 else previous[index]
            histogram[index] = (left + center + right) / 3.0

    if maxima_count != 2:
        return np.nan

    first = maxima[0]
    second = maxima[1]
    if first > second:
        tmp = first
        first = second
        second = tmp
    valley_index = first
    valley_value = histogram[first]
    for index in range(first + 1, second + 1):
        if histogram[index] < valley_value:
            valley_value = histogram[index]
            valley_index = index
    return minimum + (float(valley_index) + 0.5) * (maximum - minimum) / bin_count


@njit(cache=True)
def _histogram_local_maxima_numba(
    histogram: np.ndarray,
    maxima: np.ndarray,
) -> int:
    count = 0
    if histogram.size == 0:
        return count
    if histogram.size == 1:
        maxima[count] = 0
        return 1
    if histogram[0] > histogram[1]:
        maxima[count] = 0
        count += 1
    for index in range(1, histogram.size - 1):
        if (
            histogram[index - 1] < histogram[index]
            and histogram[index + 1] < histogram[index]
        ):
            maxima[count] = index
            count += 1
    if histogram[histogram.size - 1] > histogram[histogram.size - 2]:
        maxima[count] = histogram.size - 1
        count += 1
    return count


@njit(cache=True)
def _multiotsu_three_class_thresholds_numba(
    values: np.ndarray,
    bin_count: int,
) -> np.ndarray:
    thresholds = np.zeros(2, dtype=np.float64)
    if values.size == 0:
        return thresholds
    if bin_count < 3:
        bin_count = 3
    minimum, maximum = _histogram_range_numba(values)
    if maximum == minimum:
        thresholds[0] = minimum
        thresholds[1] = minimum
        return thresholds

    counts = _histogram_counts_numba(values, bin_count, minimum, maximum)
    nonzero_count = 0
    last_nonzero = 0
    for index in range(bin_count):
        if counts[index] > 0:
            if nonzero_count < 2:
                thresholds[nonzero_count] = (
                    minimum
                    + (float(index) + 0.5) * (maximum - minimum) / float(bin_count)
                )
            last_nonzero = index
            nonzero_count += 1
    if nonzero_count < 3:
        if nonzero_count == 2:
            thresholds[1] = (
                minimum
                + (float(last_nonzero) + 0.5)
                * (maximum - minimum)
                / float(bin_count)
            )
        return thresholds
    if nonzero_count == 3:
        return thresholds

    width = (maximum - minimum) / float(bin_count)
    total_count = 0
    for index in range(bin_count):
        total_count += counts[index]
    if total_count == 0:
        return thresholds

    cumulative_probability = np.zeros(bin_count, dtype=np.float32)
    cumulative_weighted_index = np.zeros(bin_count, dtype=np.float32)
    running_probability = np.float32(float(counts[0]) / float(total_count))
    running_weighted_index = running_probability
    cumulative_probability[0] = running_probability
    cumulative_weighted_index[0] = running_weighted_index
    for index in range(1, bin_count):
        probability = np.float32(float(counts[index]) / float(total_count))
        running_probability = np.float32(running_probability + probability)
        running_weighted_index = np.float32(
            running_weighted_index + np.float32(float(index)) * probability
        )
        cumulative_probability[index] = running_probability
        cumulative_weighted_index[index] = running_weighted_index

    best_first = 0
    best_second = 1
    best_variance = np.float32(0.0)
    for first in range(bin_count - 2):
        for second in range(first + 1, bin_count - 1):
            variance = (
                _multiotsu_interval_score_numba(
                    cumulative_probability,
                    cumulative_weighted_index,
                    0,
                    first,
                )
                + _multiotsu_interval_score_numba(
                    cumulative_probability,
                    cumulative_weighted_index,
                    first + 1,
                    second,
                )
                + _multiotsu_interval_score_numba(
                    cumulative_probability,
                    cumulative_weighted_index,
                    second + 1,
                    bin_count - 1,
                )
            )
            if variance > best_variance:
                best_variance = variance
                best_first = first
                best_second = second
    thresholds[0] = minimum + (float(best_first) + 0.5) * width
    thresholds[1] = minimum + (float(best_second) + 0.5) * width
    return thresholds


@njit(cache=True)
def _multiotsu_interval_score_numba(
    cumulative_probability: np.ndarray,
    cumulative_weighted_index: np.ndarray,
    first: int,
    last: int,
) -> np.float32:
    if first == 0:
        probability = cumulative_probability[last]
        weighted_index = cumulative_weighted_index[last]
    else:
        probability = np.float32(
            cumulative_probability[last] - cumulative_probability[first - 1]
        )
        weighted_index = np.float32(
            cumulative_weighted_index[last]
            - cumulative_weighted_index[first - 1]
        )
    if probability <= np.float32(0.0):
        return np.float32(0.0)
    return np.float32((weighted_index * weighted_index) / probability)


@njit(cache=True)
def _minimum_cross_entropy_threshold_numba(
    values: np.ndarray,
    bin_count: int,
) -> float:
    if values.size == 0:
        return 0.0
    if bin_count < 2:
        bin_count = 2
    minimum, maximum = _histogram_range_numba(values)
    if maximum == minimum:
        return float(minimum)

    counts = _histogram_counts_numba(values, bin_count, minimum, maximum)
    width = (maximum - minimum) / float(bin_count)
    cumulative_count = np.zeros(bin_count, dtype=np.float64)
    cumulative_weighted = np.zeros(bin_count, dtype=np.float64)
    total_count = 0.0
    total_weighted = 0.0
    for index in range(bin_count):
        center = minimum + (float(index) + 0.5) * width
        count = float(counts[index])
        total_count += count
        total_weighted += count * center
        cumulative_count[index] = total_count
        cumulative_weighted[index] = total_weighted

    if total_count == 0.0:
        return 0.0

    best_index = 0
    best_cross_entropy = np.inf
    for index in range(bin_count - 1):
        foreground_count = cumulative_count[index]
        background_count = total_count - foreground_count
        foreground_weighted = cumulative_weighted[index]
        background_weighted = total_weighted - foreground_weighted
        if (
            foreground_count <= 0.0
            or background_count <= 0.0
            or foreground_weighted <= 0.0
            or background_weighted <= 0.0
        ):
            continue
        foreground_mean = foreground_weighted / foreground_count
        background_mean = background_weighted / background_count
        if foreground_mean <= 0.0 or background_mean <= 0.0:
            continue
        cross_entropy = -(
            foreground_weighted * math.log(foreground_mean)
            + background_weighted * math.log(background_mean)
        )
        if cross_entropy < best_cross_entropy:
            best_cross_entropy = cross_entropy
            best_index = index
    return minimum + (float(best_index) + 0.5) * width


@njit(cache=True)
def _sauvola_threshold_image_numba(
    image: np.ndarray,
    window_size: int,
    k: float,
    dynamic_range: float,
) -> np.ndarray:
    height, width = image.shape
    if window_size < 1:
        window_size = 1
    if window_size % 2 == 0:
        window_size += 1
    radius = window_size // 2
    padded_height = height + 2 * radius
    padded_width = width + 2 * radius
    integral = np.zeros((padded_height + 1, padded_width + 1), dtype=np.float64)
    integral_squared = np.zeros(
        (padded_height + 1, padded_width + 1),
        dtype=np.float64,
    )

    for padded_y in range(padded_height):
        image_y = _reflect_index_numba(padded_y - radius, height)
        row_sum = 0.0
        row_squared_sum = 0.0
        for padded_x in range(padded_width):
            image_x = _reflect_index_numba(padded_x - radius, width)
            value = image[image_y, image_x]
            row_sum += value
            row_squared_sum += value * value
            integral[padded_y + 1, padded_x + 1] = (
                integral[padded_y, padded_x + 1] + row_sum
            )
            integral_squared[padded_y + 1, padded_x + 1] = (
                integral_squared[padded_y, padded_x + 1] + row_squared_sum
            )

    output = np.empty((height, width), dtype=np.float64)
    area = float(window_size * window_size)

    for y in range(height):
        for x in range(width):
            y0 = y
            x0 = x
            y1 = y0 + window_size
            x1 = x0 + window_size
            total = (
                integral[y1, x1]
                - integral[y0, x1]
                - integral[y1, x0]
                + integral[y0, x0]
            )
            total_squared = (
                integral_squared[y1, x1]
                - integral_squared[y0, x1]
                - integral_squared[y1, x0]
                + integral_squared[y0, x0]
            )
            mean = total / area
            variance = total_squared / area - mean * mean
            if variance < 0.0:
                variance = 0.0
            stddev = math.sqrt(variance)
            output[y, x] = mean * (1.0 + k * ((stddev / dynamic_range) - 1.0))
    return output


@njit(cache=True)
def _reflect_index_numba(index: int, size: int) -> int:
    if size <= 1:
        return 0
    while index < 0 or index >= size:
        if index < 0:
            index = -index - 1
        elif index >= size:
            index = 2 * size - index - 1
    return index


__all__ = public_names_from_objects(
    CentrosomeNumpyThresholdPrimitiveBackendStrategy,
    NumbaLogTransformConversion,
    NumbaNumpyThresholdPrimitiveBackendStrategy,
    NumbaNumpyThresholdDiagnosticsBackendStrategy,
    NumbaNumpyThresholdSmoothingBackendStrategy,
    NumpyThresholdDiagnosticsBackendStrategy,
    "Assignment",
    "AveragingMethod",
    CellProfilerAveragingMethod,
    CellProfilerOtsuMethod,
    CellProfilerThresholdAssignment,
    CellProfilerThresholdDiagnostics,
    CellProfilerThresholdMethod,
    CellProfilerThresholdScope,
    CellProfilerVarianceMethod,
    ThresholdDiagnosticsBackendStrategy,
    "ThresholdMethod",
    ThresholdPrimitiveBackendStrategy,
    ThresholdResult,
    "ThresholdScope",
    ThresholdSmoothingBackendStrategy,
    "VarianceMethod",
    cellprofiler_threshold,
    cellprofiler_threshold_diagnostics,
    threshold,
    threshold_primitives,
)
