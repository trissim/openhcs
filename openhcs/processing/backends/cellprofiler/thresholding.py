"""Threshold diagnostic backends for CellProfiler-compatible processing."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import lru_cache
import logging
import math
import time
from typing import Annotated, Any, Callable, ClassVar, Mapping, Protocol, Self, TypedDict, Unpack
import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit
import scipy.interpolate
from openhcs.constants.constants import MemoryType
from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.callable_contract import runtime_image_execution_mode
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.runtime_profile import RuntimeProfileLogger
from openhcs.core.runtime_semantics import (
    MeasurementRowAxisField,
    MeasurementScalarLiteral,
)
from openhcs.core.runtime_values import (
    ImagePayloadMetadata,
    RuntimeImagePayloadContext,
    image_intensity_scale_for_dtype,
    image_mask_for_data_domain,
    image_payload_data,
    image_payload_metadata,
)
from openhcs.interop.cellprofiler.image_normalization import (
    normalize_cellprofiler_image_payload,
)
from openhcs.interop.cellprofiler.settings_binder import (
    coerce_cellprofiler_enum,
    normalize_cellprofiler_setting_name,
    parse_cellprofiler_bool,
    parse_cellprofiler_float,
    parse_cellprofiler_int,
)
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    FieldDerivedMeasurementFeatureModule,
    MeasurementFeatureRecord,
    NoFieldsMeasurementRecordMixin,
    NoObjectNameMeasurementRecordMixin,
    ProducedImagePayloadMeasurementRecordMixin,
)
from openhcs.interop.cellprofiler.runtime.measurement_rows import (
    ModuleOwnedResultMeasurementRows,
)
from openhcs.interop.cellprofiler.runtime.payload_types import CellProfilerKwargs
from openhcs.interop.cellprofiler.semantic_defaults import (
    SourceVolumetricPixelDataExecutionContract,
)
from openhcs.interop.cellprofiler.module_declarations import (
    ProcessingContract,
    BoundModuleSettings,
    CellProfilerModule,
    ImageArtifactInputModule,
    ImageArtifactOutputModule,
    LastRepeatedSettingValuePolicy,
    MeasurementArtifactOutputModule,
    RepeatedSettingValuePolicy,
)
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    CellProfilerBackendAuthority,
)
from openhcs.processing.backends.cellprofiler.enum_attributes import (
    CellProfilerEnumAttributeMixin,
)
from openhcs.processing.backends.cellprofiler.perf_fixtures import (
    capture_array_fixture,
    capture_enabled,
)
from openhcs.processing.backends.cellprofiler.thresholding_threshold_numba_diagnostics import (
    _deterministic_normal_noise,
    _quantized_log_tables,
    _threshold_diagnostics_numba,
    _threshold_diagnostics_unmasked_finite_numba,
    _threshold_weighted_variance_unmasked_finite_numba,
    rectangular_mask_domain,
    smooth_with_deterministic_noise,
)
from openhcs.processing.backends.cellprofiler.thresholding_threshold_numba_diagnostics_quantized import (
    QuantizedThresholdDiagnosticContext,
    _threshold_diagnostics_rectangular_mask_quantized_numba,
    _threshold_diagnostics_unmasked_finite_quantized_numba,
    exact_quantized_threshold_codes,
)
from openhcs.processing.backends.cellprofiler.thresholding_threshold_numba_otsu import (
    CELLPROFILER_LI_TOLERANCE,
    _binned_mode_numba,
    _finite_flat_float32,
    _finite_flat_float64,
    _inverse_log_transform_numba,
    _isodata_threshold_numba,
    _li_threshold_numba,
    _li_tolerance_numba,
    _log_transform_numba,
    _mad_numba,
    _mean_threshold_numba,
    _minimum_cross_entropy_threshold_numba,
    _minimum_threshold_numba,
    _multiotsu_three_class_thresholds_numba,
    _otsu_threshold_numba,
    _sauvola_threshold_image_numba,
    _triangle_threshold_numba,
    _weighted_otsu_threshold_numba_compatible,
    _yen_threshold_numba,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.materialization import csv_materializer
from openhcs.interop.cellprofiler.runtime.execution_mode_policies import (
    VolumetricInputExecutionModePolicy,
)

CELLPROFILER_BASIC_THRESHOLD_SMOOTHING_SCALE = 1.3488
THRESHOLD_BACKEND_REGISTRY_KEY = "backend_key"
SCIPY_CONSTANT_BOUNDARY_MODE = "constant"
CELLPROFILER_THRESHOLD_SMOOTHING_TRUNCATE_SIGMAS = 4.0
CELLPROFILER_THRESHOLD_SMOOTHING_HALF_MASS_FACTOR = 0.6744
CELLPROFILER_MULTI_OTSU_BINS = 128
CELLPROFILER_LOG_MULTI_OTSU_BINS = 128
CELLPROFILER_LOG_MULTI_OTSU_BIN_CENTER_OFFSET = 0.0
MAX_THRESHOLD_SELECTION_UNIT_INTERVAL_SCALE = int(np.iinfo(np.uint16).max)
logger = logging.getLogger(__name__)


def log_threshold_profile(label: str, seconds: float, **fields: object) -> None:
    """Emit threshold backend profile events through the shared runtime sink."""
    RuntimeProfileLogger.log(logger, label, seconds, **fields)


def threshold_profile_sink() -> Callable[..., None] | None:
    """Return the threshold profile sink only when runtime profiling is enabled."""
    return log_threshold_profile if RuntimeProfileLogger.enabled() else None


class CellProfilerThresholdAssignment(Enum):
    """Closed foreground/background assignment for multi-class CP thresholds."""

    FOREGROUND = "Foreground"
    BACKGROUND = "Background"


class CellProfilerAveragingMethod(Enum):
    """Closed CP robust-background center estimators."""

    MEAN = "Mean"
    MEDIAN = "Median"
    MODE = "Mode"


class CellProfilerThresholdMethod(CellProfilerEnumAttributeMixin, Enum):
    """Closed CP threshold methods with global-threshold source semantics."""

    __cellprofiler_attribute_names__ = (
        "_uses_raw_global_threshold_source",
        "_uses_raw_global_threshold_source_when_log_transformed",
    )
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

    def global_threshold_selection(
        self,
        *,
        log_transform: bool,
        image: np.ndarray,
        threshold_image: np.ndarray,
        method_parameters: "GlobalThresholdMethodParameters",
    ) -> "GlobalThresholdSourceSelection":
        """Return the source image and kwargs for global threshold estimation."""
        if self._uses_raw_global_threshold_source:
            return GlobalThresholdSourceSelection(np.asarray(image), method_parameters)
        if (
            self._uses_raw_global_threshold_source_when_log_transformed
            and log_transform
        ):
            return GlobalThresholdSourceSelection(
                np.asarray(image),
                method_parameters.with_multiotsu_nbins(
                    CELLPROFILER_LOG_MULTI_OTSU_BINS
                ),
            )
        return GlobalThresholdSourceSelection(threshold_image, method_parameters)


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


class GlobalThresholdKeywordArguments(TypedDict, total=False):
    """Keyword surface for CP global threshold method parameters."""

    lower_outlier_fraction: float
    upper_outlier_fraction: float
    averaging_method: CellProfilerAveragingMethod | str
    variance_method: CellProfilerVarianceMethod | str
    number_of_deviations: float
    nbins: int
    window_size: int
    fraction: float


class ThresholdProfileSink(Protocol):
    """Runtime threshold profiling sink."""

    def __call__(self, label: str, seconds: float, **fields: object) -> None:
        """Record a threshold profile event."""


class GlobalThresholdFunction(Protocol):
    """Callable surface for global threshold estimators."""

    def __call__(
        self,
        image: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        threshold_method: (
            CellProfilerThresholdMethod | str
        ) = CellProfilerThresholdMethod.OTSU,
        threshold_min: float = 0,
        threshold_max: float = 1,
        threshold_correction_factor: float = 1,
        assign_middle_to_foreground: (
            CellProfilerThresholdAssignment | str
        ) = CellProfilerThresholdAssignment.FOREGROUND,
        log_transform: bool = False,
        proven_unit_interval_scale: int | None = None,
        method_parameters: "GlobalThresholdMethodParameters | None" = None,
        **kwargs: Unpack[GlobalThresholdKeywordArguments],
    ) -> float:
        """Compute a global threshold."""


class AdaptiveThresholdFunction(Protocol):
    """Callable surface for adaptive threshold estimators."""

    def __call__(
        self,
        image: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        threshold_method: (
            CellProfilerThresholdMethod | str
        ) = CellProfilerThresholdMethod.OTSU,
        window_size: int = 50,
        threshold_min: float = 0,
        threshold_max: float = 1,
        threshold_correction_factor: float = 1,
        assign_middle_to_foreground: (
            CellProfilerThresholdAssignment | str
        ) = CellProfilerThresholdAssignment.FOREGROUND,
        global_limits: tuple[float, float] = (0.7, 1.5),
        log_transform: bool = False,
        global_threshold_function: GlobalThresholdFunction | None = None,
        method_parameters: "GlobalThresholdMethodParameters | None" = None,
        **kwargs: Unpack[GlobalThresholdKeywordArguments],
    ) -> np.ndarray:
        """Compute adaptive thresholds."""


class ThresholdApplicationFunction(Protocol):
    """Callable surface for threshold mask application."""

    def __call__(
        self,
        image: np.ndarray,
        *,
        threshold: float | np.ndarray,
        mask: np.ndarray | None,
        smoothing: float,
    ) -> tuple[np.ndarray, float]:
        """Apply a threshold to one image."""


class RobustBackgroundCenterStrategy(
    EnumKeyedStrategyMixin[CellProfilerAveragingMethod], ABC, metaclass=AutoRegisterMeta
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
        cls, averaging_method: CellProfilerAveragingMethod | str
    ) -> "RobustBackgroundCenterStrategy":
        resolved = coerce_cellprofiler_enum(
            CellProfilerAveragingMethod, averaging_method
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

    sink: ThresholdProfileSink
    function_name: str = "cellprofiler_threshold"

    @classmethod
    def from_sink(cls, sink: ThresholdProfileSink | None) -> Self:
        """Return a profiler bound to ``sink`` or a no-op sink."""
        return cls(cls.discard if sink is None else sink)

    @staticmethod
    def discard(label: str, seconds: float, **fields: object) -> None:
        """Drop one threshold profile event."""

    def record(
        self, phase_name: str, phase_started_at: float, **metadata: object
    ) -> None:
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
        self.record("threshold_apply", phase_started_at, smoothing=float(smoothing))


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
    EnumKeyedStrategyMixin[CellProfilerVarianceMethod], ABC, metaclass=AutoRegisterMeta
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
        cls, variance_method: CellProfilerVarianceMethod | str
    ) -> "RobustBackgroundSpreadStrategy":
        resolved = coerce_cellprofiler_enum(CellProfilerVarianceMethod, variance_method)
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

    lower_outlier_fraction: float = 0.05
    upper_outlier_fraction: float = 0.05
    averaging_method: CellProfilerAveragingMethod = CellProfilerAveragingMethod.MEAN
    variance_method: CellProfilerVarianceMethod = (
        CellProfilerVarianceMethod.STANDARD_DEVIATION
    )
    number_of_deviations: float = 2.0

    @classmethod
    def from_values(
        cls,
        *,
        lower_outlier_fraction: float = 0.05,
        upper_outlier_fraction: float = 0.05,
        averaging_method: (
            CellProfilerAveragingMethod | str
        ) = CellProfilerAveragingMethod.MEAN,
        variance_method: (
            CellProfilerVarianceMethod | str
        ) = CellProfilerVarianceMethod.STANDARD_DEVIATION,
        number_of_deviations: float = 2.0,
    ) -> Self:
        """Return canonical robust-background settings."""
        return cls(
            lower_outlier_fraction=float(lower_outlier_fraction),
            upper_outlier_fraction=float(upper_outlier_fraction),
            averaging_method=coerce_cellprofiler_enum(
                CellProfilerAveragingMethod, averaging_method
            ),
            variance_method=coerce_cellprofiler_enum(
                CellProfilerVarianceMethod, variance_method
            ),
            number_of_deviations=float(number_of_deviations),
        )

    def threshold(self, image: np.ndarray) -> float:
        """Return CP robust-background threshold for ``image``."""
        flat = np.asarray(image).flatten()
        if flat.size < 3:
            return 0.0
        flat.sort()
        if flat[0] == flat[-1]:
            return float(flat[0])
        low_chop = int(round(flat.size * self.lower_outlier_fraction))
        high_chop = flat.size - int(round(flat.size * self.upper_outlier_fraction))
        trimmed = flat if low_chop == 0 else flat[low_chop:high_chop]
        center = RobustBackgroundCenterStrategy.for_averaging_method(
            self.averaging_method
        ).center(trimmed)
        spread = RobustBackgroundSpreadStrategy.for_variance_method(
            self.variance_method
        ).spread(trimmed)
        return float(center + spread * self.number_of_deviations)


@dataclass(frozen=True, slots=True)
class GlobalThresholdMethodParameters:
    """Method-specific parameters for CellProfiler global thresholding."""

    robust_background: RobustBackgroundThresholdSettings = field(
        default_factory=RobustBackgroundThresholdSettings
    )
    multiotsu_nbins: int = CELLPROFILER_MULTI_OTSU_BINS
    sauvola_window_size: int = 15
    max_intensity_fraction: float = 0.75

    @classmethod
    def from_kwargs(cls, **kwargs: Unpack[GlobalThresholdKeywordArguments]) -> Self:
        """Return canonical method parameters from public threshold kwargs."""
        accepted = {
            "lower_outlier_fraction",
            "upper_outlier_fraction",
            "averaging_method",
            "variance_method",
            "number_of_deviations",
            "nbins",
            "window_size",
            "fraction",
        }
        unknown = set(kwargs) - accepted
        if unknown:
            raise TypeError(
                "Unknown CellProfiler global threshold parameter(s): "
                + ", ".join(sorted(unknown))
            )
        return cls(
            robust_background=RobustBackgroundThresholdSettings.from_values(
                lower_outlier_fraction=(
                    kwargs["lower_outlier_fraction"]
                    if "lower_outlier_fraction" in kwargs
                    else 0.05
                ),
                upper_outlier_fraction=(
                    kwargs["upper_outlier_fraction"]
                    if "upper_outlier_fraction" in kwargs
                    else 0.05
                ),
                averaging_method=(
                    kwargs["averaging_method"]
                    if "averaging_method" in kwargs
                    else CellProfilerAveragingMethod.MEAN
                ),
                variance_method=(
                    kwargs["variance_method"]
                    if "variance_method" in kwargs
                    else CellProfilerVarianceMethod.STANDARD_DEVIATION
                ),
                number_of_deviations=(
                    kwargs["number_of_deviations"]
                    if "number_of_deviations" in kwargs
                    else 2.0
                ),
            ),
            multiotsu_nbins=int(
                kwargs["nbins"] if "nbins" in kwargs else CELLPROFILER_MULTI_OTSU_BINS
            ),
            sauvola_window_size=int(
                kwargs["window_size"] if "window_size" in kwargs else 15
            ),
            max_intensity_fraction=float(
                kwargs["fraction"] if "fraction" in kwargs else 0.75
            ),
        )

    def with_multiotsu_nbins(self, nbins: int) -> Self:
        """Return a copy with the Multi-Otsu histogram resolution overridden."""
        return replace(self, multiotsu_nbins=int(nbins))


@dataclass(frozen=True, slots=True)
class GlobalThresholdSourceSelection:
    """Image and method parameters selected for one global threshold estimate."""

    image: np.ndarray
    method_parameters: GlobalThresholdMethodParameters


def normalize_cellprofiler_image(image: np.ndarray) -> np.ndarray:
    """Return an image in CellProfiler's normalized pixel-data convention."""
    return image_payload_data(
        normalize_cellprofiler_image_payload(
            image,
            dtype=np.float32,
            allow_unproven_uint8_float_domain=True,
        )
    )


def unit_interval_scale_for_threshold_selection(
    image_data: np.ndarray, metadata: ImagePayloadMetadata
) -> int | None:
    """Return a proof scale for exact unit-interval threshold selection."""
    metadata_scale = metadata.common_unit_interval_intensity_scale()
    if _threshold_selection_scale_is_supported(metadata_scale):
        return int(metadata_scale)
    image_array = np.asarray(image_data)
    if not np.issubdtype(image_array.dtype, np.integer):
        return None
    scale = image_intensity_scale_for_dtype(image_array.dtype)
    if not _threshold_selection_scale_is_supported(scale):
        return None
    return int(scale)


def _threshold_selection_scale_is_supported(scale: int | None) -> bool:
    """Return whether dense unit-interval threshold selection supports ``scale``."""
    return (
        scale is not None
        and scale > 1
        and (scale <= MAX_THRESHOLD_SELECTION_UNIT_INTERVAL_SCALE)
    )


def threshold_mask_for_image_domain(
    mask: np.ndarray | None, image_shape: tuple[int, ...], *, context: str
) -> np.ndarray | None:
    """Return the semantic threshold mask, dropping explicit all-true masks."""
    if mask is None:
        return None
    mask_array = np.asarray(mask, dtype=np.bool_)
    if mask_array.shape != image_shape:
        raise ValueError(
            f"{context} mask must match the image shape; got mask {mask_array.shape!r} for image {image_shape!r}."
        )
    if bool(np.all(mask_array)):
        return None
    return mask_array


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

    @staticmethod
    @lru_cache(maxsize=32)
    def gaussian_kernel_1d(sigma: float) -> np.ndarray:
        sigma_value = float(sigma)
        if sigma_value <= 0.0:
            return np.ones((1,), dtype=np.float64)
        radius = max(
            1,
            int(round(CELLPROFILER_THRESHOLD_SMOOTHING_TRUNCATE_SIGMAS * sigma_value)),
        )
        coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (coordinates / sigma_value) ** 2)
        kernel /= np.sum(kernel)
        kernel.setflags(write=False)
        return kernel

    def gaussian_filter(self, array: np.ndarray) -> np.ndarray:
        array_data = np.asarray(array, dtype=np.float64)
        from scipy import ndimage as ndi

        return ndi.gaussian_filter(
            array_data,
            sigma=self.sigma,
            mode=SCIPY_CONSTANT_BOUNDARY_MODE,
            cval=0,
            truncate=4.0,
        )

    @staticmethod
    @lru_cache(maxsize=32)
    def full_mask_weight(shape: tuple[int, ...], sigma: float) -> np.ndarray:
        mask = np.ones(shape, dtype=np.float64)
        from scipy import ndimage as ndi

        weight = ndi.gaussian_filter(
            mask, sigma=sigma, mode=SCIPY_CONSTANT_BOUNDARY_MODE, cval=0, truncate=4.0
        )
        weight.setflags(write=False)
        return weight

    def smooth(
        self, image: np.ndarray, mask: np.ndarray | None
    ) -> tuple[np.ndarray, float]:
        """Return the image CellProfiler thresholds against after estimation."""
        if not self.enabled:
            return (np.asarray(image), 0.0)
        image_array = np.asarray(image, dtype=np.float64)
        mask_array = threshold_mask_for_image_domain(
            mask, image_array.shape, context="Threshold application"
        )
        full_mask = mask_array is None
        if capture_enabled():
            capture_array_fixture(
                "threshold_application",
                image=image_array,
                mask=(
                    np.ones(image_array.shape, dtype=bool)
                    if mask_array is None
                    else mask_array
                ),
                smoothing=np.asarray(self.smoothing, dtype=np.float64),
            )
        masked_image = (
            image_array if full_mask else np.where(mask_array, image_array, 0.0)
        )
        smoothed_image = self.gaussian_filter(masked_image)
        mask_weight = (
            self.full_mask_weight(image_array.shape, self.sigma)
            if full_mask
            else self.gaussian_filter(mask_array.astype(np.float64))
        )
        denominator = mask_weight + np.finfo(float).eps
        if full_mask:
            smoothed_image /= denominator
            return (smoothed_image, self.sigma)
        output = np.zeros_like(image_array)
        valid = mask_weight != 0
        output[valid] = smoothed_image[valid] / denominator[valid]
        return (output, self.sigma)


@dataclass(frozen=True, slots=True)
class ThresholdApplicationRequest:
    """Executable CellProfiler threshold application request."""

    image: np.ndarray
    threshold: float | np.ndarray
    mask: np.ndarray | None = None
    smoothing: float = 0.0

    @property
    def normalized_mask(self) -> np.ndarray | None:
        return threshold_mask_for_image_domain(
            self.mask, np.asarray(self.image).shape, context="Threshold application"
        )

    def apply(self) -> tuple[np.ndarray, float]:
        mask = self.normalized_mask
        if self.smoothing == 0:
            thresholded = np.asarray(self.image) >= self.threshold
            if mask is None:
                return (thresholded, 0.0)
            return (thresholded & mask, 0.0)
        blurred_image, sigma = ThresholdApplicationSmoothing(self.smoothing).smooth(
            self.image, mask
        )
        thresholded = blurred_image >= self.threshold
        if mask is None:
            return (thresholded, sigma)
        return (thresholded & mask, sigma)


class ThresholdSmoothingBackendStrategy(
    CellProfilerBackendStrategyMixin, ABC, metaclass=AutoRegisterMeta
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

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY, CellProfilerBackendProvider.NUMBA
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
                f"CellProfiler threshold smoothing currently supports 2-D NumPy planes, got shape {image_array.shape!r}."
            )
        if mask_array.shape != image_array.shape:
            raise ValueError(
                f"Threshold smoothing mask must match the image shape; got mask {mask_array.shape!r} for image {image_array.shape!r}."
            )
        sigma, kernel = _threshold_smoothing_kernel(
            smoothing, threshold_method, log_transform=log_transform
        )
        return (
            _masked_kernel_convolution_2d_numba(image_array, mask_array, kernel),
            sigma,
        )


def _threshold_smoothing_kernel(
    smoothing: float, threshold_method: object | None, *, log_transform: bool = False
) -> tuple[float, np.ndarray]:
    sigma, radius = _threshold_smoothing_kernel_parameters(smoothing)
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    y, x = np.meshgrid(coordinates, coordinates, indexing="ij")
    radius_squared = float(radius * radius)
    distance_squared = x * x + y * y
    effective_sigma = 2.0 * sigma
    kernel = np.exp(-0.5 * distance_squared / (effective_sigma * effective_sigma))
    kernel[distance_squared > radius_squared] = 0.0
    kernel /= np.sum(kernel)
    return (sigma, kernel.astype(np.float64, copy=False))


def _threshold_smoothing_kernel_parameters(smoothing: float) -> tuple[float, int]:
    sigma = float(smoothing) / CELLPROFILER_THRESHOLD_SMOOTHING_HALF_MASS_FACTOR
    radius = max(
        1, int(math.ceil(sigma * CELLPROFILER_THRESHOLD_SMOOTHING_TRUNCATE_SIGMAS))
    )
    return (sigma, radius)


@njit(cache=True)
def _masked_kernel_convolution_2d_numba(
    image: np.ndarray, mask: np.ndarray, kernel: np.ndarray
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
                    if ix < 0 or ix >= width or (not mask[iy, ix]):
                        continue
                    kernel_value = kernel[ky, kx]
                    weight += kernel_value
                    weighted_sum += image[iy, ix] * kernel_value
            output[y, x] = weighted_sum / (weight + eps)
    return output


class ThresholdDiagnosticsBackendStrategy(
    CellProfilerBackendStrategyMixin, ABC, metaclass=AutoRegisterMeta
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
        self, image: np.ndarray, mask: np.ndarray, binary_image: np.ndarray
    ) -> float:
        """Compute weighted foreground/background log variance."""

    @abstractmethod
    def sum_of_entropies(
        self, image: np.ndarray, mask: np.ndarray, binary_image: np.ndarray
    ) -> float:
        """Compute foreground plus background log-histogram entropy."""


class NumpyThresholdDiagnosticsBackendStrategy(ThresholdDiagnosticsBackendStrategy):
    """Independent NumPy implementation of CellProfiler threshold diagnostics."""

    backend_key = CellProfilerBackendAuthority.backend_key(MemoryType.NUMPY)
    memory_type = MemoryType.NUMPY
    is_default_backend = False

    def weighted_variance(
        self, image: np.ndarray, mask: np.ndarray, binary_image: np.ndarray
    ) -> float:
        return _numpy_threshold_weighted_variance(image, mask, binary_image)

    def sum_of_entropies(
        self, image: np.ndarray, mask: np.ndarray, binary_image: np.ndarray
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
        image_array = np.asarray(image)
        binary_array = np.asarray(binary_image, dtype=np.bool_)
        if binary_array.shape != image_array.shape:
            raise ValueError(
                "Threshold diagnostics binary image must match the image shape; "
                f"got binary {binary_array.shape!r} for image {image_array.shape!r}."
            )
        mask_array = threshold_mask_for_image_domain(
            mask,
            image_array.shape,
            context="Threshold diagnostics",
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
    """Measure ND images as one CellProfiler image domain."""

    domain = ThresholdDiagnosticDomain.ND_IMAGE

    def diagnostics(self, request: ThresholdDiagnosticRequest) -> tuple[float, float]:
        return request.backend.diagnostics_whole_image(request)


class NumbaNumpyThresholdDiagnosticsBackendStrategy(
    ThresholdDiagnosticsBackendStrategy
):
    """Numba-accelerated NumPy implementation of threshold diagnostics."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY, CellProfilerBackendProvider.NUMBA
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def prepare_backend(self) -> None:
        image64 = np.linspace(0.0, 1.0, 16, dtype=np.float64).reshape((4, 4))
        image32 = image64.astype(np.float32)
        full_mask = np.ones(image32.shape, dtype=np.bool_)
        partial_mask = full_mask.copy()
        partial_mask[:, :1] = False
        for image in (image64, image32):
            binary = image > 0.5
            self.diagnostics(image, full_mask, binary)
            self.diagnostics(image[None, ...], full_mask[None, ...], binary[None, ...])
            self.diagnostics(image, partial_mask, binary)
            self.diagnostics(
                image[None, ...], partial_mask[None, ...], binary[None, ...]
            )
        quantized_image = np.rint(image32 * np.float32(255)) / np.float32(255)
        quantized_binary = quantized_image > 0.5
        self.diagnostics(
            quantized_image,
            None,
            quantized_binary,
            proven_unit_interval_scale=255,
        )
        self.diagnostics(
            quantized_image,
            partial_mask,
            quantized_binary,
            proven_unit_interval_scale=255,
        )
        self.diagnostics(
            quantized_image[None, ...],
            None,
            quantized_binary[None, ...],
            proven_unit_interval_scale=255,
        )
        self.diagnostics(
            quantized_image[None, ...],
            partial_mask[None, ...],
            quantized_binary[None, ...],
            proven_unit_interval_scale=255,
        )
        _quantized_log_tables(MAX_THRESHOLD_SELECTION_UNIT_INTERVAL_SCALE)

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
        return self._diagnostics_single_plane(
            request.image,
            request.mask,
            request.binary_image,
            request.proven_unit_interval_scale,
        )

    def _diagnostics_single_plane(
        self,
        image_array: np.ndarray,
        mask_array: np.ndarray | None,
        binary_array: np.ndarray,
        proven_unit_interval_scale: int | None,
    ) -> tuple[float, float]:
        if (
            (mask_array is None or bool(np.all(mask_array)))
            and bool(np.all(np.isfinite(image_array)))
        ):
            weighted_variance = _threshold_weighted_variance_unmasked_finite_numba(
                np.ascontiguousarray(image_array),
                np.ascontiguousarray(binary_array),
            )
            entropy_mask = (
                np.ones(image_array.shape, dtype=np.bool_)
                if mask_array is None
                else mask_array
            )
            return (
                weighted_variance,
                _numpy_threshold_sum_of_entropies(
                    image_array,
                    entropy_mask,
                    binary_array,
                ),
            )
        weighted_variance, _sum_of_entropies = self._diagnostics_planar(
            image_array,
            mask_array,
            binary_array,
            proven_unit_interval_scale,
        )
        entropy_mask = (
            np.ones(image_array.shape, dtype=np.bool_)
            if mask_array is None
            else mask_array
        )
        return (
            weighted_variance,
            _numpy_threshold_sum_of_entropies(
                image_array,
                entropy_mask,
                binary_array,
            ),
        )

    def _diagnostics_planar(
        self,
        image_array: np.ndarray,
        mask_array: np.ndarray | None,
        binary_array: np.ndarray,
        proven_unit_interval_scale: int | None,
    ) -> tuple[float, float]:
        if mask_array is None:
            self._validate_unmasked_inputs(image_array, binary_array)
            full_mask = True
        else:
            self._validate_inputs(image_array, mask_array, binary_array)
            full_mask = bool(np.all(mask_array))
            if not full_mask and proven_unit_interval_scale is not None:
                mask_domain = rectangular_mask_domain(mask_array)
                if mask_domain is not None:
                    cropped_image = image_array[mask_domain.slices]
                    codes = exact_quantized_threshold_codes(
                        image_array,
                        int(proven_unit_interval_scale),
                    )
                    if (
                        codes is not None
                        and bool(np.all(np.isfinite(cropped_image)))
                    ):
                        scale = int(proven_unit_interval_scale)
                        log_tables = _quantized_log_tables(scale)
                        context = QuantizedThresholdDiagnosticContext(
                            codes=codes,
                            binary_image=np.ascontiguousarray(binary_array),
                            noise=_deterministic_normal_noise(image_array.shape),
                            values=log_tables.values,
                            weighted_log_values=log_tables.weighted_log_values,
                            entropy_log_values=log_tables.entropy_log_values,
                            entropy_log_delta_values=(
                                log_tables.entropy_log_delta_values
                            ),
                        )
                        y_slice, x_slice = mask_domain.slices
                        weighted_variance, sum_of_entropies = (
                            _threshold_diagnostics_rectangular_mask_quantized_numba(
                                context,
                                int(y_slice.start),
                                int(y_slice.stop),
                                int(x_slice.start),
                                int(x_slice.stop),
                            )
                        )
                        return (float(weighted_variance), float(sum_of_entropies))
        if full_mask and bool(np.all(np.isfinite(image_array))):
            weighted_variance, sum_of_entropies = (
                _threshold_diagnostics_unmasked_finite_numba(
                    np.ascontiguousarray(image_array),
                    np.ascontiguousarray(binary_array),
                    _deterministic_normal_noise(image_array.shape),
                )
            )
            return (float(weighted_variance), float(sum_of_entropies))
        if mask_array is None:
            mask_array = np.ones(image_array.shape, dtype=np.bool_)
            weighted_variance, sum_of_entropies = _threshold_diagnostics_numba(
                np.ascontiguousarray(image_array),
                np.ascontiguousarray(mask_array),
                np.ascontiguousarray(binary_array),
                _deterministic_normal_noise(image_array.shape),
            )
            return (float(weighted_variance), float(sum_of_entropies))
        weighted_variance, sum_of_entropies = _threshold_diagnostics_numba(
            np.ascontiguousarray(image_array),
            np.ascontiguousarray(mask_array),
            np.ascontiguousarray(binary_array),
            _deterministic_normal_noise(image_array.shape),
        )
        return (float(weighted_variance), float(sum_of_entropies))

    def diagnostics_whole_image(
        self,
        request: ThresholdDiagnosticRequest,
    ) -> tuple[float, float]:
        """Evaluate an ND CellProfiler image as one flattened measurement domain."""
        image_array = request.image
        binary_array = request.binary_image
        flat_image = np.ascontiguousarray(image_array.reshape(-1, 1))
        flat_binary = np.ascontiguousarray(binary_array.reshape(-1, 1))
        flat_mask = None
        if request.mask is not None:
            flat_mask = np.ascontiguousarray(request.mask.reshape(-1, 1))
        noise = _deterministic_normal_noise(image_array.shape).reshape(-1, 1)

        full_mask = flat_mask is None or bool(np.all(flat_mask))
        if full_mask and bool(np.all(np.isfinite(flat_image))):
            if request.proven_unit_interval_scale is not None:
                scale = int(request.proven_unit_interval_scale)
                codes = exact_quantized_threshold_codes(image_array, scale)
                if codes is not None:
                    log_tables = _quantized_log_tables(scale)
                    context = QuantizedThresholdDiagnosticContext(
                        codes=np.ascontiguousarray(codes.reshape(-1, 1)),
                        binary_image=flat_binary,
                        noise=noise,
                        values=log_tables.values,
                        weighted_log_values=log_tables.weighted_log_values,
                        entropy_log_values=log_tables.entropy_log_values,
                        entropy_log_delta_values=(
                            log_tables.entropy_log_delta_values
                        ),
                    )
                    weighted_variance, sum_of_entropies = (
                        _threshold_diagnostics_unmasked_finite_quantized_numba(
                            context
                        )
                    )
                    return (float(weighted_variance), float(sum_of_entropies))
            weighted_variance, sum_of_entropies = (
                _threshold_diagnostics_unmasked_finite_numba(
                    flat_image,
                    flat_binary,
                    noise,
                )
            )
            return (float(weighted_variance), float(sum_of_entropies))

        if flat_mask is None:
            flat_mask = np.ones(flat_image.shape, dtype=np.bool_)
        weighted_variance, sum_of_entropies = _threshold_diagnostics_numba(
            flat_image,
            flat_mask,
            flat_binary,
            noise,
        )
        return (float(weighted_variance), float(sum_of_entropies))

    def _validate_unmasked_inputs(
        self, image_array: np.ndarray, binary_array: np.ndarray
    ) -> None:
        if image_array.ndim != 2:
            raise NotImplementedError(
                f"CellProfiler threshold diagnostics currently support 2-D NumPy planes, got shape {image_array.shape!r}."
            )
        if binary_array.shape != image_array.shape:
            raise ValueError(
                f"Threshold diagnostics binary image must match the image shape; got binary {binary_array.shape!r} for image {image_array.shape!r}."
            )

    def weighted_variance(
        self, image: np.ndarray, mask: np.ndarray, binary_image: np.ndarray
    ) -> float:
        image_array = np.asarray(image)
        mask_array = np.asarray(mask, dtype=np.bool_)
        binary_array = np.asarray(binary_image, dtype=np.bool_)
        self._validate_inputs(image_array, mask_array, binary_array)
        return _numpy_threshold_weighted_variance(
            image_array, mask_array, binary_array
        )

    def sum_of_entropies(
        self, image: np.ndarray, mask: np.ndarray, binary_image: np.ndarray
    ) -> float:
        image_array = np.asarray(image)
        mask_array = np.asarray(mask, dtype=np.bool_)
        binary_array = np.asarray(binary_image, dtype=np.bool_)
        self._validate_inputs(image_array, mask_array, binary_array)
        return _numpy_threshold_sum_of_entropies(
            image_array, mask_array, binary_array
        )

    def _validate_inputs(
        self, image_array: np.ndarray, mask_array: np.ndarray, binary_array: np.ndarray
    ) -> None:
        if image_array.ndim != 2:
            raise NotImplementedError(
                f"CellProfiler threshold diagnostics currently support 2-D NumPy planes, got shape {image_array.shape!r}."
            )
        if mask_array.shape != image_array.shape:
            raise ValueError(
                f"Threshold diagnostics mask must match the image shape; got mask {mask_array.shape!r} for image {image_array.shape!r}."
            )
        if binary_array.shape != image_array.shape:
            raise ValueError(
                f"Threshold diagnostics binary image must match the image shape; got binary {binary_array.shape!r} for image {image_array.shape!r}."
            )


class ThresholdPrimitiveBackendStrategy(
    CellProfilerBackendStrategyMixin, ABC, metaclass=AutoRegisterMeta
):
    """Small threshold helper primitives supplied by an explicit provider."""

    __registry_key__ = THRESHOLD_BACKEND_REGISTRY_KEY
    __skip_if_no_key__ = True

    @abstractmethod
    def log_transform(self, values: np.ndarray) -> tuple[np.ndarray, object]:
        """Return CP-compatible log-transformed values and conversion state."""

    @abstractmethod
    def inverse_log_transform(
        self, values: float | np.ndarray, conversion: object
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
        self, image: np.ndarray, *, window_size: int
    ) -> np.ndarray:
        """Return per-pixel Sauvola thresholds."""

    @abstractmethod
    def minimum_cross_entropy_threshold(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
        *,
        proven_unit_interval_scale: int | None = None,
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

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY, CellProfilerBackendProvider.NUMBA
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def prepare_backend(self) -> None:
        values = np.linspace(0.01, 1.0, 32, dtype=np.float64)
        image = values[:25].reshape((5, 5))
        quantized_image = np.rint(
            image.astype(np.float32) * np.float32(255)
        ) / np.float32(255)
        quantized_mask = np.ones(quantized_image.shape, dtype=np.bool_)
        quantized_mask[:, :1] = False
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
        self.minimum_cross_entropy_threshold(
            quantized_image, proven_unit_interval_scale=255
        )
        self.minimum_cross_entropy_threshold(
            quantized_image, mask=quantized_mask, proven_unit_interval_scale=255
        )

    def log_transform(self, values: np.ndarray) -> tuple[np.ndarray, object]:
        values_array = np.asarray(values, dtype=np.float32)
        transformed, noise_min, log_min, log_max = _log_transform_numba(
            np.ascontiguousarray(values_array.ravel())
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
        self, values: float | np.ndarray, conversion: object
    ) -> float | np.ndarray:
        if not isinstance(conversion, NumbaLogTransformConversion):
            raise TypeError(
                "Numba threshold primitive inverse_log_transform requires NumbaLogTransformConversion state."
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
            np.ascontiguousarray(values_array.ravel())
        )
        threshold = _weighted_otsu_threshold_numba_compatible(transformed, 256)
        return float(
            _inverse_log_transform_numba(
                np.asarray([threshold], dtype=np.float64), log_min, log_max
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
            _finite_flat_float64(values), int(nbins)
        )

    def sauvola_threshold_image(
        self, image: np.ndarray, *, window_size: int
    ) -> np.ndarray:
        image_array = np.asarray(image, dtype=np.float64)
        if image_array.ndim != 2:
            raise NotImplementedError(
                f"CellProfiler Sauvola thresholding currently supports 2-D NumPy planes, got shape {image_array.shape!r}."
            )
        return _sauvola_threshold_image_numba(
            np.ascontiguousarray(image_array), int(window_size), 0.2, 1.0
        )

    def minimum_cross_entropy_threshold(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
        *,
        proven_unit_interval_scale: int | None = None,
    ) -> float:
        image_array = np.asarray(image)
        if proven_unit_interval_scale is not None:
            return _li_threshold_quantized_numpy(
                image_array, mask, int(proven_unit_interval_scale)
            )
        if mask is None:
            if image_array.dtype == np.float32:
                values32 = _finite_flat_float32(image_array)
                return _li_threshold_float32_numpy(values32)
            values = _finite_flat_float64(image_array)
        else:
            mask_array = np.asarray(mask, dtype=bool)
            if mask_array.shape != image_array.shape:
                raise ValueError(
                    f"Minimum cross-entropy mask must match the image shape; got mask {mask_array.shape!r} for image {image_array.shape!r}."
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

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY, CellProfilerBackendProvider.CENTROSOME
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.CENTROSOME
    is_default_backend = False

    def log_transform(self, values: np.ndarray) -> tuple[np.ndarray, object]:
        import centrosome.threshold

        return centrosome.threshold.log_transform(values)

    def inverse_log_transform(
        self, values: float | np.ndarray, conversion: object
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
            "Centrosome threshold primitive backend does not provide Li thresholding. Select the Numba backend explicitly."
        )

    def triangle_threshold(self, values: np.ndarray) -> float:
        raise NotImplementedError(
            "Centrosome threshold primitive backend does not provide Triangle thresholding. Select the Numba backend explicitly."
        )

    def isodata_threshold(self, values: np.ndarray) -> float:
        raise NotImplementedError(
            "Centrosome threshold primitive backend does not provide Isodata thresholding. Select the Numba backend explicitly."
        )

    def mean_threshold(self, values: np.ndarray) -> float:
        raise NotImplementedError(
            "Centrosome threshold primitive backend does not provide Mean thresholding. Select the Numba backend explicitly."
        )

    def yen_threshold(self, values: np.ndarray) -> float:
        raise NotImplementedError(
            "Centrosome threshold primitive backend does not provide Yen thresholding. Select the Numba backend explicitly."
        )

    def minimum_threshold(self, values: np.ndarray) -> float:
        raise NotImplementedError(
            "Centrosome threshold primitive backend does not provide histogram Minimum thresholding. Select the Numba backend explicitly."
        )

    def multiotsu_thresholds(self, values: np.ndarray, *, nbins: int) -> np.ndarray:
        raise NotImplementedError(
            "Centrosome threshold primitive backend does not provide Multi-Otsu thresholding. Select the Numba backend explicitly."
        )

    def sauvola_threshold_image(
        self, image: np.ndarray, *, window_size: int
    ) -> np.ndarray:
        raise NotImplementedError(
            "Centrosome threshold primitive backend does not provide Sauvola thresholding. Select the Numba backend explicitly."
        )

    def minimum_cross_entropy_threshold(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
        *,
        proven_unit_interval_scale: int | None = None,
    ) -> float:
        del proven_unit_interval_scale
        raise NotImplementedError(
            "Centrosome threshold primitive backend does not provide CP-style minimum cross-entropy thresholding. Select the Numba backend explicitly."
        )


def threshold_primitives(
    *, backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION
) -> ThresholdPrimitiveBackendStrategy:
    """Return the selected threshold primitive backend."""
    return ThresholdPrimitiveBackendStrategy.for_memory_type(
        MemoryType.NUMPY, backend_provider=backend_provider
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
    proven_unit_interval_scale: int | None
    method_parameters: GlobalThresholdMethodParameters


@dataclass(frozen=True, slots=True)
class CellProfilerThresholdResult:
    """Threshold mask and diagnostics from one CP-compatible threshold request."""

    final_threshold: float
    original_threshold: float
    mask: np.ndarray
    weighted_variance: float = 0.0
    sum_of_entropies: float = 0.0


@dataclass(frozen=True, slots=True)
class CellProfilerThresholdSettings:
    """CellProfiler threshold settings independent of image provenance."""

    use_advanced_settings: bool
    threshold_scope: CellProfilerThresholdScope | str
    threshold_method: CellProfilerThresholdMethod | str
    threshold_correction_factor: float
    threshold_min: float
    threshold_max: float
    threshold_smoothing_scale: float
    otsu_class_count: CellProfilerOtsuMethod | str
    assign_middle_to_foreground: CellProfilerThresholdAssignment | str
    log_transform: bool
    adaptive_window_size: int
    lower_outlier_fraction: float
    upper_outlier_fraction: float
    averaging_method: CellProfilerAveragingMethod | str
    variance_method: CellProfilerVarianceMethod | str
    number_of_deviations: float
    manual_threshold: float
    smooth_threshold_application: bool = True

    def normalized(self) -> Self:
        """Return settings with CellProfiler enum values and basic-mode defaults."""
        settings = replace(
            self,
            threshold_scope=coerce_cellprofiler_enum(
                CellProfilerThresholdScope, self.threshold_scope
            ),
            threshold_method=coerce_cellprofiler_enum(
                CellProfilerThresholdMethod, self.threshold_method
            ),
            otsu_class_count=coerce_cellprofiler_enum(
                CellProfilerOtsuMethod, self.otsu_class_count
            ),
            assign_middle_to_foreground=coerce_cellprofiler_enum(
                CellProfilerThresholdAssignment, self.assign_middle_to_foreground
            ),
            averaging_method=coerce_cellprofiler_enum(
                CellProfilerAveragingMethod, self.averaging_method
            ),
            variance_method=coerce_cellprofiler_enum(
                CellProfilerVarianceMethod, self.variance_method
            ),
        )
        if settings.use_advanced_settings:
            return settings
        return replace(
            settings,
            threshold_scope=CellProfilerThresholdScope.GLOBAL,
            threshold_method=CellProfilerThresholdMethod.MINIMUM_CROSS_ENTROPY,
            log_transform=False,
            threshold_smoothing_scale=CELLPROFILER_BASIC_THRESHOLD_SMOOTHING_SCALE,
        )

    def effective_method(self) -> CellProfilerThresholdMethod:
        """Return the method CP actually evaluates for these settings."""
        return threshold_method_for_class_count(
            coerce_cellprofiler_enum(
                CellProfilerThresholdMethod, self.threshold_method
            ),
            coerce_cellprofiler_enum(CellProfilerOtsuMethod, self.otsu_class_count),
        )

    def method_parameters(self) -> GlobalThresholdMethodParameters:
        """Return global-threshold method parameters for these settings."""
        return GlobalThresholdMethodParameters(
            robust_background=RobustBackgroundThresholdSettings.from_values(
                lower_outlier_fraction=self.lower_outlier_fraction,
                upper_outlier_fraction=self.upper_outlier_fraction,
                averaging_method=self.averaging_method,
                variance_method=self.variance_method,
                number_of_deviations=self.number_of_deviations,
            )
        )

    def application_smoothing(self) -> float:
        """Return the smoothing used when applying the final threshold."""
        if not self.smooth_threshold_application:
            return 0.0
        return float(self.threshold_smoothing_scale)

    def with_threshold_module_controls(
        self, *, automatic: bool, predefined_threshold: float | None
    ) -> Self:
        """Return settings after applying public Threshold-module controls."""
        settings = self
        if automatic:
            settings = replace(
                settings,
                threshold_scope=CellProfilerThresholdScope.GLOBAL,
                threshold_method=CellProfilerThresholdMethod.MINIMUM_CROSS_ENTROPY,
                log_transform=False,
                threshold_smoothing_scale=1.0,
            )
        method = coerce_cellprofiler_enum(
            CellProfilerThresholdMethod, settings.threshold_method
        )
        manual_threshold = predefined_threshold
        if method is CellProfilerThresholdMethod.MANUAL and manual_threshold is None:
            manual_threshold = 0.0
        if manual_threshold is None:
            return settings
        return replace(
            settings,
            threshold_method=CellProfilerThresholdMethod.MANUAL,
            manual_threshold=float(manual_threshold),
        )


@dataclass(frozen=True, slots=True)
class CellProfilerThresholdRequest:
    """Closed CP-compatible threshold execution contract."""

    image: np.ndarray
    image_mask: np.ndarray | None
    settings: CellProfilerThresholdSettings
    proven_unit_interval_scale: int | None = None
    enabled: bool = True
    log_profile_function: ThresholdProfileSink | None = None

    def calculate(self) -> CellProfilerThresholdResult:
        """Apply thresholding and compute CP threshold diagnostics."""
        thresholded, threshold_value, original_threshold = self.threshold_tuple()
        if not self.enabled:
            return CellProfilerThresholdResult(
                final_threshold=float(threshold_value),
                original_threshold=float(original_threshold),
                mask=thresholded,
            )
        diagnostics = cellprofiler_threshold_diagnostics(
            self.image,
            thresholded,
            final_threshold=threshold_value,
            original_threshold=original_threshold,
            mask=self.image_mask,
            proven_unit_interval_scale=self.proven_unit_interval_scale,
            log_profile_function=self.log_profile_function,
        )
        return CellProfilerThresholdResult(
            final_threshold=float(threshold_value),
            original_threshold=float(diagnostics.original_threshold),
            mask=thresholded,
            weighted_variance=float(diagnostics.weighted_variance),
            sum_of_entropies=float(diagnostics.sum_of_entropies),
        )

    def threshold_tuple(
        self,
        *,
        global_threshold_function: GlobalThresholdFunction | None = None,
        adaptive_threshold_function: AdaptiveThresholdFunction | None = None,
        apply_threshold_function: ThresholdApplicationFunction | None = None,
    ) -> tuple[np.ndarray, float, float]:
        """Apply thresholding and return ``(mask, final, original)``."""
        if not self.enabled:
            return (self.threshold_free_mask, 0.0, 0.0)
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
        profiler = CellProfilerThresholdProfiler.from_sink(self.log_profile_function)
        total_started_at = time.perf_counter()
        phase_started_at = time.perf_counter()
        threshold_image = np.asarray(self.image)
        threshold_mask = threshold_mask_for_image_domain(
            self.image_mask, threshold_image.shape, context="CellProfiler threshold"
        )
        settings = self.settings.normalized()
        profiler.record("threshold_coerce_settings", phase_started_at)
        if settings.threshold_method is CellProfilerThresholdMethod.MEASUREMENT:
            raise NotImplementedError(
                "Measurement-based thresholding requires a prior measurement source."
            )
        method_parameters = settings.method_parameters()
        effective_method = settings.effective_method()
        if settings.threshold_method is CellProfilerThresholdMethod.MANUAL:
            final_threshold: float | np.ndarray = float(settings.manual_threshold)
            original_threshold = float(settings.manual_threshold)
        elif settings.threshold_scope is CellProfilerThresholdScope.ADAPTIVE:
            final_threshold, original_threshold = self._adaptive_thresholds(
                settings=settings,
                effective_method=effective_method,
                method_parameters=method_parameters,
                threshold_image=threshold_image,
                threshold_mask=threshold_mask,
                adaptive_threshold=adaptive_threshold,
                global_threshold=global_threshold,
                profiler=profiler,
            )
        else:
            final_threshold, original_threshold = self._global_thresholds(
                settings=settings,
                effective_method=effective_method,
                method_parameters=method_parameters,
                threshold_image=threshold_image,
                threshold_mask=threshold_mask,
                global_threshold=global_threshold,
                profiler=profiler,
            )
        application_smoothing = settings.application_smoothing()
        phase_started_at = time.perf_counter()
        if apply_threshold_function is None:
            binary, _sigma = ThresholdApplicationRequest(
                image=self.image,
                threshold=final_threshold,
                mask=threshold_mask,
                smoothing=application_smoothing,
            ).apply()
        else:
            binary, _sigma = apply_threshold_function(
                self.image,
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

    @property
    def threshold_free_mask(self) -> np.ndarray:
        """Return the all-foreground mask for methods without thresholding."""
        if self.image_mask is None:
            return np.ones_like(self.image, dtype=bool)
        return np.asarray(self.image_mask, dtype=bool)

    def _adaptive_thresholds(
        self,
        *,
        settings: CellProfilerThresholdSettings,
        effective_method: CellProfilerThresholdMethod,
        method_parameters: GlobalThresholdMethodParameters,
        threshold_image: np.ndarray,
        threshold_mask: np.ndarray | None,
        adaptive_threshold: AdaptiveThresholdFunction,
        global_threshold: GlobalThresholdFunction,
        profiler: CellProfilerThresholdProfiler,
    ) -> tuple[float | np.ndarray, float]:
        phase_started_at = time.perf_counter()
        final_threshold = adaptive_threshold(
            threshold_image,
            mask=threshold_mask,
            threshold_method=effective_method,
            window_size=settings.adaptive_window_size,
            threshold_min=settings.threshold_min,
            threshold_max=settings.threshold_max,
            threshold_correction_factor=settings.threshold_correction_factor,
            assign_middle_to_foreground=settings.assign_middle_to_foreground,
            log_transform=settings.log_transform,
            global_threshold_function=global_threshold,
            method_parameters=method_parameters,
        )
        profiler.record_method(
            "threshold_adaptive_final", phase_started_at, effective_method
        )
        phase_started_at = time.perf_counter()
        original_threshold = float(
            np.mean(
                np.atleast_1d(
                    adaptive_threshold(
                        threshold_image,
                        mask=threshold_mask,
                        threshold_method=effective_method,
                        window_size=settings.adaptive_window_size,
                        threshold_min=(
                            settings.threshold_min
                            if not settings.use_advanced_settings
                            else 0
                        ),
                        threshold_max=(
                            settings.threshold_max
                            if not settings.use_advanced_settings
                            else 1
                        ),
                        threshold_correction_factor=(
                            settings.threshold_correction_factor
                            if not settings.use_advanced_settings
                            else 1
                        ),
                        assign_middle_to_foreground=settings.assign_middle_to_foreground,
                        log_transform=settings.log_transform,
                        global_threshold_function=global_threshold,
                        method_parameters=method_parameters,
                    )
                )
            )
        )
        profiler.record_method(
            "threshold_adaptive_original", phase_started_at, effective_method
        )
        return (final_threshold, original_threshold)

    def _global_thresholds(
        self,
        *,
        settings: CellProfilerThresholdSettings,
        effective_method: CellProfilerThresholdMethod,
        method_parameters: GlobalThresholdMethodParameters,
        threshold_image: np.ndarray,
        threshold_mask: np.ndarray | None,
        global_threshold: GlobalThresholdFunction,
        profiler: CellProfilerThresholdProfiler,
    ) -> tuple[float, float]:
        selection = effective_method.global_threshold_selection(
            log_transform=settings.log_transform,
            image=self.image,
            threshold_image=threshold_image,
            method_parameters=method_parameters,
        )
        phase_started_at = time.perf_counter()
        raw_threshold = global_threshold(
            selection.image,
            mask=threshold_mask,
            threshold_method=effective_method,
            threshold_min=0,
            threshold_max=1,
            threshold_correction_factor=1,
            assign_middle_to_foreground=settings.assign_middle_to_foreground,
            log_transform=settings.log_transform,
            proven_unit_interval_scale=self.proven_unit_interval_scale,
            method_parameters=selection.method_parameters,
        )
        profiler.record_global_raw(phase_started_at, effective_method, selection.image)
        phase_started_at = time.perf_counter()
        final_threshold = clip_threshold(
            raw_threshold * settings.threshold_correction_factor,
            settings.threshold_min,
            settings.threshold_max,
        )
        original_threshold = (
            final_threshold
            if not settings.use_advanced_settings
            else clip_threshold(raw_threshold, 0, 1)
        )
        profiler.record_method("threshold_clip", phase_started_at, effective_method)
        return (final_threshold, original_threshold)


class GlobalThresholdMethodStrategy(
    EnumKeyedStrategyMixin[CellProfilerThresholdMethod], ABC, metaclass=AutoRegisterMeta
):
    """Nominal implementation for one CellProfiler global threshold method."""

    __registry_key__ = "method_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "method"
    method: ClassVar[CellProfilerThresholdMethod]
    method_label: ClassVar[str | None] = None

    @classmethod
    def for_method(
        cls, method: CellProfilerThresholdMethod
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
            proven_unit_interval_scale=request.proven_unit_interval_scale,
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
        return request.method_parameters.robust_background.threshold(request.values)


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
            0 if request.assignment is CellProfilerThresholdAssignment.FOREGROUND else 1
        )
        nbins = request.method_parameters.multiotsu_nbins
        thresholds = request.primitives.multiotsu_thresholds(
            request.values, nbins=nbins
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
                    window_size=request.method_parameters.sauvola_window_size,
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


class MaxIntensityPercentageGlobalThresholdStrategy(
    HelperBackedGlobalThresholdStrategy
):
    method = CellProfilerThresholdMethod.MAX_INTENSITY_PERCENTAGE
    method_label = method.value

    @staticmethod
    def _threshold_helper(request: GlobalThresholdRequest) -> float:
        return float(
            np.max(request.values) * request.method_parameters.max_intensity_fraction
        )


def cellprofiler_get_global_threshold(
    image: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    threshold_method: (
        CellProfilerThresholdMethod | str
    ) = CellProfilerThresholdMethod.OTSU,
    threshold_min: float = 0,
    threshold_max: float = 1,
    threshold_correction_factor: float = 1,
    assign_middle_to_foreground: (
        CellProfilerThresholdAssignment | str
    ) = CellProfilerThresholdAssignment.FOREGROUND,
    log_transform: bool = False,
    proven_unit_interval_scale: int | None = None,
    method_parameters: GlobalThresholdMethodParameters | None = None,
    **kwargs: Unpack[GlobalThresholdKeywordArguments],
) -> float:
    """Compute one global threshold using independent CP-compatible semantics."""
    primitives = threshold_primitives()
    method = coerce_cellprofiler_enum(CellProfilerThresholdMethod, threshold_method)
    assignment = coerce_cellprofiler_enum(
        CellProfilerThresholdAssignment, assign_middle_to_foreground
    )
    if method_parameters is not None and kwargs:
        raise TypeError(
            "Pass either method_parameters or individual threshold method keyword arguments, not both."
        )
    resolved_parameters = (
        GlobalThresholdMethodParameters.from_kwargs(**kwargs)
        if method_parameters is None
        else method_parameters
    )
    threshold_image = np.asarray(image, dtype=np.float32)
    if log_transform:
        threshold_image, conversion = primitives.log_transform(threshold_image)
    else:
        conversion = None
    threshold_mask = threshold_mask_for_image_domain(
        mask, threshold_image.shape, context="Global threshold"
    )
    values = (
        threshold_image[threshold_mask]
        if threshold_mask is not None
        else threshold_image.ravel()
    )
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
                proven_unit_interval_scale=(
                    proven_unit_interval_scale if not log_transform else None
                ),
                method_parameters=resolved_parameters,
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
    threshold_method: (
        CellProfilerThresholdMethod | str
    ) = CellProfilerThresholdMethod.OTSU,
    window_size: int = 50,
    threshold_min: float = 0,
    threshold_max: float = 1,
    threshold_correction_factor: float = 1,
    assign_middle_to_foreground: (
        CellProfilerThresholdAssignment | str
    ) = CellProfilerThresholdAssignment.FOREGROUND,
    global_limits: tuple[float, float] = (0.7, 1.5),
    log_transform: bool = False,
    global_threshold_function: GlobalThresholdFunction | None = None,
    method_parameters: GlobalThresholdMethodParameters | None = None,
    **kwargs: Unpack[GlobalThresholdKeywordArguments],
) -> np.ndarray:
    """Compute CP-style adaptive thresholds without depending on CP packages."""
    primitives = threshold_primitives()
    global_threshold = (
        cellprofiler_get_global_threshold
        if global_threshold_function is None
        else global_threshold_function
    )
    if method_parameters is not None and kwargs:
        raise TypeError(
            "Pass either method_parameters or individual threshold method keyword arguments, not both."
        )
    resolved_parameters = (
        GlobalThresholdMethodParameters.from_kwargs(**kwargs)
        if method_parameters is None
        else method_parameters
    )
    method = coerce_cellprofiler_enum(CellProfilerThresholdMethod, threshold_method)
    assignment = coerce_cellprofiler_enum(
        CellProfilerThresholdAssignment, assign_middle_to_foreground
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
            transformed, window_size=window_size
        )
    else:
        thresholds = adaptive_threshold_blocks(
            transformed,
            window_size=window_size,
            threshold_method=method,
            assign_middle_to_foreground=assignment,
            global_threshold_function=global_threshold,
            method_parameters=resolved_parameters,
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
        method_parameters=resolved_parameters,
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
    averaging_method: (
        CellProfilerAveragingMethod | str
    ) = CellProfilerAveragingMethod.MEAN,
    variance_method: (
        CellProfilerVarianceMethod | str
    ) = CellProfilerVarianceMethod.STANDARD_DEVIATION,
    number_of_deviations: float = 2,
) -> float:
    return RobustBackgroundThresholdSettings.from_values(
        lower_outlier_fraction=lower_outlier_fraction,
        upper_outlier_fraction=upper_outlier_fraction,
        averaging_method=averaging_method,
        variance_method=variance_method,
        number_of_deviations=number_of_deviations,
    ).threshold(image)


def adaptive_threshold_blocks(
    image: np.ndarray,
    *,
    window_size: int,
    threshold_method: CellProfilerThresholdMethod,
    assign_middle_to_foreground: CellProfilerThresholdAssignment,
    method_parameters: GlobalThresholdMethodParameters,
    global_threshold_function: GlobalThresholdFunction | None = None,
) -> np.ndarray:
    image_size = np.array(image.shape[:2], dtype=int)
    nblocks = image_size // window_size
    if any((count < 2 for count in nblocks)):
        raise ValueError(
            f"Adaptive window cannot exceed 50% of an image dimension.\nWindow of {window_size}px is too large for a {image_size[1]}x{image_size[0]} image"
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
                method_parameters=method_parameters,
                global_threshold_function=global_threshold_function,
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
    method_parameters: GlobalThresholdMethodParameters,
    global_threshold_function: GlobalThresholdFunction | None = None,
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
        method_parameters=method_parameters,
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


def clip_threshold(
    threshold: float, threshold_min: float, threshold_max: float
) -> float:
    return float(min(max(float(threshold), threshold_min), threshold_max))


def cellprofiler_threshold(
    image: np.ndarray,
    *,
    use_advanced_settings: bool,
    threshold_scope: CellProfilerThresholdScope | str,
    threshold_method: CellProfilerThresholdMethod | str,
    otsu_class_count: CellProfilerOtsuMethod | str,
    assign_middle_to_foreground: CellProfilerThresholdAssignment | str,
    log_transform: bool,
    threshold_correction_factor: float,
    threshold_min: float,
    threshold_max: float,
    threshold_smoothing_scale: float,
    adaptive_window_size: int,
    lower_outlier_fraction: float,
    upper_outlier_fraction: float,
    averaging_method: CellProfilerAveragingMethod | str,
    variance_method: CellProfilerVarianceMethod | str,
    number_of_deviations: float,
    manual_threshold: float,
    mask: np.ndarray | None = None,
    smooth_threshold_application: bool = True,
    proven_unit_interval_scale: int | None = None,
    global_threshold_function: GlobalThresholdFunction | None = None,
    adaptive_threshold_function: AdaptiveThresholdFunction | None = None,
    apply_threshold_function: ThresholdApplicationFunction | None = None,
    log_profile_function: ThresholdProfileSink | None = None,
) -> tuple[np.ndarray, float, float]:
    """Apply CellProfiler threshold semantics without a CP workspace."""
    return CellProfilerThresholdRequest(
        image=image,
        image_mask=mask,
        settings=CellProfilerThresholdSettings(
            use_advanced_settings=use_advanced_settings,
            threshold_scope=threshold_scope,
            threshold_method=threshold_method,
            threshold_correction_factor=threshold_correction_factor,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            threshold_smoothing_scale=threshold_smoothing_scale,
            otsu_class_count=otsu_class_count,
            assign_middle_to_foreground=assign_middle_to_foreground,
            log_transform=log_transform,
            adaptive_window_size=adaptive_window_size,
            lower_outlier_fraction=lower_outlier_fraction,
            upper_outlier_fraction=upper_outlier_fraction,
            averaging_method=averaging_method,
            variance_method=variance_method,
            number_of_deviations=number_of_deviations,
            manual_threshold=manual_threshold,
            smooth_threshold_application=smooth_threshold_application,
        ),
        proven_unit_interval_scale=proven_unit_interval_scale,
        log_profile_function=log_profile_function,
    ).threshold_tuple(
        global_threshold_function=global_threshold_function,
        adaptive_threshold_function=adaptive_threshold_function,
        apply_threshold_function=apply_threshold_function,
    )


def cellprofiler_threshold_diagnostics(
    image: np.ndarray,
    binary: np.ndarray,
    *,
    final_threshold: float,
    original_threshold: float,
    mask: np.ndarray | None = None,
    proven_unit_interval_scale: int | None = None,
    log_profile_function: ThresholdProfileSink | None = None,
) -> CellProfilerThresholdDiagnostics:
    """Return CellProfiler's image-level threshold quality measurements."""
    log_profile = CellProfilerThresholdProfiler.from_sink(log_profile_function).sink
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
    (
        weighted_variance,
        sum_of_entropies,
    ) = ThresholdDiagnosticsBackendStrategy.for_memory_type().diagnostics(
        image,
        measurement_mask,
        binary_image,
        proven_unit_interval_scale=proven_unit_interval_scale,
    )
    log_profile(
        "threshold_diagnostics_backend",
        time.perf_counter() - phase_started_at,
        function="cellprofiler_threshold_diagnostics",
        shape=tuple(np.shape(image)),
        proven_unit_interval_scale=proven_unit_interval_scale,
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
class ThresholdMeasurementFeatureRecord(MeasurementFeatureRecord):
    """Record that can expose its threshold-diagnostic measurement fields."""

    def threshold_measurement_record(self) -> MeasurementFeatureRecord:
        """Return the row whose fields are emitted as Threshold measurements."""
        return self


@dataclass
class ThresholdResult(ThresholdMeasurementFeatureRecord):
    """Threshold measurement row emitted by the CP-compatible Threshold module."""

    slice_index: Annotated[int, MeasurementRowAxisField.SLICE_INDEX]
    final_threshold: float
    original_threshold: float
    guide_threshold: float
    sigma: float
    weighted_variance: float = 0.0
    sum_of_entropies: float = 0.0

    def threshold_measurement_record(self) -> MeasurementFeatureRecord:
        return ObjectThresholdResult(
            slice_index=self.slice_index,
            final_threshold=self.final_threshold,
            original_threshold=self.original_threshold,
            weighted_variance=self.weighted_variance,
            sum_of_entropies=self.sum_of_entropies,
        )


@dataclass
class ObjectThresholdResult(ThresholdMeasurementFeatureRecord):
    """Threshold measurement row emitted by object-producing threshold modules."""

    slice_index: Annotated[int, MeasurementRowAxisField.SLICE_INDEX]
    final_threshold: float
    original_threshold: float = 0.0
    weighted_variance: float = 0.0
    sum_of_entropies: float = 0.0


@runtime_image_execution_mode(ImagePayloadExecutionMode.FULL_STACK)
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
    source_payload = normalize_cellprofiler_image_payload(
        image,
        dtype=np.float32,
        allow_unproven_uint8_float_domain=True,
    )
    image = np.asarray(image_payload_data(source_payload), dtype=np.float32)
    metadata = image_payload_metadata(source_payload)
    projected_mask = image_mask_for_data_domain(
        explicit_mask=mask, source_payload=source_payload, data=image
    )
    mask = None if projected_mask is None else np.asarray(projected_mask, dtype=bool)
    guide_threshold = 0.0
    proven_unit_interval_scale = unit_interval_scale_for_threshold_selection(
        image, metadata
    )
    settings = CellProfilerThresholdSettings(
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
        manual_threshold=0.0,
        smooth_threshold_application=True,
    ).with_threshold_module_controls(
        automatic=automatic, predefined_threshold=predefined_threshold
    )
    threshold_result = CellProfilerThresholdRequest(
        image=image,
        image_mask=mask,
        settings=settings,
        proven_unit_interval_scale=proven_unit_interval_scale,
        log_profile_function=threshold_profile_sink(),
    ).calculate()
    output_image = RuntimeImagePayloadContext(
        threshold_result.mask.astype(np.float32),
        mask=mask,
        metadata=image_payload_metadata(
            source_payload
        ).without_unit_interval_intensity_scale(),
    ).payload()
    return (
        output_image,
        ThresholdResult(
            slice_index=0,
            final_threshold=float(threshold_result.final_threshold),
            original_threshold=float(threshold_result.original_threshold),
            guide_threshold=guide_threshold,
            sigma=float(settings.normalized().threshold_smoothing_scale),
            weighted_variance=threshold_result.weighted_variance,
            sum_of_entropies=threshold_result.sum_of_entropies,
        ),
    )


def prepare_threshold() -> None:
    """Warm threshold backend strategy families before timed execution."""
    ThresholdSmoothingBackendStrategy.prepare_registered_family()
    ThresholdDiagnosticsBackendStrategy.prepare_registered_family()
    ThresholdPrimitiveBackendStrategy.prepare_registered_family()


threshold.__openhcs_prepare__ = prepare_threshold


@njit(cache=True)
def _quantized_counts_unmasked_numba(image: np.ndarray, scale: int) -> np.ndarray:
    counts = np.zeros(scale + 1, dtype=np.int64)
    flat_image = image.ravel()
    scale_float = float(scale)
    for index in range(flat_image.size):
        value = float(flat_image[index])
        if not np.isfinite(value):
            continue
        code = int(np.rint(value * scale_float))
        if code < 0:
            code = 0
        elif code > scale:
            code = scale
        counts[code] += 1
    return counts


@njit(cache=True)
def _quantized_counts_masked_numba(
    image: np.ndarray, mask: np.ndarray, scale: int
) -> np.ndarray:
    counts = np.zeros(scale + 1, dtype=np.int64)
    flat_image = image.ravel()
    flat_mask = mask.ravel()
    scale_float = float(scale)
    for index in range(flat_image.size):
        if not flat_mask[index]:
            continue
        value = float(flat_image[index])
        if not np.isfinite(value):
            continue
        code = int(np.rint(value * scale_float))
        if code < 0:
            code = 0
        elif code > scale:
            code = scale
        counts[code] += 1
    return counts


def _li_threshold_quantized_numpy(
    image: np.ndarray, mask: np.ndarray | None, scale: int
) -> float:
    """Return dense Li semantics from a proven unit-interval quantized domain."""
    if scale <= 1:
        raise ValueError(
            f"Unit-interval scale must be greater than one, got {scale!r}."
        )
    image_array = np.asarray(image, dtype=np.float32)
    mask_array = threshold_mask_for_image_domain(
        mask, image_array.shape, context="Minimum cross-entropy threshold"
    )
    counts = (
        _quantized_counts_unmasked_numba(np.ascontiguousarray(image_array), scale)
        if mask_array is None
        else _quantized_counts_masked_numba(
            np.ascontiguousarray(image_array), np.ascontiguousarray(mask_array), scale
        )
    )
    return _li_threshold_from_quantized_counts_numpy(counts, scale)


def _li_threshold_from_quantized_counts_numpy(counts: np.ndarray, scale: int) -> float:
    active_codes = np.flatnonzero(counts)
    if active_codes.size == 0:
        return 0.0
    values = (active_codes.astype(np.float32) / np.float32(scale)).astype(np.float32)
    if active_codes.size == 1:
        return float(values[0])
    image_min = values[0]
    shifted_values = (values - image_min).astype(np.float32)
    positive_diffs = np.diff(values.astype(np.float64))
    positive_diffs = positive_diffs[positive_diffs > 0]
    tolerance = max(float(np.min(positive_diffs) / 2.0), CELLPROFILER_LI_TOLERANCE)
    active_counts = counts[active_codes]
    total_count = int(np.sum(active_counts))
    threshold_next = np.float32(
        np.sum(shifted_values * active_counts.astype(np.float32), dtype=np.float32)
        / np.float32(total_count)
    )
    threshold_current = np.float32(-2.0 * tolerance)
    iterations = 0
    while (
        abs(float(threshold_next) - float(threshold_current)) > tolerance
        and iterations < 1000
    ):
        threshold_current = threshold_next
        foreground = shifted_values > threshold_current
        foreground_count = int(np.sum(active_counts[foreground]))
        background_count = total_count - foreground_count
        if foreground_count == 0 or background_count == 0:
            break
        foreground_mean = np.float32(
            np.sum(
                shifted_values[foreground]
                * active_counts[foreground].astype(np.float32),
                dtype=np.float32,
            )
            / np.float32(foreground_count)
        )
        background_mean = np.float32(
            np.sum(
                shifted_values[~foreground]
                * active_counts[~foreground].astype(np.float32),
                dtype=np.float32,
            )
            / np.float32(background_count)
        )
        if background_mean == 0.0:
            break
        threshold_next = np.float32(
            (background_mean - foreground_mean)
            / (np.log(background_mean) - np.log(foreground_mean))
        )
        iterations += 1
    return float(np.float32(threshold_next + image_min))


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
    while abs(threshold_next - threshold_current) > tolerance and iterations < 1000:
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
    image: np.ndarray, mask: np.ndarray, binary_image: np.ndarray
) -> float:
    image_array = np.asarray(image)
    mask_array = np.asarray(mask, dtype=bool)
    if not np.any(mask_array):
        return 0.0
    minval = float(np.max(image_array[mask_array]) / 256)
    if minval == 0:
        return 0.0
    fg = np.log2(np.maximum(image_array[binary_image & mask_array], minval))
    bg = np.log2(np.maximum(image_array[~binary_image & mask_array], minval))
    nfg = fg.size
    nbg = bg.size
    if nfg == 0:
        return float(np.var(bg))
    if nbg == 0:
        return float(np.var(fg))
    return float((np.var(fg) * nfg + np.var(bg) * nbg) / (nfg + nbg))


def _numpy_threshold_sum_of_entropies(
    image: np.ndarray, mask: np.ndarray, binary_image: np.ndarray
) -> float:
    image_array = np.asarray(image)
    mask_array = np.asarray(mask, dtype=bool).copy()
    binary_array = np.asarray(binary_image, dtype=bool)
    if (
        mask_array.shape == image_array.shape
        and bool(np.all(mask_array))
        and bool(np.all(np.isfinite(image_array)))
    ):
        return _numpy_threshold_sum_of_entropies_full_finite(
            image_array,
            binary_array,
        )
    mask_array[np.isnan(image_array)] = False
    if not np.any(mask_array):
        return 0.0
    minval = float(np.max(image_array[mask_array]) / 256)
    if minval == 0:
        return 0.0
    clamped_image = image_array.copy()
    clamped_image[clamped_image < minval] = minval
    smoothed_image = smooth_with_deterministic_noise(clamped_image, bits=8)
    im_min = np.min(smoothed_image)
    im_max = np.max(smoothed_image)
    upper = np.log2(im_max)
    lower = np.log2(im_min)
    if upper == lower:
        return float(math.log(np.sum(mask_array), 2))
    fg = smoothed_image[binary_image & mask_array]
    bg = smoothed_image[~binary_image & mask_array]
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


def _numpy_threshold_sum_of_entropies_full_finite(
    image: np.ndarray, binary_image: np.ndarray
) -> float:
    """Return exact CellProfiler entropy for a full finite image domain."""
    minval = float(np.max(image) / 256)
    if minval == 0:
        return 0.0
    smoothed_image = smooth_with_deterministic_noise(
        np.maximum(image, minval),
        bits=8,
    )
    im_min = np.min(smoothed_image)
    im_max = np.max(smoothed_image)
    upper = np.log2(im_max)
    lower = np.log2(im_min)
    if upper == lower:
        return float(math.log(image.size, 2))
    fg = smoothed_image[binary_image]
    bg = smoothed_image[~binary_image]
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


def _threshold_setting_token(value: Any) -> str:
    """Return a stable comparison token for threshold setting values."""
    if isinstance(value, Enum) and isinstance(value.value, str):
        value = value.value
    return " ".join(str(value).strip().lower().replace("-", " ").split())


class ThresholdSettingScope(Enum):
    """Serialized threshold strategy names used to select active method rows."""

    GLOBAL = "global"
    ADAPTIVE = "adaptive"

    @classmethod
    def from_module(cls, module: "ModuleBlock") -> "ThresholdSettingScope | None":
        value = LastRepeatedSettingValuePolicy().value(module, "Threshold strategy")
        token = _threshold_setting_token(value or "")
        for scope in cls:
            if token == scope.value:
                return scope
        return None


class ThresholdMethodRowSelectionPolicy(
    EnumKeyedStrategyMixin[ThresholdSettingScope], ABC, metaclass=AutoRegisterMeta
):
    """Select the active threshold-method row for one threshold scope."""

    __registry_key__ = "scope_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "scope"
    __enum_label_attr__ = "scope_label"
    scope: ClassVar[ThresholdSettingScope | None] = None
    scope_label: ClassVar[str | None] = None

    @abstractmethod
    def selected_value(self, values: tuple[str, ...]) -> str:
        """Return the CellProfiler threshold-method value active for this scope."""


class GlobalThresholdMethodRowSelectionPolicy(ThresholdMethodRowSelectionPolicy):
    """Global thresholding uses the first method row in upgraded CP settings."""

    scope = ThresholdSettingScope.GLOBAL

    def selected_value(self, values: tuple[str, ...]) -> str:
        return values[0]


class AdaptiveThresholdMethodRowSelectionPolicy(ThresholdMethodRowSelectionPolicy):
    """Adaptive thresholding uses the local-method row in upgraded CP settings."""

    scope = ThresholdSettingScope.ADAPTIVE

    def selected_value(self, values: tuple[str, ...]) -> str:
        return values[-1]


class ThresholdMethodRepeatedSettingValuePolicy(RepeatedSettingValuePolicy):
    """Resolve CP's global/local threshold method rows from threshold scope."""

    setting_name = "Thresholding method"

    def _resolve_repeated_value(
        self,
        module: "ModuleBlock",
        setting_name: str | "SettingNameFamily",
        values: tuple[str, ...],
    ) -> str:
        scope = ThresholdSettingScope.from_module(module)
        if scope is not None:
            return ThresholdMethodRowSelectionPolicy.for_enum_member(
                scope
            ).selected_value(values)
        raise ValueError(
            f"{module.name}({module.module_num}) has repeated {setting_name!r} rows but no supported threshold strategy."
        )


class LegacyCellProfilerThresholdVersionAuthority(ABC):
    """Nominal authority for CellProfiler threshold-setting schema versions."""

    setting_name: ClassVar[str] = "Threshold setting version"

    @classmethod
    def version_for(cls, module: "ModuleBlock") -> int | None:
        value = CellProfilerModule.setting_value(module, cls.setting_name)
        if value is None:
            return None
        return MeasurementScalarLiteral(value).integer_value

    @classmethod
    def is_legacy_v10_or_older(cls, module: "ModuleBlock") -> bool:
        version = cls.version_for(module)
        return version is not None and version <= 10


class ThresholdSettingsModule(CellProfilerModule):
    """Module parent for declarations that consume CellProfiler threshold rows."""

    include_threshold_advanced_setting: ClassVar[bool] = False
    threshold_settings: ClassVar[Mapping[str, str]] = {
        "Threshold strategy": "threshold_scope",
        "Thresholding method": "threshold_method",
        "Threshold smoothing scale": "threshold_smoothing_scale",
        "Threshold correction factor": "threshold_correction_factor",
        "Two-class or three-class thresholding?": "otsu_class_count",
        "Assign pixels in the middle intensity class to the foreground or the background?": "assign_middle_to_foreground",
        "Log transform before thresholding?": "log_transform",
        "Size of adaptive window": "adaptive_window_size",
        "Lower outlier fraction": "lower_outlier_fraction",
        "Upper outlier fraction": "upper_outlier_fraction",
        "Averaging method": "averaging_method",
        "Variance method": "variance_method",
        "# of deviations": "number_of_deviations",
        "Manual threshold": "manual_threshold",
    }
    ignored_threshold_settings: ClassVar[tuple[str, ...]] = (
        "Threshold setting version",
        "Select the measurement to threshold with",
    )
    float_threshold_settings: ClassVar[frozenset[str]] = frozenset(
        {
            "Threshold smoothing scale",
            "Threshold correction factor",
            "Lower outlier fraction",
            "Upper outlier fraction",
            "# of deviations",
            "Manual threshold",
        }
    )
    int_threshold_settings: ClassVar[frozenset[str]] = frozenset(
        {"Size of adaptive window"}
    )
    bool_threshold_settings: ClassVar[frozenset[str]] = frozenset(
        {"Log transform before thresholding?"}
    )
    legacy_threshold_method_names: ClassVar[Mapping[str, str]] = {
        "robustbackground": "Robust Background",
        "minimum cross entropy": "Minimum Cross-Entropy",
    }
    otsu_method_token: ClassVar[str] = "otsu"
    three_class_otsu_token: ClassVar[str] = "three classes"

    @classmethod
    def bind_settings(
        cls,
        module: "ModuleBlock",
        *,
        binder: "SettingsBinder",
        param_mapping: Mapping[str, Any],
        ignored_unmapped_settings: frozenset[str] = frozenset(),
    ) -> "BoundModuleSettings":
        if cls.setting_bindings:
            bound = cls._bind_declared_settings(
                module, binder=binder, param_mapping=param_mapping
            )
        else:
            bound = cls._bind_generic_settings(
                module, binder=binder, param_mapping=param_mapping
            )
        kwargs = dict(bound.kwargs)
        unmapped_kwargs = dict(bound.unmapped_kwargs)
        cls.bind_threshold_settings(module, binder, kwargs, unmapped_kwargs)
        bound = BoundModuleSettings(
            kwargs, unmapped_kwargs, bound.invocation_options, bound.setting_coverage
        )
        return cls._finalize_bound_settings(
            module,
            binder=binder,
            bound=cls.postprocess_bound_settings(module, bound),
            ignored_unmapped_settings=ignored_unmapped_settings,
        )

    @classmethod
    def bind_threshold_settings(
        cls,
        module: "ModuleBlock",
        binder: "SettingsBinder",
        kwargs: dict[str, Any],
        unmapped_kwargs: dict[str, Any],
    ) -> None:
        if cls.include_threshold_advanced_setting:
            cls._bind_optional_repeated_threshold_setting(
                module,
                binder,
                "Use advanced settings?",
                "use_advanced_settings",
                kwargs,
                unmapped_kwargs,
                LastRepeatedSettingValuePolicy(),
            )
        for setting_name, parameter_name in cls.threshold_settings.items():
            value = RepeatedSettingValuePolicy.for_setting(setting_name).value(
                module, setting_name
            )
            if value is not None:
                kwargs[parameter_name] = cls.parse_threshold_setting(
                    binder, setting_name, value
                )
            cls.consume_setting(unmapped_kwargs, setting_name)
        cls.upgrade_legacy_threshold_kwargs(module, kwargs)
        cls.bind_threshold_bounds(module, binder, kwargs, unmapped_kwargs)
        for setting_name in cls.ignored_threshold_settings:
            cls.consume_setting(unmapped_kwargs, setting_name)

    @classmethod
    def _bind_optional_repeated_threshold_setting(
        cls,
        module: "ModuleBlock",
        binder: "SettingsBinder",
        setting_name: str,
        parameter_name: str,
        kwargs: dict[str, Any],
        unmapped_kwargs: dict[str, Any],
        policy: RepeatedSettingValuePolicy,
    ) -> None:
        value = policy.value(module, setting_name)
        if value is not None:
            kwargs[parameter_name] = binder.parse_value(setting_name, value)
        cls.consume_setting(unmapped_kwargs, setting_name)

    @classmethod
    def bind_threshold_bounds(
        cls,
        module: "ModuleBlock",
        binder: "SettingsBinder",
        kwargs: dict[str, Any],
        unmapped_kwargs: dict[str, Any],
    ) -> None:
        setting_name = "Lower and upper bounds on threshold"
        bounds = LastRepeatedSettingValuePolicy().value(module, setting_name)
        if bounds is not None:
            parsed_bounds = binder.parse_value(setting_name, bounds)
            if not isinstance(parsed_bounds, tuple) or len(parsed_bounds) != 2:
                raise ValueError(
                    f"{module.name} threshold bounds must contain two values, got {bounds!r}."
                )
            kwargs["threshold_min"] = parsed_bounds[0]
            kwargs["threshold_max"] = parsed_bounds[1]
        cls.consume_setting(unmapped_kwargs, setting_name)

    @classmethod
    def parse_threshold_setting(
        cls, binder: "SettingsBinder", setting_name: str, value: str
    ) -> Any:
        """Parse threshold settings by semantic field, not generic literal shape."""
        if setting_name in cls.float_threshold_settings:
            return parse_cellprofiler_float(value)
        if setting_name in cls.int_threshold_settings:
            return parse_cellprofiler_int(value)
        if setting_name in cls.bool_threshold_settings:
            return parse_cellprofiler_bool(value)
        return binder.parse_value(setting_name, value)

    @classmethod
    def upgrade_legacy_threshold_kwargs(
        cls, module: "ModuleBlock", kwargs: dict[str, Any]
    ) -> None:
        if not LegacyCellProfilerThresholdVersionAuthority.is_legacy_v10_or_older(
            module
        ):
            return
        threshold_method = kwargs.get("threshold_method")
        if threshold_method is not None:
            method_token = _threshold_setting_token(threshold_method)
            kwargs["threshold_method"] = cls.legacy_threshold_method_names.get(
                method_token, threshold_method
            )
        if "log_transform" not in kwargs:
            log_transform_default = cls.legacy_log_transform_default(module, kwargs)
            if log_transform_default is not None:
                kwargs["log_transform"] = log_transform_default

    @classmethod
    def legacy_log_transform_default(
        cls, module: "ModuleBlock", kwargs: Mapping[str, Any]
    ) -> bool | None:
        if not LegacyCellProfilerThresholdVersionAuthority.is_legacy_v10_or_older(
            module
        ):
            return None
        threshold_method = _threshold_setting_token(kwargs.get("threshold_method", ""))
        otsu_class_count = _threshold_setting_token(kwargs.get("otsu_class_count", ""))
        return (
            threshold_method == cls.otsu_method_token
            and otsu_class_count == cls.three_class_otsu_token
        )

    @staticmethod
    def consume_setting(unmapped_kwargs: dict[str, Any], setting_name: str) -> None:
        unmapped_kwargs.pop(
            CellProfilerModule.normalize_setting_name(setting_name), None
        )


class ThresholdMeasurementRecordRowsMixin(FieldDerivedMeasurementFeatureModule):
    """Declares threshold result rows on threshold-producing module MROs."""

    measurement_feature_family = "Threshold"
    measurement_feature_token_aliases = (("original", "Orig"),)

    @classmethod
    def measurement_feature_name(
        cls,
        field_name: str,
        *qualified_parts: object,
    ) -> str:
        parts = (
            cls.measurement_feature_stem(field_name),
            *(str(part) for part in qualified_parts if part not in (None, "")),
        )
        return "_".join(parts)

    @dataclass(frozen=True, slots=True)
    class MeasurementRows(ModuleOwnedResultMeasurementRows):
        """Project absorbed threshold stats into CP measurement rows."""

        registry_key = "threshold"
        object_name: str

        @classmethod
        def for_request(cls, module_type, request):
            return cls(
                request.output_value,
                module_type=module_type,
                object_name=module_type.threshold_measurement_object_name(request),
            )

        def rows(self) -> list[CellProfilerKwargs]:
            records: list[MeasurementFeatureRecord] = []
            for source_record in self.source_rows():
                if not isinstance(source_record, ThresholdMeasurementFeatureRecord):
                    raise TypeError(
                        "Threshold measurement rows must be emitted as "
                        "ThresholdMeasurementFeatureRecord dataclasses."
                    )
                records.append(source_record.threshold_measurement_record())
            return self.module_type.measurement_feature_rows_from_records(
                tuple(records),
                qualified_parts=(self.object_name,),
            )

    @classmethod
    def threshold_measurement_object_name(cls, request) -> str:
        raise NotImplementedError(
            f"{cls.__name__} must declare threshold_measurement_object_name()."
        )


class OutputObjectThresholdMeasurementRecordRowsMixin(
    ThresholdMeasurementRecordRowsMixin
):
    """Qualify threshold rows by the emitted object artifact."""

    @classmethod
    def threshold_measurement_object_name(cls, request) -> str:
        return request.single_output_object_name()


class ProducedImageThresholdMeasurementRecordRowsMixin(
    ThresholdMeasurementRecordRowsMixin
):
    """Qualify threshold rows by the emitted image artifact."""

    @classmethod
    def threshold_measurement_object_name(cls, request) -> str:
        return cls.primary_image_measurement_source(
            request
        ).require_produced_artifact().artifact_spec.name


class ThresholdExecutionDomainContract(SourceVolumetricPixelDataExecutionContract):
    contract_key = "Threshold.execution_domain"
    source_filename = "threshold.py"
    callable_name = "threshold"

    @property
    def absorbed_callable(self) -> Callable[..., Any]:
        return threshold


class ThresholdModule(
    ProducedImageThresholdMeasurementRecordRowsMixin,
    NoObjectNameMeasurementRecordMixin,
    ProducedImagePayloadMeasurementRecordMixin,
    NoFieldsMeasurementRecordMixin,
    VolumetricInputExecutionModePolicy,
    ImageArtifactInputModule,
    ImageArtifactOutputModule,
    MeasurementArtifactOutputModule,
    ThresholdSettingsModule,
):
    module_name = "Threshold"
    function_name = "threshold"
    validated = True
    confidence = 1.0
    measurement_category_prefixes = (("threshold",),)
    measurement_feature_part_rewrites = {
        ("otsu",): ("threshold", "otsu"),
    }
    semantic_default_contract_types = (ThresholdExecutionDomainContract,)
    ignored_settings = ("Select the input image", "Name the output image")
    image_input_settings = ("Select the input image",)
    image_output_settings = ("Name the output image",)
    threshold_parameter_aliases = {
        "threshold_smoothing_scale": "smoothing",
        "adaptive_window_size": "window_size",
    }

    @classmethod
    def artifact_contract_outputs(
        cls, builder: "_SymbolTableBuilder", module: "ModuleBlock"
    ) -> tuple[object, ...]:
        return (
            *cls.declared_output_artifacts_from_settings(builder, module),
            cls.measurement_output_artifact(builder, module),
        )

    @classmethod
    def postprocess_bound_settings(
        cls, module: "ModuleBlock", bound: BoundModuleSettings
    ) -> BoundModuleSettings:
        kwargs = dict(bound.kwargs)
        for source_name, target_name in cls.threshold_parameter_aliases.items():
            if source_name in kwargs:
                kwargs[target_name] = kwargs.pop(source_name)
        manual_threshold = kwargs.pop("manual_threshold", None)
        if (
            manual_threshold is not None
            and normalize_cellprofiler_setting_name(kwargs.get("threshold_method", ""))
            == "manual"
        ):
            kwargs["predefined_threshold"] = manual_threshold
        return BoundModuleSettings(
            kwargs,
            bound.unmapped_kwargs,
            bound.invocation_options,
            bound.setting_coverage,
        )


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
    CellProfilerThresholdRequest,
    CellProfilerThresholdResult,
    CellProfilerThresholdSettings,
    CellProfilerThresholdScope,
    CellProfilerVarianceMethod,
    ThresholdDiagnosticsBackendStrategy,
    ThresholdModule,
    "ThresholdMethod",
    ThresholdPrimitiveBackendStrategy,
    ThresholdResult,
    "ThresholdScope",
    ThresholdSmoothingBackendStrategy,
    "VarianceMethod",
    cellprofiler_threshold,
    cellprofiler_threshold_diagnostics,
    prepare_threshold,
    threshold,
    threshold_primitives,
)
