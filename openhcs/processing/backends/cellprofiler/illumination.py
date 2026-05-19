"""Illumination backends for CellProfiler-compatible processing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import logging
import time
from typing import ClassVar
import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit

from openhcs.constants.constants import MemoryType
from openhcs.core.callable_contract import processing_prepare
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.runtime_values import (
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    image_payload_with_context,
    project_image_mask_to_data_domain,
)
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    cellprofiler_backend_key,
)
from openhcs.processing.backends.cellprofiler.morphology import MorphologyBackendStrategy
from openhcs.processing.backends.cellprofiler.granularity import (
    CellProfilerRuntimeProfiler,
)
from openhcs.processing.backends.cellprofiler.smoothing import MaskedLinearFilterRequest
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.materialization import csv_materializer

NDIMAGE_CONSTANT_MODE = "constant"
ROBUST_FACTOR = 0.02
logger = logging.getLogger(__name__)
runtime_profiler = CellProfilerRuntimeProfiler(logger)


@dataclass(frozen=True, slots=True)
class IlluminationMask:
    """Mask aligned with CellProfiler illumination input and stack slices."""

    mask: object | None
    pixel_data: np.ndarray

    @property
    def normalized(self) -> np.ndarray | None:
        if self.mask is None:
            return None
        mask_array = np.asarray(self.mask, dtype=bool)
        return IlluminationMaskNormalizationStrategy.for_request(
            IlluminationMaskNormalizationRequest(
                mask_array=mask_array,
                pixel_data=self.pixel_data,
            )
        ).normalize(mask_array)

    def for_stack_slice(self, slice_index: int) -> np.ndarray | None:
        mask = self.normalized
        if mask is None:
            return None
        if mask.ndim >= 3 and slice_index < mask.shape[0]:
            return np.asarray(mask[slice_index], dtype=bool)
        return mask

    def for_output(self, illumination: np.ndarray) -> np.ndarray | None:
        mask = self.normalized
        if mask is None:
            return None
        if mask.shape == illumination.shape:
            return mask
        if illumination.ndim == 2 and mask.ndim >= 3:
            return np.any(mask, axis=0)
        return mask


@dataclass(frozen=True, slots=True)
class IlluminationMaskNormalizationRequest:
    """Shape facts needed to normalize an illumination mask."""

    mask_array: np.ndarray
    pixel_data: np.ndarray


class IlluminationMaskNormalizationStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal matcher for closed illumination mask shape normalization cases."""

    __registry_key__ = "registry_key"
    __skip_if_no_key__ = True

    registry_key: ClassVar[str | None] = None
    priority: ClassVar[int] = 100

    @classmethod
    def strategies(cls) -> tuple["IlluminationMaskNormalizationStrategy", ...]:
        return tuple(
            strategy_type()
            for strategy_type in sorted(
                cls.__registry__.values(),
                key=lambda candidate: candidate.priority,
            )
        )

    @classmethod
    def for_request(
        cls,
        request: IlluminationMaskNormalizationRequest,
    ) -> "IlluminationMaskNormalizationStrategy":
        for strategy in cls.strategies():
            if strategy.matches(request):
                return strategy
        return IlluminationMaskAsProvidedNormalizationStrategy()

    @abstractmethod
    def matches(self, request: IlluminationMaskNormalizationRequest) -> bool:
        """Return whether this strategy owns the request shape."""

    @abstractmethod
    def normalize(self, mask_array: np.ndarray) -> np.ndarray:
        """Return the normalized mask."""


class ExactIlluminationMaskNormalizationStrategy(IlluminationMaskNormalizationStrategy):
    """Mask already matches the illumination pixel data."""

    registry_key = "exact"
    priority = 0

    def matches(self, request: IlluminationMaskNormalizationRequest) -> bool:
        return request.mask_array.shape == request.pixel_data.shape

    def normalize(self, mask_array: np.ndarray) -> np.ndarray:
        return mask_array


class SpatialIlluminationMaskNormalizationStrategy(IlluminationMaskNormalizationStrategy):
    """Mask matches the trailing spatial domain of a stack-like image."""

    registry_key = "spatial"
    priority = 10

    def matches(self, request: IlluminationMaskNormalizationRequest) -> bool:
        return request.mask_array.shape == request.pixel_data.shape[-request.mask_array.ndim :]

    def normalize(self, mask_array: np.ndarray) -> np.ndarray:
        return mask_array


class SingletonPlaneIlluminationMaskNormalizationStrategy(IlluminationMaskNormalizationStrategy):
    """Single-plane stack mask used for a two-dimensional illumination image."""

    registry_key = "singleton_plane"
    priority = 20

    def matches(self, request: IlluminationMaskNormalizationRequest) -> bool:
        return (
            request.pixel_data.ndim == 2
            and request.mask_array.ndim == 3
            and request.mask_array.shape[0] == 1
        )

    def normalize(self, mask_array: np.ndarray) -> np.ndarray:
        return mask_array[0]


class IlluminationMaskAsProvidedNormalizationStrategy(IlluminationMaskNormalizationStrategy):
    """Fallback for legacy CellProfiler-compatible mask broadcasting behavior."""

    registry_key = "as_provided"
    priority = 30

    def matches(self, request: IlluminationMaskNormalizationRequest) -> bool:
        return True

    def normalize(self, mask_array: np.ndarray) -> np.ndarray:
        return mask_array


@dataclass(frozen=True, slots=True)
class IlluminationGaussianFilter:
    """Gaussian filtering semantics for illumination smoothing and dilation."""

    pixel_data: np.ndarray
    mask: np.ndarray | None
    sigma: float

    def apply(self) -> np.ndarray:
        from scipy.ndimage import gaussian_filter

        if self.mask is None:
            return gaussian_filter(
                self.pixel_data,
                self.sigma,
                mode=NDIMAGE_CONSTANT_MODE,
                cval=0,
            )
        return MaskedLinearFilterRequest(
            pixels=self.pixel_data,
            mask=self.mask,
            operation=lambda image: gaussian_filter(
                image,
                self.sigma,
                mode=NDIMAGE_CONSTANT_MODE,
                cval=0,
            ),
        ).apply()


class IntensityChoice(Enum):
    """CellProfiler CorrectIlluminationCalculate intensity source."""

    REGULAR = "regular"
    BACKGROUND = "background"


class SmoothingMethod(Enum):
    """CellProfiler CorrectIlluminationCalculate smoothing method."""

    NONE = "none"
    CONVEX_HULL = "convex_hull"
    FIT_POLYNOMIAL = "fit_polynomial"
    MEDIAN_FILTER = "median_filter"
    GAUSSIAN_FILTER = "gaussian_filter"
    TO_AVERAGE = "to_average"
    SPLINES = "splines"


class FilterSizeMethod(Enum):
    """CellProfiler CorrectIlluminationCalculate filter-size mode."""

    AUTOMATIC = "automatic"
    OBJECT_SIZE = "object_size"
    MANUALLY = "manually"


class RescaleOption(Enum):
    """CellProfiler CorrectIlluminationCalculate output rescale mode."""

    YES = "yes"
    NO = "no"
    MEDIAN = "median"


class IlluminationCorrectionMethod(Enum):
    """CellProfiler CorrectIlluminationApply arithmetic mode."""

    DIVIDE = "divide"
    SUBTRACT = "subtract"


class SplineBgMode(Enum):
    """CellProfiler CorrectIlluminationCalculate spline background mode."""

    AUTO = "auto"
    DARK = "dark"
    BRIGHT = "bright"
    GRAY = "gray"


class CalculationScope(Enum):
    """CellProfiler CorrectIlluminationCalculate image aggregation scope."""

    EACH = "each"
    ALL_FIRST_CYCLE = "all_first_cycle"
    ALL_ACROSS_CYCLES = "all_across_cycles"

    @property
    def uses_all_images(self) -> bool:
        return self is not CalculationScope.EACH


def coerce_illumination_enum(enum_type: type[Enum], value: object) -> Enum:
    """Coerce CellProfiler UI literals for illumination-owned enums."""
    return coerce_cellprofiler_enum(enum_type, value)


@dataclass
class IlluminationStats:
    """Runtime measurements emitted by CorrectIlluminationCalculate."""

    slice_index: int
    min_value: float
    max_value: float
    mean_value: float
    calculation_type: str
    smoothing_method: str


@dataclass(frozen=True, slots=True)
class IlluminationCorrectionRequest:
    """One image/function pair for CorrectIlluminationApply."""

    image_pixels: np.ndarray
    illumination_function: np.ndarray


class IlluminationCorrectionStrategy(
    EnumKeyedStrategyMixin[IlluminationCorrectionMethod],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal correction implementation for one CellProfiler method."""

    __registry_key__ = "strategy_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "method"

    method: ClassVar[IlluminationCorrectionMethod]
    strategy_label: ClassVar[str | None] = None

    @classmethod
    def for_method(
        cls,
        method: IlluminationCorrectionMethod,
    ) -> "IlluminationCorrectionStrategy":
        return cls.for_enum_member(method)

    @abstractmethod
    def apply(self, request: IlluminationCorrectionRequest) -> np.ndarray:
        """Apply the correction method."""


class DivideIlluminationCorrectionStrategy(IlluminationCorrectionStrategy):
    method = IlluminationCorrectionMethod.DIVIDE

    def apply(self, request: IlluminationCorrectionRequest) -> np.ndarray:
        output_dtype = np.result_type(
            request.image_pixels,
            request.illumination_function,
            1e-10,
        )
        output = np.empty(request.image_pixels.shape, dtype=output_dtype)
        nonzero = request.illumination_function != 0
        np.divide(
            request.image_pixels,
            request.illumination_function,
            out=output,
            where=nonzero,
        )
        if not np.all(nonzero):
            np.divide(
                request.image_pixels,
                output_dtype.type(1e-10),
                out=output,
                where=~nonzero,
            )
        return output


class SubtractIlluminationCorrectionStrategy(IlluminationCorrectionStrategy):
    method = IlluminationCorrectionMethod.SUBTRACT

    def apply(self, request: IlluminationCorrectionRequest) -> np.ndarray:
        output = np.empty(
            request.image_pixels.shape,
            dtype=np.result_type(
                request.image_pixels,
                request.illumination_function,
                0.0,
            ),
        )
        np.subtract(
            request.image_pixels,
            request.illumination_function,
            out=output,
        )
        return output


@dataclass(frozen=True, slots=True)
class IlluminationCorrectionSettingSequence:
    """Settings expanded to match every image/function pair."""

    methods: tuple[IlluminationCorrectionMethod, ...]
    truncate_low: tuple[bool, ...]
    truncate_high: tuple[bool, ...]

    @classmethod
    def from_settings(
        cls,
        method: (
            IlluminationCorrectionMethod
            | str
            | tuple[IlluminationCorrectionMethod | str, ...]
        ),
        truncate_low: bool | tuple[bool, ...],
        truncate_high: bool | tuple[bool, ...],
        pair_count: int,
    ) -> "IlluminationCorrectionSettingSequence":
        return cls(
            methods=cls.methods_for_pair_count(method, pair_count),
            truncate_low=cls.bool_setting_for_pair_count(
                truncate_low,
                pair_count,
                parameter_name="truncate_low",
            ),
            truncate_high=cls.bool_setting_for_pair_count(
                truncate_high,
                pair_count,
                parameter_name="truncate_high",
            ),
        )

    @staticmethod
    def methods_for_pair_count(
        value: (
            IlluminationCorrectionMethod
            | str
            | tuple[IlluminationCorrectionMethod | str, ...]
        ),
        pair_count: int,
    ) -> tuple[IlluminationCorrectionMethod, ...]:
        if isinstance(value, tuple):
            if len(value) != pair_count:
                raise ValueError(
                    "CorrectIlluminationApply method count must match "
                    f"image/function pair count; got {len(value)} methods for "
                    f"{pair_count} pairs."
                )
            return tuple(
                coerce_cellprofiler_enum(IlluminationCorrectionMethod, method)
                for method in value
            )
        method = coerce_cellprofiler_enum(IlluminationCorrectionMethod, value)
        return (method,) * pair_count

    @staticmethod
    def bool_setting_for_pair_count(
        value: bool | tuple[bool, ...],
        pair_count: int,
        *,
        parameter_name: str,
    ) -> tuple[bool, ...]:
        if isinstance(value, tuple):
            if len(value) != pair_count:
                raise ValueError(
                    f"CorrectIlluminationApply {parameter_name} count must match "
                    f"image/function pair count; got {len(value)} values for "
                    f"{pair_count} pairs."
                )
            return tuple(bool(item) for item in value)
        return (bool(value),) * pair_count


@dataclass(frozen=True, slots=True)
class IlluminationCorrectionInputStack:
    """Stacked image/function pairs and source metadata for application."""

    image: object
    pixel_stack: np.ndarray

    @classmethod
    def from_image(cls, image: object) -> "IlluminationCorrectionInputStack":
        pixel_stack = np.asarray(image_payload_data(image))
        if pixel_stack.ndim < 3 or pixel_stack.shape[0] % 2 != 0:
            raise ValueError(
                "CorrectIlluminationApply requires stacked image/function pairs "
                f"with shape (2*N, ...), got {pixel_stack.shape!r}."
            )
        return cls(image=image, pixel_stack=pixel_stack)

    @property
    def pair_count(self) -> int:
        return int(self.pixel_stack.shape[0] // 2)

    def image_pixels(self, pair_index: int) -> np.ndarray:
        return self.pixel_stack[self.input_index(pair_index)]

    def illumination_function(self, pair_index: int) -> np.ndarray:
        return self.pixel_stack[self.input_index(pair_index) + 1]

    @staticmethod
    def input_index(pair_index: int) -> int:
        return pair_index * 2

    def input_mask(self, pair_index: int) -> object | None:
        mask = image_payload_mask(self.image)
        if mask is None:
            return None
        mask_array = np.asarray(mask, dtype=bool)
        input_index = self.input_index(pair_index)
        if mask_array.ndim == 3 and mask_array.shape[0] > 0:
            return mask_array[input_index : input_index + 1]
        return mask_array


class IlluminationCorrection:
    """Load-bearing CorrectIlluminationApply execution policy."""

    def apply_pair(
        self,
        source: IlluminationCorrectionInputStack,
        pair_index: int,
        *,
        method: IlluminationCorrectionMethod,
        truncate_low: bool,
        truncate_high: bool,
    ) -> np.ndarray:
        image_pixels = source.image_pixels(pair_index)
        illumination_function = source.illumination_function(pair_index)
        if image_pixels.shape != illumination_function.shape:
            raise ValueError(
                f"Input image shape {image_pixels.shape} and illumination function "
                f"shape {illumination_function.shape} must be equal."
            )

        output_pixels = IlluminationCorrectionStrategy.for_method(method).apply(
            IlluminationCorrectionRequest(
                image_pixels=image_pixels,
                illumination_function=illumination_function,
            )
        )
        if truncate_low:
            np.maximum(output_pixels, 0.0, out=output_pixels)
        if truncate_high:
            np.minimum(output_pixels, 1.0, out=output_pixels)
        return image_payload_with_context(
            output_pixels[np.newaxis, ...].astype(np.float32, copy=False),
            mask=source.input_mask(pair_index),
            metadata=(
                image_payload_metadata(source.image)
                .for_channel(source.input_index(pair_index))
                .without_unit_interval_intensity_scale()
            ),
        )


@dataclass(frozen=True, slots=True)
class IlluminationCalculationRequest:
    """Complete semantic request for CorrectIlluminationCalculate."""

    pixel_data: np.ndarray
    mask: np.ndarray | None
    intensity_choice: IntensityChoice
    dilate_objects: bool
    object_dilation_radius: int
    block_size: int
    rescale_option: RescaleOption
    smoothing_method: SmoothingMethod
    filter_size_method: FilterSizeMethod
    object_width: int
    manual_filter_size: int
    automatic_splines: bool
    spline_bg_mode: SplineBgMode
    spline_points: int
    spline_threshold: float
    spline_rescale: float
    spline_max_iterations: int
    spline_convergence: float
    calculation_scope: CalculationScope
    morphology: MorphologyBackendStrategy
    convex_hull_backend_provider: CellProfilerBackendProvider | None
    rank_median_backend_provider: CellProfilerBackendProvider | None
    slice_index: int = 0

    @property
    def is_multi_image_stack(self) -> bool:
        return self.pixel_data.ndim >= 3

    @property
    def spatial_image_shape(self) -> tuple[int, ...]:
        if self.calculation_scope.uses_all_images and self.is_multi_image_stack:
            return tuple(self.pixel_data.shape[1:])
        return tuple(self.pixel_data.shape)

    def for_stack_slice(
        self,
        slice_index: int,
        mask: np.ndarray | None,
    ) -> "IlluminationCalculationRequest":
        return IlluminationCalculationRequest(
            pixel_data=np.asarray(self.pixel_data[slice_index]),
            mask=mask,
            intensity_choice=self.intensity_choice,
            dilate_objects=self.dilate_objects,
            object_dilation_radius=self.object_dilation_radius,
            block_size=self.block_size,
            rescale_option=self.rescale_option,
            smoothing_method=self.smoothing_method,
            filter_size_method=self.filter_size_method,
            object_width=self.object_width,
            manual_filter_size=self.manual_filter_size,
            automatic_splines=self.automatic_splines,
            spline_bg_mode=self.spline_bg_mode,
            spline_points=self.spline_points,
            spline_threshold=self.spline_threshold,
            spline_rescale=self.spline_rescale,
            spline_max_iterations=self.spline_max_iterations,
            spline_convergence=self.spline_convergence,
            calculation_scope=CalculationScope.EACH,
            morphology=self.morphology,
            convex_hull_backend_provider=self.convex_hull_backend_provider,
            rank_median_backend_provider=self.rank_median_backend_provider,
            slice_index=slice_index,
        )


class IlluminationCalculation:
    """Load-bearing CorrectIlluminationCalculate execution policy."""

    function_name = "correct_illumination_calculate"

    def calculate(self, request: IlluminationCalculationRequest) -> tuple[np.ndarray, IlluminationStats]:
        total_started_at = time.perf_counter()
        filter_size = self.filter_size(request)
        avg_image = self.average_image(request)
        dilated_image = self.apply_dilation(request, avg_image)
        smoothed_image = self.smooth(request, dilated_image, filter_size)
        output_image = self.apply_scaling(
            smoothed_image,
            request.mask,
            request.rescale_option,
        ).astype(np.float32)
        runtime_profiler.log(
            "cic_total",
            time.perf_counter() - total_started_at,
            function=self.function_name,
        )
        return output_image, self.stats(request, output_image)

    def filter_size(self, request: IlluminationCalculationRequest) -> float:
        phase_started_at = time.perf_counter()
        filter_size = SmoothingFilterSizeStrategy.for_method(
            request.filter_size_method,
        ).calculate(
            SmoothingFilterSizeRequest(
                image_shape=request.spatial_image_shape,
                object_width=request.object_width,
                manual_filter_size=request.manual_filter_size,
            )
        )
        runtime_profiler.log(
            "cic_filter_size",
            time.perf_counter() - phase_started_at,
            function=self.function_name,
            method=request.filter_size_method.value,
            smoothing=request.smoothing_method.value,
        )
        return filter_size

    def average_image(self, request: IlluminationCalculationRequest) -> np.ndarray:
        phase_started_at = time.perf_counter()
        if not (
            request.calculation_scope.uses_all_images
            and request.is_multi_image_stack
        ):
            average_image = self.preprocess_for_averaging(
                request.pixel_data,
                request.mask,
                request,
            )
        else:
            if request.pixel_data.shape[0] == 0:
                raise ValueError(
                    "All-image illumination calculation requires at least one image."
                )
            illumination_mask = IlluminationMask(request.mask, request.pixel_data)
            averaged_inputs = [
                self.preprocess_for_averaging(
                    np.asarray(slice_data),
                    illumination_mask.for_stack_slice(slice_index),
                    request,
                )
                for slice_index, slice_data in enumerate(request.pixel_data)
            ]
            average_image = np.mean(np.stack(averaged_inputs, axis=0), axis=0)
        runtime_profiler.log(
            "cic_average_image",
            time.perf_counter() - phase_started_at,
            function=self.function_name,
            method=request.intensity_choice.value,
            scope=request.calculation_scope.value,
        )
        return average_image

    def preprocess_for_averaging(
        self,
        pixel_data: np.ndarray,
        mask: np.ndarray | None,
        request: IlluminationCalculationRequest,
    ) -> np.ndarray:
        if (
            request.intensity_choice == IntensityChoice.REGULAR
            or request.smoothing_method == SmoothingMethod.SPLINES
        ):
            result = pixel_data.copy()
            if mask is not None:
                result[~mask] = 0
            return result
        return request.morphology.blockwise_minimum(
            pixel_data,
            mask,
            request.block_size,
        )

    def apply_dilation(
        self,
        request: IlluminationCalculationRequest,
        pixel_data: np.ndarray,
    ) -> np.ndarray:
        phase_started_at = time.perf_counter()
        if not request.dilate_objects:
            result = pixel_data
        else:
            result = IlluminationGaussianFilter(
                pixel_data,
                request.mask,
                request.object_dilation_radius,
            ).apply()
            if request.mask is not None:
                result[~request.mask] = 0
        runtime_profiler.log(
            "cic_dilation",
            time.perf_counter() - phase_started_at,
            function=self.function_name,
            enabled=request.dilate_objects,
        )
        return result

    def smooth(
        self,
        request: IlluminationCalculationRequest,
        pixel_data: np.ndarray,
        filter_size: float,
    ) -> np.ndarray:
        phase_started_at = time.perf_counter()
        smoothed_image = SmoothingPlaneStrategy.for_method(request.smoothing_method).smooth(
            SmoothingPlaneRequest(
                pixel_data=pixel_data,
                mask=request.mask,
                smoothing_method=request.smoothing_method,
                filter_size=filter_size,
                spline_bg_mode=request.spline_bg_mode,
                spline_points=request.spline_points,
                spline_threshold=request.spline_threshold,
                spline_rescale=request.spline_rescale,
                spline_max_iterations=request.spline_max_iterations,
                spline_convergence=request.spline_convergence,
                automatic_splines=request.automatic_splines,
                morphology=request.morphology,
                convex_hull_backend_provider=request.convex_hull_backend_provider,
                rank_median_backend_provider=request.rank_median_backend_provider,
            )
        )
        runtime_profiler.log(
            "cic_smoothing",
            time.perf_counter() - phase_started_at,
            function=self.function_name,
            method=request.smoothing_method.value,
        )
        return smoothed_image

    def apply_scaling(
        self,
        pixel_data: np.ndarray,
        mask: np.ndarray | None,
        rescale_option: RescaleOption,
    ) -> np.ndarray:
        phase_started_at = time.perf_counter()
        if rescale_option == RescaleOption.NO:
            result = pixel_data
        else:
            projected_mask = project_image_mask_to_data_domain(mask, pixel_data)
            if projected_mask is not None:
                sorted_data = pixel_data[(pixel_data > 0) & projected_mask]
            else:
                sorted_data = pixel_data[pixel_data > 0]

            if sorted_data.size == 0:
                result = pixel_data
            elif rescale_option == RescaleOption.YES:
                idx = int(len(sorted_data) * ROBUST_FACTOR)
                robust_minimum = np.partition(sorted_data, idx)[idx]
                result = pixel_data.copy()
                result[result < robust_minimum] = robust_minimum
                if robust_minimum != 0:
                    result = result / robust_minimum
            else:
                idx = len(sorted_data) // 2
                robust_minimum = np.partition(sorted_data, idx)[idx]
                result = pixel_data.copy()
                if robust_minimum != 0:
                    result = result / robust_minimum
        runtime_profiler.log(
            "cic_scaling",
            time.perf_counter() - phase_started_at,
            function=self.function_name,
            method=rescale_option.value,
        )
        return result

    def stats(
        self,
        request: IlluminationCalculationRequest,
        output_image: np.ndarray,
    ) -> IlluminationStats:
        phase_started_at = time.perf_counter()
        stats = IlluminationStats(
            slice_index=request.slice_index,
            min_value=float(np.min(output_image)),
            max_value=float(np.max(output_image)),
            mean_value=float(np.mean(output_image)),
            calculation_type=request.intensity_choice.value,
            smoothing_method=request.smoothing_method.value,
        )
        runtime_profiler.log(
            "cic_stats",
            time.perf_counter() - phase_started_at,
            function=self.function_name,
        )
        return stats


@numpy(contract=ProcessingContract.FLEXIBLE)
@special_outputs(
    (
        "illumination_stats",
        csv_materializer(
            fields=[
                "slice_index",
                "min_value",
                "max_value",
                "mean_value",
                "calculation_type",
                "smoothing_method",
            ],
            analysis_type="illumination_correction",
        ),
    )
)
def correct_illumination_calculate(
    image: np.ndarray,
    intensity_choice: IntensityChoice | str = IntensityChoice.REGULAR,
    dilate_objects: bool = False,
    object_dilation_radius: int = 1,
    block_size: int = 60,
    rescale_option: RescaleOption | str = RescaleOption.YES,
    smoothing_method: SmoothingMethod | str = SmoothingMethod.FIT_POLYNOMIAL,
    filter_size_method: FilterSizeMethod | str = FilterSizeMethod.AUTOMATIC,
    object_width: int = 10,
    manual_filter_size: int = 10,
    automatic_splines: bool = True,
    spline_bg_mode: SplineBgMode | str = SplineBgMode.AUTO,
    spline_points: int = 5,
    spline_threshold: float = 2.0,
    spline_rescale: float = 2.0,
    spline_max_iterations: int = 40,
    spline_convergence: float = 0.001,
    calculation_scope: CalculationScope | str = CalculationScope.EACH,
    convex_hull_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    rank_median_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> tuple[np.ndarray, IlluminationStats]:
    """Calculate an illumination correction function."""
    intensity_choice = coerce_cellprofiler_enum(IntensityChoice, intensity_choice)
    rescale_option = coerce_cellprofiler_enum(RescaleOption, rescale_option)
    smoothing_method = coerce_cellprofiler_enum(SmoothingMethod, smoothing_method)
    filter_size_method = coerce_cellprofiler_enum(FilterSizeMethod, filter_size_method)
    spline_bg_mode = coerce_cellprofiler_enum(SplineBgMode, spline_bg_mode)
    calculation_scope = coerce_cellprofiler_enum(CalculationScope, calculation_scope)

    morphology = MorphologyBackendStrategy.for_callable(
        correct_illumination_calculate,
    )

    pixel_data = np.asarray(image_payload_data(image))
    illumination_mask = IlluminationMask(image_payload_mask(image), pixel_data)
    mask = illumination_mask.normalized
    metadata = image_payload_metadata(image).without_unit_interval_intensity_scale()

    request = IlluminationCalculationRequest(
        pixel_data=pixel_data,
        mask=mask,
        intensity_choice=intensity_choice,
        dilate_objects=dilate_objects,
        object_dilation_radius=object_dilation_radius,
        block_size=block_size,
        rescale_option=rescale_option,
        smoothing_method=smoothing_method,
        filter_size_method=filter_size_method,
        object_width=object_width,
        manual_filter_size=manual_filter_size,
        automatic_splines=automatic_splines,
        spline_bg_mode=spline_bg_mode,
        spline_points=spline_points,
        spline_threshold=spline_threshold,
        spline_rescale=spline_rescale,
        spline_max_iterations=spline_max_iterations,
        spline_convergence=spline_convergence,
        calculation_scope=calculation_scope,
        morphology=morphology,
        convex_hull_backend_provider=convex_hull_backend_provider,
        rank_median_backend_provider=rank_median_backend_provider,
    )
    calculation = IlluminationCalculation()

    if request.is_multi_image_stack and not request.calculation_scope.uses_all_images:
        slice_results = [
            calculation.calculate(
                request.for_stack_slice(
                    slice_index,
                    illumination_mask.for_stack_slice(slice_index),
                )
            )
            for slice_index in range(pixel_data.shape[0])
        ]
        illumination_stack = np.stack(
            [result[0] for result in slice_results],
            axis=0,
        ).astype(np.float32)
        return (
            image_payload_with_context(
                illumination_stack,
                mask=illumination_mask.for_output(illumination_stack),
                metadata=metadata,
            ),
            [result[1] for result in slice_results],
        )

    illumination, stats = calculation.calculate(request)
    return (
        image_payload_with_context(
            illumination,
            mask=illumination_mask.for_output(illumination),
            metadata=metadata,
        ),
        stats,
    )


def _prepare_correct_illumination_calculate() -> None:
    """Compile common illumination kernels outside measured step execution."""
    image = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape((64, 64))
    correct_illumination_calculate.__wrapped__(
        image,
        smoothing_method=SmoothingMethod.FIT_POLYNOMIAL,
        filter_size_method=FilterSizeMethod.AUTOMATIC,
        rescale_option=RescaleOption.YES,
    )
    background = np.zeros((64, 64), dtype=np.float32)
    background[::8, ::8] = 1.0
    correct_illumination_calculate.__wrapped__(
        background,
        intensity_choice=IntensityChoice.BACKGROUND,
        block_size=8,
        smoothing_method=SmoothingMethod.MEDIAN_FILTER,
        filter_size_method=FilterSizeMethod.MANUALLY,
        manual_filter_size=32,
        rescale_option=RescaleOption.NO,
    )
    nonconstant_background = np.linspace(
        0.0,
        1.0,
        96 * 96,
        dtype=np.float32,
    ).reshape((96, 96))
    correct_illumination_calculate.__wrapped__(
        nonconstant_background,
        intensity_choice=IntensityChoice.REGULAR,
        smoothing_method=SmoothingMethod.MEDIAN_FILTER,
        filter_size_method=FilterSizeMethod.MANUALLY,
        manual_filter_size=96,
        rescale_option=RescaleOption.NO,
    )


correct_illumination_calculate.__openhcs_prepare__ = (
    _prepare_correct_illumination_calculate
)


@numpy
def correct_illumination_apply(
    image: np.ndarray,
    method: (
        IlluminationCorrectionMethod
        | str
        | tuple[IlluminationCorrectionMethod | str, ...]
    ) = IlluminationCorrectionMethod.DIVIDE,
    truncate_low: bool | tuple[bool, ...] = True,
    truncate_high: bool | tuple[bool, ...] = True,
) -> np.ndarray | tuple[np.ndarray, ...]:
    """Apply illumination correction to stacked image/function pairs."""
    source = IlluminationCorrectionInputStack.from_image(image)
    settings = IlluminationCorrectionSettingSequence.from_settings(
        method,
        truncate_low,
        truncate_high,
        source.pair_count,
    )
    correction = IlluminationCorrection()
    outputs = tuple(
        correction.apply_pair(
            source,
            pair_index,
            method=settings.methods[pair_index],
            truncate_low=settings.truncate_low[pair_index],
            truncate_high=settings.truncate_high[pair_index],
        )
        for pair_index in range(source.pair_count)
    )
    if source.pair_count == 1:
        return outputs[0]
    return outputs


@processing_prepare(correct_illumination_apply)
def _prepare_correct_illumination_apply() -> None:
    """Materialize correction strategy registry before timed execution."""
    pixels = np.stack(
        (
            np.full((16, 16), 0.5, dtype=np.float32),
            np.full((16, 16), 0.25, dtype=np.float32),
        ),
        axis=0,
    )
    correct_illumination_apply.__wrapped__(
        pixels,
        method=IlluminationCorrectionMethod.DIVIDE,
    )
    correct_illumination_apply.__wrapped__(
        pixels,
        method=IlluminationCorrectionMethod.SUBTRACT,
    )


@dataclass(frozen=True, slots=True)
class SmoothingFilterSizeRequest:
    """Inputs needed to derive a smoothing filter size."""

    image_shape: tuple[int, ...]
    object_width: int
    manual_filter_size: int


class SmoothingFilterSizeStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal filter-size derivation for one closed CellProfiler mode."""

    __registry_key__ = "method_label"
    __skip_if_no_key__ = True
    method_label: ClassVar[str | None] = None
    method: ClassVar[FilterSizeMethod | None] = None

    @classmethod
    def for_method(
        cls,
        method: FilterSizeMethod,
    ) -> "SmoothingFilterSizeStrategy":
        return cls.__registry__[method.value]()

    @abstractmethod
    def calculate(self, request: SmoothingFilterSizeRequest) -> float:
        """Return the smoothing filter size."""


class ManualSmoothingFilterSizeStrategy(SmoothingFilterSizeStrategy):
    method = FilterSizeMethod.MANUALLY
    method_label = method.value

    def calculate(self, request: SmoothingFilterSizeRequest) -> float:
        return float(request.manual_filter_size)


class ObjectWidthSmoothingFilterSizeStrategy(SmoothingFilterSizeStrategy):
    method = FilterSizeMethod.OBJECT_SIZE
    method_label = method.value

    def calculate(self, request: SmoothingFilterSizeRequest) -> float:
        return request.object_width * 2.35 / 3.5


class AutomaticSmoothingFilterSizeStrategy(SmoothingFilterSizeStrategy):
    method = FilterSizeMethod.AUTOMATIC
    method_label = method.value

    def calculate(self, request: SmoothingFilterSizeRequest) -> float:
        return min(30.0, float(np.max(request.image_shape)) / 40.0)


@dataclass(frozen=True, slots=True)
class SmoothingPlaneRequest:
    """Authoritative smoothing context for illumination background estimation."""

    pixel_data: np.ndarray
    mask: np.ndarray | None
    smoothing_method: SmoothingMethod
    filter_size: float
    spline_bg_mode: SplineBgMode
    spline_points: int
    spline_threshold: float
    spline_rescale: float
    spline_max_iterations: int
    spline_convergence: float
    automatic_splines: bool
    morphology: object
    convex_hull_backend_provider: CellProfilerBackendProvider | None
    rank_median_backend_provider: CellProfilerBackendProvider | None

    @property
    def sigma(self) -> float:
        return self.filter_size / 2.35


class SmoothingPlaneStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal smoothing implementation for one closed CellProfiler mode."""

    __registry_key__ = "method_label"
    __skip_if_no_key__ = True
    method_label: ClassVar[str | None] = None
    method: ClassVar[SmoothingMethod | None] = None

    @classmethod
    def for_method(cls, method: SmoothingMethod) -> "SmoothingPlaneStrategy":
        return cls.__registry__[method.value]()

    @abstractmethod
    def smooth(self, request: SmoothingPlaneRequest) -> np.ndarray:
        """Return the smoothed illumination plane."""


class NoSmoothingPlaneStrategy(SmoothingPlaneStrategy):
    method = SmoothingMethod.NONE
    method_label = method.value

    def smooth(self, request: SmoothingPlaneRequest) -> np.ndarray:
        return request.pixel_data


class FitPolynomialSmoothingPlaneStrategy(SmoothingPlaneStrategy):
    method = SmoothingMethod.FIT_POLYNOMIAL
    method_label = method.value

    def smooth(self, request: SmoothingPlaneRequest) -> np.ndarray:
        return fit_polynomial_surface(
            request.pixel_data,
            request.mask,
        )


class GaussianFilterSmoothingPlaneStrategy(SmoothingPlaneStrategy):
    method = SmoothingMethod.GAUSSIAN_FILTER
    method_label = method.value

    def smooth(self, request: SmoothingPlaneRequest) -> np.ndarray:
        return IlluminationGaussianFilter(
            request.pixel_data,
            request.mask,
            request.sigma,
        ).apply()


class MedianFilterSmoothingPlaneStrategy(SmoothingPlaneStrategy):
    method = SmoothingMethod.MEDIAN_FILTER
    method_label = method.value

    def smooth(self, request: SmoothingPlaneRequest) -> np.ndarray:
        filter_sigma = max(1, int(request.sigma + 0.5))
        return RankMedianSmoothingBackendStrategy.for_memory_type(
            backend_provider=request.rank_median_backend_provider,
        ).smooth_background_plane(
            request.pixel_data,
            mask=request.mask,
            radius=filter_sigma,
            morphology=request.morphology,
        )


class AverageSmoothingPlaneStrategy(SmoothingPlaneStrategy):
    method = SmoothingMethod.TO_AVERAGE
    method_label = method.value

    def smooth(self, request: SmoothingPlaneRequest) -> np.ndarray:
        if request.mask is not None:
            mean_val = np.mean(request.pixel_data[request.mask])
        else:
            mean_val = np.mean(request.pixel_data)
        return np.full(
            request.pixel_data.shape,
            mean_val,
            dtype=request.pixel_data.dtype,
        )


class ConvexHullSmoothingPlaneStrategy(SmoothingPlaneStrategy):
    method = SmoothingMethod.CONVEX_HULL
    method_label = method.value

    def smooth(self, request: SmoothingPlaneRequest) -> np.ndarray:
        return ConvexHullSmoothingBackendStrategy.for_memory_type(
            backend_provider=request.convex_hull_backend_provider,
        ).smooth_background_plane(
            request.pixel_data,
            mask=request.mask,
            filter_size=request.filter_size,
            morphology=request.morphology,
        )


class SplinesSmoothingPlaneStrategy(SmoothingPlaneStrategy):
    method = SmoothingMethod.SPLINES
    method_label = method.value

    def smooth(self, request: SmoothingPlaneRequest) -> np.ndarray:
        from scipy.interpolate import RectBivariateSpline

        pixel_data = request.pixel_data
        h, w = pixel_data.shape
        if request.automatic_splines:
            shortest_side = min(h, w)
            scale = max(1, shortest_side // 200)
            n_points = 5
        else:
            scale = int(request.spline_rescale)
            n_points = request.spline_points
        downsampled = pixel_data[::scale, ::scale]
        dh, dw = downsampled.shape
        y_points = np.linspace(0, dh - 1, n_points)
        x_points = np.linspace(0, dw - 1, n_points)
        yi = np.clip(np.round(y_points).astype(int), 0, dh - 1)
        xi = np.clip(np.round(x_points).astype(int), 0, dw - 1)
        spline = RectBivariateSpline(
            y_points,
            x_points,
            downsampled[np.ix_(yi, xi)],
            kx=3,
            ky=3,
        )
        result = spline(
            np.linspace(0, dh - 1, h),
            np.linspace(0, dw - 1, w),
        )
        if request.mask is not None:
            result[request.mask] -= np.mean(result[request.mask])
        else:
            result -= np.mean(result)
        return result


def fit_polynomial_surface(
    pixel_data: np.ndarray,
    mask: np.ndarray | None,
) -> np.ndarray:
    """Fit CP's quadratic illumination surface without dense design matrices."""
    image = np.ascontiguousarray(pixel_data, dtype=np.float64)
    if image.ndim != 2:
        raise NotImplementedError(
            "Fit-polynomial illumination smoothing currently supports 2-D "
            f"NumPy planes, got shape {image.shape!r}."
        )
    mask_array = (
        np.empty((0, 0), dtype=np.bool_)
        if mask is None
        else np.ascontiguousarray(mask, dtype=np.bool_)
    )
    if mask is not None and mask_array.shape != image.shape:
        raise ValueError(
            "Fit-polynomial illumination mask must match the image shape; got "
            f"mask {mask_array.shape!r} for image {image.shape!r}."
        )
    if mask is None:
        gram = fit_polynomial_unmasked_gram(image.shape[0], image.shape[1])
        rhs = _fit_polynomial_unmasked_rhs_numba(image)
    else:
        gram, rhs = _fit_polynomial_normal_equations_numba(
            image,
            mask_array,
            True,
        )
    coeffs = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    return _evaluate_polynomial_surface_numba(
        image.shape[0],
        image.shape[1],
        np.ascontiguousarray(coeffs, dtype=np.float64),
    )


@lru_cache(maxsize=16)
def fit_polynomial_unmasked_gram(height: int, width: int) -> np.ndarray:
    return _fit_polynomial_unmasked_gram_numba(int(height), int(width))


@njit(cache=True)
def _fit_polynomial_unmasked_gram_numba(height: int, width: int) -> np.ndarray:
    gram = np.zeros((6, 6), dtype=np.float64)
    features = np.empty(6, dtype=np.float64)
    for row in range(height):
        y_value = row / height - 0.5
        y2 = y_value * y_value
        for col in range(width):
            x_value = col / width - 0.5
            features[0] = x_value * x_value
            features[1] = y2
            features[2] = x_value * y_value
            features[3] = x_value
            features[4] = y_value
            features[5] = 1.0
            for i in range(6):
                for j in range(6):
                    gram[i, j] += features[i] * features[j]
    return gram


@njit(cache=True)
def _fit_polynomial_unmasked_rhs_numba(pixel_data: np.ndarray) -> np.ndarray:
    height, width = pixel_data.shape
    rhs = np.zeros(6, dtype=np.float64)
    for row in range(height):
        y_value = row / height - 0.5
        y2 = y_value * y_value
        for col in range(width):
            x_value = col / width - 0.5
            value = pixel_data[row, col]
            rhs[0] += x_value * x_value * value
            rhs[1] += y2 * value
            rhs[2] += x_value * y_value * value
            rhs[3] += x_value * value
            rhs[4] += y_value * value
            rhs[5] += value
    return rhs


@njit(cache=True)
def _fit_polynomial_normal_equations_numba(
    pixel_data: np.ndarray,
    mask: np.ndarray,
    has_mask: bool,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = pixel_data.shape
    gram = np.zeros((6, 6), dtype=np.float64)
    rhs = np.zeros(6, dtype=np.float64)
    features = np.empty(6, dtype=np.float64)
    for row in range(height):
        y_value = row / height - 0.5
        y2 = y_value * y_value
        for col in range(width):
            if has_mask and not mask[row, col]:
                continue
            x_value = col / width - 0.5
            features[0] = x_value * x_value
            features[1] = y2
            features[2] = x_value * y_value
            features[3] = x_value
            features[4] = y_value
            features[5] = 1.0
            value = pixel_data[row, col]
            for i in range(6):
                rhs[i] += features[i] * value
                for j in range(6):
                    gram[i, j] += features[i] * features[j]
    return gram, rhs


@njit(cache=True)
def _evaluate_polynomial_surface_numba(
    height: int,
    width: int,
    coeffs: np.ndarray,
) -> np.ndarray:
    output = np.empty((height, width), dtype=np.float64)
    for row in range(height):
        y_value = row / height - 0.5
        y2 = y_value * y_value
        for col in range(width):
            x_value = col / width - 0.5
            output[row, col] = (
                coeffs[0] * x_value * x_value
                + coeffs[1] * y2
                + coeffs[2] * x_value * y_value
                + coeffs[3] * x_value
                + coeffs[4] * y_value
                + coeffs[5]
            )
    return output


class ConvexHullSmoothingBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Convex-hull illumination smoothing keyed by OpenHCS memory/provider."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def smooth_background_plane(
        self,
        pixel_data: np.ndarray,
        *,
        mask: np.ndarray | None,
        filter_size: float,
        morphology: object,
    ) -> np.ndarray:
        """Return a smoothed illumination background plane."""


class RankMedianSmoothingBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Rank-median illumination smoothing keyed by OpenHCS memory/provider."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @staticmethod
    def disk_rows(footprint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Collapse a dense disk footprint into per-row horizontal radii."""
        center_y = footprint.shape[0] // 2
        center_x = footprint.shape[1] // 2
        rows: list[int] = []
        radii: list[int] = []
        for y in range(footprint.shape[0]):
            xs = np.flatnonzero(footprint[y])
            if xs.size == 0:
                continue
            rows.append(y - center_y)
            radii.append(int(np.max(np.abs(xs - center_x))))
        return (
            np.asarray(rows, dtype=np.int64),
            np.asarray(radii, dtype=np.int64),
        )

    @abstractmethod
    def smooth_background_plane(
        self,
        pixel_data: np.ndarray,
        *,
        mask: np.ndarray | None,
        radius: int,
        morphology: object,
    ) -> np.ndarray:
        """Return a rank-median smoothed illumination background plane."""


class NumbaNumpyRankMedianSmoothingBackendStrategy(
    RankMedianSmoothingBackendStrategy,
):
    """NumPy-memory rank median matching skimage rank median border semantics."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = False

    def prepare_backend(self) -> None:
        """Compile rank-median numba kernels during compiler preparation."""
        footprint = np.ones((3, 3), dtype=np.bool_)
        row_offsets_y, row_radii_x = self.disk_rows(footprint)
        scaled = np.arange(16, dtype=np.uint16).reshape((4, 4))
        code_image = np.arange(16, dtype=np.int32).reshape((4, 4))
        mask = np.ones(scaled.shape, dtype=np.bool_)

        _rank_median_global_minimum_is_majority_everywhere_numba(
            scaled,
            row_offsets_y,
            row_radii_x,
            np.uint16(0),
        )
        _rank_median_codes_2d_sliding_histogram_numba(
            code_image,
            row_offsets_y,
            row_radii_x,
            int(code_image.size),
        )
        _rank_median_uint16_2d_sliding_histogram_numba(
            scaled,
            mask,
            row_offsets_y,
            row_radii_x,
        )

    def smooth_background_plane(
        self,
        pixel_data: np.ndarray,
        *,
        mask: np.ndarray | None,
        radius: int,
        morphology: object,
    ) -> np.ndarray:
        image = np.asarray(pixel_data, dtype=np.float32)
        if image.ndim != 2:
            raise NotImplementedError(
                "Rank-median illumination smoothing currently supports 2-D "
                f"NumPy planes, got shape {image.shape!r}."
            )
        mask = project_image_mask_to_data_domain(mask, image)
        if mask is not None and np.asarray(mask).shape != image.shape:
            raise ValueError(
                "Rank-median illumination mask must match the image shape; got "
                f"mask {np.asarray(mask).shape!r} for image {image.shape!r}."
        )
        footprint = np.asarray(morphology.disk_footprint(radius), dtype=np.bool_)
        row_offsets_y, row_radii_x = self.disk_rows(footprint)
        scaled = (image * 65535.0).astype(np.uint16)
        mask_array = (
            np.ones(image.shape, dtype=np.bool_)
            if mask is None
            else np.asarray(mask, dtype=np.bool_)
        )
        effective_scaled = scaled.copy()
        effective_scaled[~mask_array] = np.uint16(0)
        minimum_value = np.min(effective_scaled)
        phase_started_at = time.perf_counter()
        if np.all(effective_scaled == minimum_value):
            runtime_profiler.log(
                "rank_median_constant_minimum",
                time.perf_counter() - phase_started_at,
                radius=radius,
            )
            return np.full(image.shape, minimum_value, dtype=np.float32) / 65535.0
        runtime_profiler.log(
            "rank_median_constant_minimum",
            time.perf_counter() - phase_started_at,
            radius=radius,
        )

        phase_started_at = time.perf_counter()
        if _rank_median_global_minimum_is_majority_everywhere_numba(
            np.ascontiguousarray(effective_scaled),
            row_offsets_y,
            row_radii_x,
            minimum_value,
        ):
            runtime_profiler.log(
                "rank_median_minimum_majority",
                time.perf_counter() - phase_started_at,
                radius=radius,
                result=True,
            )
            return np.full(image.shape, minimum_value, dtype=np.float32) / 65535.0
        runtime_profiler.log(
            "rank_median_minimum_majority",
            time.perf_counter() - phase_started_at,
            radius=radius,
            result=False,
        )

        phase_started_at = time.perf_counter()
        values, inverse = np.unique(effective_scaled, return_inverse=True)
        runtime_profiler.log(
            "rank_median_unique_codes",
            time.perf_counter() - phase_started_at,
            radius=radius,
            value_count=int(values.size),
        )
        phase_started_at = time.perf_counter()
        code_image = inverse.reshape(image.shape).astype(np.int32, copy=False)
        result_codes = _rank_median_codes_2d_sliding_histogram_numba(
            np.ascontiguousarray(code_image),
            row_offsets_y,
            row_radii_x,
            int(values.size),
        )
        runtime_profiler.log(
            "rank_median_numba_codes",
            time.perf_counter() - phase_started_at,
            radius=radius,
            value_count=int(values.size),
        )
        result = values[result_codes]
        return result.astype(np.float32) / 65535.0


class NativeNumpyRankMedianSmoothingBackendStrategy(
    RankMedianSmoothingBackendStrategy,
):
    """Compact-domain skimage rank-median backend for NumPy planes."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NATIVE,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NATIVE
    is_default_backend = True

    def prepare_backend(self) -> None:
        """Compile exact compact-domain rank-median kernels during preparation."""
        code_image = np.arange(16, dtype=np.uint8).reshape((4, 4))
        footprint = np.ones((3, 3), dtype=np.bool_)
        row_offsets_y, row_radii_x = self.disk_rows(footprint)
        _rank_median_small_codes_2d_sliding_histogram_numba(
            code_image,
            row_offsets_y,
            row_radii_x,
            int(code_image.size),
        )

    def smooth_background_plane(
        self,
        pixel_data: np.ndarray,
        *,
        mask: np.ndarray | None,
        radius: int,
        morphology: object,
    ) -> np.ndarray:
        image = np.asarray(pixel_data, dtype=np.float32)
        projected_mask = project_image_mask_to_data_domain(mask, image)
        mask_array = (
            None if projected_mask is None else np.asarray(projected_mask, dtype=np.bool_)
        )
        if mask_array is not None and mask_array.shape != image.shape:
            raise ValueError(
                "Rank-median illumination mask must match the image shape; got "
                f"mask {mask_array.shape!r} for image {image.shape!r}."
        )
        footprint = np.asarray(morphology.disk_footprint(radius), dtype=np.bool_)
        row_offsets_y, row_radii_x = self.disk_rows(footprint)
        scaled = (image * 65535.0).astype(np.uint16)
        effective_scaled = scaled if mask_array is None else scaled.copy()
        if mask_array is not None:
            effective_scaled[~mask_array] = np.uint16(0)
        minimum_value = np.min(effective_scaled)
        if not np.all(effective_scaled == minimum_value) and (
            _rank_median_global_minimum_is_majority_everywhere_numba(
                np.ascontiguousarray(effective_scaled),
                row_offsets_y,
                row_radii_x,
                minimum_value,
            )
        ):
            return np.full(image.shape, minimum_value, dtype=np.float32) / 65535.0
        return self._smooth_compact_rank_median(
            effective_scaled,
            footprint,
        )

    @staticmethod
    def _smooth_compact_rank_median(
        scaled: np.ndarray,
        footprint: np.ndarray,
    ) -> np.ndarray:
        import skimage.filters

        effective_scaled = np.asarray(scaled, dtype=np.uint16)
        values, inverse = np.unique(effective_scaled, return_inverse=True)
        if values.size == 1:
            result = skimage.filters.median(
                effective_scaled,
                footprint,
                behavior="rank",
            )
            return result.astype(np.float32) / 65535.0
        code_dtype = (
            np.uint8 if values.size <= np.iinfo(np.uint8).max + 1 else np.uint16
        )
        code_image = inverse.reshape(effective_scaled.shape).astype(code_dtype)
        if code_dtype == np.uint8:
            row_offsets_y, row_radii_x = RankMedianSmoothingBackendStrategy.disk_rows(
                footprint
            )
            result_codes = _rank_median_small_codes_2d_sliding_histogram_numba(
                np.ascontiguousarray(code_image),
                row_offsets_y,
                row_radii_x,
                int(values.size),
            )
            return values[result_codes].astype(np.float32) / 65535.0
        result_codes = skimage.filters.median(
            code_image,
            footprint,
            behavior="rank",
        )
        return values[result_codes].astype(np.float32) / 65535.0


class CentrosomeNumpyConvexHullSmoothingBackendStrategy(
    ConvexHullSmoothingBackendStrategy,
):
    """CellProfiler/centrosome reference convex-hull smoothing for NumPy planes."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.CENTROSOME,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.CENTROSOME
    is_default_backend = True

    def smooth_background_plane(
        self,
        pixel_data: np.ndarray,
        *,
        mask: np.ndarray | None,
        filter_size: float,
        morphology: object,
    ) -> np.ndarray:
        del filter_size, morphology
        import centrosome.cpmorphology
        import centrosome.filter

        image = np.asarray(pixel_data, dtype=np.float32)
        mask_array = (
            None
            if mask is None
            else np.asarray(mask, dtype=bool)
        )
        eroded = centrosome.cpmorphology.grey_erosion(
            image,
            2,
            mask_array,
        )
        transformed = centrosome.filter.convex_hull_transform(
            eroded,
            mask=mask_array,
        )
        return np.asarray(
            centrosome.cpmorphology.grey_dilation(
                transformed,
                2,
                mask_array,
            ),
            dtype=np.float32,
        )


class LegacyFastNumpyConvexHullSmoothingBackendStrategy(
    ConvexHullSmoothingBackendStrategy,
):
    """Fast CP3-compatible convex-hull smoothing for NumPy planes."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.LEGACY_FAST,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.LEGACY_FAST
    is_default_backend = False

    def smooth_background_plane(
        self,
        pixel_data: np.ndarray,
        *,
        mask: np.ndarray | None,
        filter_size: float,
        morphology: object,
    ) -> np.ndarray:
        del morphology
        from scipy.ndimage import grey_dilation, grey_erosion, maximum_filter

        image = np.asarray(pixel_data, dtype=np.float32)
        if image.ndim != 2:
            raise NotImplementedError(
                "Legacy-fast convex-hull smoothing currently supports 2-D "
                f"NumPy planes, got shape {image.shape!r}."
            )
        result = grey_dilation(
            maximum_filter(
                grey_erosion(image, size=3),
                size=max(1, int(filter_size)),
            ),
            size=3,
        )
        if mask is not None:
            result = np.asarray(result, dtype=np.float32)
            result[~np.asarray(mask, dtype=bool)] = 0
        return result.astype(np.float32, copy=False)


class ExactLevelSetNumpyConvexHullSmoothingBackendStrategy(
    ConvexHullSmoothingBackendStrategy,
):
    """Numba-accelerated exact level-set convex-hull reconstruction."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = False

    def prepare_backend(self) -> None:
        """Compile exact convex-hull smoothing kernels during compiler preparation."""
        image = np.arange(16, dtype=np.float32).reshape((4, 4))
        mask = np.ones(image.shape, dtype=np.bool_)
        morphology = MorphologyBackendStrategy.for_memory_type()
        self.smooth_background_plane(
            image,
            mask=mask,
            filter_size=3,
            morphology=morphology,
        )

    def smooth_background_plane(
        self,
        pixel_data: np.ndarray,
        *,
        mask: np.ndarray | None,
        filter_size: float,
        morphology: object,
    ) -> np.ndarray:
        del filter_size
        image = np.asarray(pixel_data, dtype=np.float32)
        if image.ndim != 2:
            raise NotImplementedError(
                "Exact convex-hull smoothing currently supports 2-D NumPy "
                f"planes, got shape {image.shape!r}."
            )
        valid_mask = (
            np.ones(image.shape, dtype=bool)
            if mask is None
            else np.asarray(mask, dtype=bool)
        )
        if valid_mask.shape != image.shape:
            raise ValueError(
                "Convex-hull smoothing requires a mask matching the 2-D "
                f"image plane, got mask {valid_mask.shape!r} for image "
                f"{image.shape!r}."
            )
        if not np.any(valid_mask):
            return np.zeros(image.shape, dtype=np.float32)

        eroded = _cellprofiler_masked_grey_erosion(
            image,
            valid_mask,
            _convex_hull_smoothing_footprint(morphology),
        )
        valid_values = eroded[valid_mask]
        thresholds = np.linspace(
            float(np.min(valid_values)),
            float(np.max(valid_values)),
            256,
            dtype=np.float32,
        )[1:]
        hull = _exact_level_set_convex_hull_smoothing_numba(
            np.ascontiguousarray(eroded, dtype=np.float32),
            np.ascontiguousarray(valid_mask, dtype=np.bool_),
            np.ascontiguousarray(thresholds, dtype=np.float32),
        )
        return _cellprofiler_masked_grey_dilation(
            hull,
            valid_mask,
            _convex_hull_smoothing_footprint(morphology),
        )


class NativeExactLevelSetNumpyConvexHullSmoothingBackendStrategy(
    ConvexHullSmoothingBackendStrategy,
):
    """Reference exact level-set convex-hull reconstruction for NumPy planes."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NATIVE,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NATIVE
    is_default_backend = False

    def smooth_background_plane(
        self,
        pixel_data: np.ndarray,
        *,
        mask: np.ndarray | None,
        filter_size: float,
        morphology: object,
    ) -> np.ndarray:
        del filter_size
        return _native_exact_level_set_convex_hull_smoothing(
            np.asarray(pixel_data, dtype=np.float32),
            None if mask is None else np.asarray(mask, dtype=bool),
            morphology,
        )


def _native_exact_level_set_convex_hull_smoothing(
    image: np.ndarray,
    mask: np.ndarray | None,
    morphology: object,
) -> np.ndarray:
    if image.ndim != 2:
        raise NotImplementedError(
            "Native exact convex-hull smoothing currently supports 2-D NumPy "
            f"planes, got shape {image.shape!r}."
        )
    valid_mask = (
        np.ones(image.shape, dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool)
    )
    if valid_mask.shape != image.shape:
        raise ValueError(
            "Convex-hull smoothing requires a mask matching the 2-D image "
            f"plane, got mask {valid_mask.shape!r} for image {image.shape!r}."
        )
    if not np.any(valid_mask):
        return np.zeros(image.shape, dtype=np.float32)

    footprint = _convex_hull_smoothing_footprint(morphology)
    eroded = _cellprofiler_masked_grey_erosion(image, valid_mask, footprint)
    valid_values = eroded[valid_mask]
    minimum = float(np.min(valid_values))
    maximum = float(np.max(valid_values))
    output = np.full(image.shape, minimum, dtype=np.float32)
    output[~valid_mask] = 0
    if maximum <= minimum:
        return _cellprofiler_masked_grey_dilation(output, valid_mask, footprint)

    for threshold in np.linspace(minimum, maximum, 256, dtype=np.float32)[1:]:
        level_mask = valid_mask & (eroded >= float(threshold))
        if not np.any(level_mask):
            continue
        output[morphology.convex_hull_image(level_mask) & valid_mask] = threshold
    return _cellprofiler_masked_grey_dilation(output, valid_mask, footprint)


def _convex_hull_smoothing_footprint(morphology: object) -> np.ndarray:
    """Return CP's radius-2 disk footprint for convex-hull smoothing."""
    return np.asarray(morphology.disk_footprint(2), dtype=bool)


def _cellprofiler_masked_grey_erosion(
    image: np.ndarray,
    mask: np.ndarray,
    footprint: np.ndarray,
) -> np.ndarray:
    """Match centrosome.cpmorphology.grey_erosion masking semantics."""
    from scipy import ndimage as ndi

    radius = max(1, int(np.ceil(np.max(np.asarray(footprint.shape)) / 2 - 0.5)))
    padded = np.ones(np.asarray(image.shape) + radius * 2, dtype=image.dtype)
    core = tuple(slice(radius, -radius) for _axis in image.shape)
    padded[core] = image
    padded_core = padded[core]
    padded_core[~mask] = 1
    eroded = ndi.grey_erosion(padded, footprint=footprint)[core]
    result = np.asarray(eroded, dtype=np.float32)
    result[~mask] = image[~mask]
    return result


def _cellprofiler_masked_grey_dilation(
    image: np.ndarray,
    mask: np.ndarray,
    footprint: np.ndarray,
) -> np.ndarray:
    """Match centrosome.cpmorphology.grey_dilation masking semantics."""
    from scipy import ndimage as ndi

    radius = max(1, int(np.ceil(np.max(np.asarray(footprint.shape)) / 2 - 0.5)))
    padded = np.zeros(np.asarray(image.shape) + radius * 2, dtype=image.dtype)
    core = tuple(slice(radius, -radius) for _axis in image.shape)
    padded[core] = image
    padded_core = padded[core]
    padded_core[~mask] = 0
    dilated = ndi.grey_dilation(padded, footprint=footprint)[core]
    result = np.asarray(dilated, dtype=np.float32)
    result[~mask] = image[~mask]
    return result


@njit(cache=True)
def _exact_level_set_convex_hull_smoothing_numba(
    image: np.ndarray,
    valid_mask: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    height, width = image.shape
    minimum = np.float32(0.0)
    maximum = np.float32(0.0)
    found_valid = False
    for y in range(height):
        for x in range(width):
            if not valid_mask[y, x]:
                continue
            value = image[y, x]
            if not found_valid:
                minimum = value
                maximum = value
                found_valid = True
            else:
                if value < minimum:
                    minimum = value
                if value > maximum:
                    maximum = value

    output = np.empty((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            output[y, x] = minimum if valid_mask[y, x] else np.float32(0.0)
    if (not found_valid) or maximum <= minimum:
        return output

    row_count2 = height * 2 + 1
    min_col_by_row = np.empty(row_count2, dtype=np.int64)
    max_col_by_row = np.empty(row_count2, dtype=np.int64)
    point_capacity = max(2, row_count2 * 2)
    point_x = np.empty(point_capacity, dtype=np.int64)
    point_y = np.empty(point_capacity, dtype=np.int64)
    hull_x = np.empty(point_capacity * 2, dtype=np.int64)
    hull_y = np.empty(point_capacity * 2, dtype=np.int64)
    for level_index in range(thresholds.size):
        threshold = thresholds[level_index]
        point_count = _collect_diamond_extreme_points(
            image,
            valid_mask,
            threshold,
            min_col_by_row,
            max_col_by_row,
            point_x,
            point_y,
        )
        if point_count == 0:
            continue
        hull_count = _monotone_chain_hull(
            point_x,
            point_y,
            point_count,
            hull_x,
            hull_y,
        )
        _paint_convex_hull(
            output,
            valid_mask,
            threshold,
            hull_x,
            hull_y,
            hull_count,
        )
    return output


@njit(cache=True)
def _collect_diamond_extreme_points(
    image: np.ndarray,
    valid_mask: np.ndarray,
    threshold: np.float32,
    min_col_by_row: np.ndarray,
    max_col_by_row: np.ndarray,
    point_x: np.ndarray,
    point_y: np.ndarray,
) -> int:
    height, width = image.shape
    row_count2 = height * 2 + 1
    for row_index in range(row_count2):
        min_col_by_row[row_index] = 9223372036854775807
        max_col_by_row[row_index] = -9223372036854775807

    for y in range(height):
        for x in range(width):
            if valid_mask[y, x] and image[y, x] >= threshold:
                _add_diamond_vertex(min_col_by_row, max_col_by_row, 2 * y - 1, 2 * x)
                _add_diamond_vertex(min_col_by_row, max_col_by_row, 2 * y + 1, 2 * x)
                _add_diamond_vertex(min_col_by_row, max_col_by_row, 2 * y, 2 * x - 1)
                _add_diamond_vertex(min_col_by_row, max_col_by_row, 2 * y, 2 * x + 1)

    point_count = 0
    for row_index in range(row_count2):
        max_col = max_col_by_row[row_index]
        if max_col < -9223372036854775800:
            continue
        row2 = row_index - 1
        min_col = min_col_by_row[row_index]
        point_x[point_count] = row2
        point_y[point_count] = min_col
        point_count += 1
        if max_col != min_col:
            point_x[point_count] = row2
            point_y[point_count] = max_col
            point_count += 1
    return point_count


@njit(cache=True)
def _add_diamond_vertex(
    min_col_by_row: np.ndarray,
    max_col_by_row: np.ndarray,
    row2: int,
    col2: int,
) -> None:
    row_index = row2 + 1
    if col2 < min_col_by_row[row_index]:
        min_col_by_row[row_index] = col2
    if col2 > max_col_by_row[row_index]:
        max_col_by_row[row_index] = col2


@njit(cache=True)
def _cross_points(
    ax: int,
    ay: int,
    bx: int,
    by: int,
    cx: int,
    cy: int,
) -> int:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


@njit(cache=True)
def _monotone_chain_hull(
    point_x: np.ndarray,
    point_y: np.ndarray,
    point_count: int,
    hull_x: np.ndarray,
    hull_y: np.ndarray,
) -> int:
    if point_count <= 1:
        if point_count == 1:
            hull_x[0] = point_x[0]
            hull_y[0] = point_y[0]
        return point_count

    hull_count = 0
    for index in range(point_count):
        px = point_x[index]
        py = point_y[index]
        while hull_count >= 2 and _cross_points(
            hull_x[hull_count - 2],
            hull_y[hull_count - 2],
            hull_x[hull_count - 1],
            hull_y[hull_count - 1],
            px,
            py,
        ) <= 0:
            hull_count -= 1
        hull_x[hull_count] = px
        hull_y[hull_count] = py
        hull_count += 1

    lower_count = hull_count
    for index in range(point_count - 2, -1, -1):
        px = point_x[index]
        py = point_y[index]
        while hull_count > lower_count and _cross_points(
            hull_x[hull_count - 2],
            hull_y[hull_count - 2],
            hull_x[hull_count - 1],
            hull_y[hull_count - 1],
            px,
            py,
        ) <= 0:
            hull_count -= 1
        hull_x[hull_count] = px
        hull_y[hull_count] = py
        hull_count += 1

    if hull_count > 1:
        hull_count -= 1
    return hull_count


@njit(cache=True)
def _paint_convex_hull(
    output: np.ndarray,
    valid_mask: np.ndarray,
    threshold: np.float32,
    hull_x: np.ndarray,
    hull_y: np.ndarray,
    hull_count: int,
) -> None:
    if hull_count <= 0:
        return
    if hull_count == 1:
        if hull_x[0] % 2 != 0 or hull_y[0] % 2 != 0:
            return
        y = hull_x[0] // 2
        x = hull_y[0] // 2
        if (
            y >= 0
            and y < valid_mask.shape[0]
            and x >= 0
            and x < valid_mask.shape[1]
            and valid_mask[y, x]
        ):
            output[y, x] = threshold
        return

    min_row2 = hull_x[0]
    max_row2 = hull_x[0]
    min_col2 = hull_y[0]
    max_col2 = hull_y[0]
    for index in range(1, hull_count):
        row2 = hull_x[index]
        col2 = hull_y[index]
        if row2 < min_row2:
            min_row2 = row2
        if row2 > max_row2:
            max_row2 = row2
        if col2 < min_col2:
            min_col2 = col2
        if col2 > max_col2:
            max_col2 = col2

    if hull_count == 2:
        _paint_line_hull(
            output,
            valid_mask,
            threshold,
            hull_x[0],
            hull_y[0],
            hull_x[1],
            hull_y[1],
            min_row2,
            max_row2,
            min_col2,
            max_col2,
        )
        return

    area2 = 0
    for index in range(hull_count):
        next_index = 0 if index == hull_count - 1 else index + 1
        area2 += hull_x[index] * hull_y[next_index]
        area2 -= hull_x[next_index] * hull_y[index]
    positive_orientation = area2 >= 0

    image_height, image_width = output.shape
    min_y = max(0, _ceil_div2(min_row2))
    max_y = min(image_height - 1, _floor_div2(max_row2))
    min_x = max(0, _ceil_div2(min_col2))
    max_x = min(image_width - 1, _floor_div2(max_col2))

    for y in range(min_y, max_y + 1):
        query_row2 = y * 2
        for x in range(min_x, max_x + 1):
            if not valid_mask[y, x]:
                continue
            query_col2 = x * 2
            inside = True
            for index in range(hull_count):
                next_index = 0 if index == hull_count - 1 else index + 1
                cross = _cross_points(
                    hull_x[index],
                    hull_y[index],
                    hull_x[next_index],
                    hull_y[next_index],
                    query_row2,
                    query_col2,
                )
                if positive_orientation:
                    if cross < 0:
                        inside = False
                        break
                elif cross > 0:
                    inside = False
                    break
            if inside:
                output[y, x] = threshold


@njit(cache=True)
def _ceil_div2(value: int) -> int:
    if value >= 0:
        return (value + 1) // 2
    return value // 2


@njit(cache=True)
def _floor_div2(value: int) -> int:
    if value >= 0:
        return value // 2
    return -((-value + 1) // 2)


@njit(cache=True)
def _paint_line_hull(
    output: np.ndarray,
    valid_mask: np.ndarray,
    threshold: np.float32,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    min_row2: int,
    max_row2: int,
    min_col2: int,
    max_col2: int,
) -> None:
    dx = x1 - x0
    dy = y1 - y0
    length2 = dx * dx + dy * dy
    if length2 == 0:
        if valid_mask[y0, x0]:
            output[y0, x0] = threshold
        return
    image_height, image_width = output.shape
    min_y = max(0, _ceil_div2(min_row2))
    max_y = min(image_height - 1, _floor_div2(max_row2))
    min_x = max(0, _ceil_div2(min_col2))
    max_x = min(image_width - 1, _floor_div2(max_col2))
    for y in range(min_y, max_y + 1):
        query_row2 = y * 2
        for x in range(min_x, max_x + 1):
            if not valid_mask[y, x]:
                continue
            query_col2 = x * 2
            dot = (query_row2 - x0) * dx + (query_col2 - y0) * dy
            if dot < 0 or dot > length2:
                continue
            cross = dx * (query_col2 - y0) - dy * (query_row2 - x0)
            if cross == 0:
                output[y, x] = threshold


@njit(cache=True)
def _rank_median_global_minimum_is_majority_everywhere_numba(
    image: np.ndarray,
    row_offsets_y: np.ndarray,
    row_radii_x: np.ndarray,
    minimum_value: np.uint16,
) -> bool:
    height, width = image.shape
    for y in range(height):
        total_count = 0
        minimum_count = 0

        for row_index in range(row_offsets_y.shape[0]):
            yy = y + row_offsets_y[row_index]
            if yy < 0 or yy >= height:
                continue
            radius_x = row_radii_x[row_index]
            right = radius_x
            if right >= width:
                right = width - 1
            for xx in range(0, right + 1):
                total_count += 1
                if image[yy, xx] == minimum_value:
                    minimum_count += 1

        if minimum_count <= total_count // 2:
            return False

        for x in range(1, width):
            for row_index in range(row_offsets_y.shape[0]):
                yy = y + row_offsets_y[row_index]
                if yy < 0 or yy >= height:
                    continue
                radius_x = row_radii_x[row_index]

                remove_x = x - 1 - radius_x
                if remove_x >= 0 and remove_x < width:
                    total_count -= 1
                    if image[yy, remove_x] == minimum_value:
                        minimum_count -= 1

                add_x = x + radius_x
                if add_x >= 0 and add_x < width:
                    total_count += 1
                    if image[yy, add_x] == minimum_value:
                        minimum_count += 1

            if minimum_count <= total_count // 2:
                return False
    return True


@njit(cache=True)
def _rank_median_codes_2d_sliding_histogram_numba(
    code_image: np.ndarray,
    row_offsets_y: np.ndarray,
    row_radii_x: np.ndarray,
    value_count: int,
) -> np.ndarray:
    height, width = code_image.shape
    output = np.empty((height, width), dtype=np.int32)
    for y in range(height):
        tree = np.zeros(value_count + 1, dtype=np.int64)
        count = 0

        for row_index in range(row_offsets_y.shape[0]):
            yy = y + row_offsets_y[row_index]
            if yy < 0 or yy >= height:
                continue
            radius_x = row_radii_x[row_index]
            right = radius_x
            if right >= width:
                right = width - 1
            for xx in range(0, right + 1):
                _fenwick_add_code(tree, int(code_image[yy, xx]), 1)
                count += 1

        if count == 0:
            output[y, 0] = 0
        else:
            output[y, 0] = _fenwick_select_code(tree, count // 2)

        for x in range(1, width):
            for row_index in range(row_offsets_y.shape[0]):
                yy = y + row_offsets_y[row_index]
                if yy < 0 or yy >= height:
                    continue
                radius_x = row_radii_x[row_index]

                remove_x = x - 1 - radius_x
                if remove_x >= 0 and remove_x < width:
                    _fenwick_add_code(tree, int(code_image[yy, remove_x]), -1)
                    count -= 1

                add_x = x + radius_x
                if add_x >= 0 and add_x < width:
                    _fenwick_add_code(tree, int(code_image[yy, add_x]), 1)
                    count += 1

            if count == 0:
                output[y, x] = 0
            else:
                output[y, x] = _fenwick_select_code(tree, count // 2)
    return output


@njit(cache=True)
def _rank_median_small_codes_2d_sliding_histogram_numba(
    code_image: np.ndarray,
    row_offsets_y: np.ndarray,
    row_radii_x: np.ndarray,
    value_count: int,
) -> np.ndarray:
    height, width = code_image.shape
    output = np.empty((height, width), dtype=np.int32)
    for y in range(height):
        histogram = np.zeros(value_count, dtype=np.int32)
        count = 0

        for row_index in range(row_offsets_y.shape[0]):
            yy = y + row_offsets_y[row_index]
            if yy < 0 or yy >= height:
                continue
            radius_x = row_radii_x[row_index]
            right = radius_x
            if right >= width:
                right = width - 1
            for xx in range(0, right + 1):
                histogram[int(code_image[yy, xx])] += 1
                count += 1

        output[y, 0] = _rank_median_select_small_code(histogram, count)

        for x in range(1, width):
            for row_index in range(row_offsets_y.shape[0]):
                yy = y + row_offsets_y[row_index]
                if yy < 0 or yy >= height:
                    continue
                radius_x = row_radii_x[row_index]

                remove_x = x - 1 - radius_x
                if remove_x >= 0 and remove_x < width:
                    histogram[int(code_image[yy, remove_x])] -= 1
                    count -= 1

                add_x = x + radius_x
                if add_x >= 0 and add_x < width:
                    histogram[int(code_image[yy, add_x])] += 1
                    count += 1

            output[y, x] = _rank_median_select_small_code(histogram, count)
    return output


@njit(cache=True)
def _rank_median_select_small_code(histogram: np.ndarray, count: int) -> int:
    target = count // 2 + 1
    cumulative = 0
    for code in range(histogram.shape[0]):
        cumulative += histogram[code]
        if cumulative >= target:
            return code
    return max(0, histogram.shape[0] - 1)


@njit(cache=True)
def _fenwick_add_code(tree: np.ndarray, code: int, delta: int) -> None:
    index = code + 1
    while index < tree.shape[0]:
        tree[index] += delta
        index += index & -index


@njit(cache=True)
def _fenwick_select_code(tree: np.ndarray, kth: int) -> int:
    index = 0
    bit = 1
    while bit < tree.shape[0]:
        bit <<= 1
    bit >>= 1
    target = kth + 1
    while bit != 0:
        next_index = index + bit
        if next_index < tree.shape[0] and tree[next_index] < target:
            index = next_index
            target -= tree[next_index]
        bit >>= 1
    return index


@njit(cache=True)
def _rank_median_uint16_2d_sliding_histogram_numba(
    image: np.ndarray,
    mask: np.ndarray,
    row_offsets_y: np.ndarray,
    row_radii_x: np.ndarray,
) -> np.ndarray:
    height, width = image.shape
    output = np.empty((height, width), dtype=np.uint16)
    histogram_size = 65536
    for y in range(height):
        tree = np.zeros(histogram_size + 1, dtype=np.int64)
        count = 0

        for row_index in range(row_offsets_y.shape[0]):
            yy = y + row_offsets_y[row_index]
            if yy < 0 or yy >= height:
                continue
            radius_x = row_radii_x[row_index]
            right = radius_x
            if right >= width:
                right = width - 1
            for xx in range(0, right + 1):
                value = image[yy, xx] if mask[yy, xx] else np.uint16(0)
                _fenwick_add_uint16(tree, value, 1)
                count += 1

        if count == 0:
            output[y, 0] = np.uint16(0)
        else:
            output[y, 0] = _fenwick_select_uint16(tree, count // 2)

        for x in range(1, width):
            for row_index in range(row_offsets_y.shape[0]):
                yy = y + row_offsets_y[row_index]
                if yy < 0 or yy >= height:
                    continue
                radius_x = row_radii_x[row_index]

                remove_x = x - 1 - radius_x
                if remove_x >= 0 and remove_x < width:
                    value = (
                        image[yy, remove_x]
                        if mask[yy, remove_x]
                        else np.uint16(0)
                    )
                    _fenwick_add_uint16(tree, value, -1)
                    count -= 1

                add_x = x + radius_x
                if add_x >= 0 and add_x < width:
                    value = image[yy, add_x] if mask[yy, add_x] else np.uint16(0)
                    _fenwick_add_uint16(tree, value, 1)
                    count += 1

            if count == 0:
                output[y, x] = np.uint16(0)
            else:
                output[y, x] = _fenwick_select_uint16(tree, count // 2)
    return output


@njit(cache=True)
def _fenwick_add_uint16(tree: np.ndarray, value: np.uint16, delta: int) -> None:
    index = int(value) + 1
    while index < tree.shape[0]:
        tree[index] += delta
        index += index & -index


@njit(cache=True)
def _fenwick_select_uint16(tree: np.ndarray, kth: int) -> np.uint16:
    index = 0
    bit = 32768
    target = kth + 1
    while bit != 0:
        next_index = index + bit
        if next_index < tree.shape[0] and tree[next_index] < target:
            index = next_index
            target -= tree[next_index]
        bit >>= 1
    return np.uint16(index)


@njit(cache=True)
def _rank_median_uint16_2d_numba(
    image: np.ndarray,
    mask: np.ndarray,
    offsets_y: np.ndarray,
    offsets_x: np.ndarray,
) -> np.ndarray:
    height, width = image.shape
    output = np.empty((height, width), dtype=np.uint16)
    footprint_size = offsets_y.shape[0]
    for y in range(height):
        values = np.empty(footprint_size, dtype=np.uint16)
        for x in range(width):
            count = 0
            for offset_index in range(footprint_size):
                yy = y + offsets_y[offset_index]
                xx = x + offsets_x[offset_index]
                if 0 <= yy < height and 0 <= xx < width:
                    values[count] = image[yy, xx] if mask[yy, xx] else 0
                    count += 1
            output[y, x] = _select_uint16(values, count, count // 2)
    return output


@njit(cache=True)
def _select_uint16(values: np.ndarray, count: int, kth: int) -> np.uint16:
    left = 0
    right = count - 1
    while True:
        if left == right:
            return values[left]
        pivot_index = (left + right) // 2
        pivot_index = _partition_uint16(values, left, right, pivot_index)
        if kth == pivot_index:
            return values[kth]
        if kth < pivot_index:
            right = pivot_index - 1
        else:
            left = pivot_index + 1


@njit(cache=True)
def _partition_uint16(
    values: np.ndarray,
    left: int,
    right: int,
    pivot_index: int,
) -> int:
    pivot_value = values[pivot_index]
    values[pivot_index] = values[right]
    values[right] = pivot_value
    store_index = left
    for index in range(left, right):
        if values[index] < pivot_value:
            current = values[store_index]
            values[store_index] = values[index]
            values[index] = current
            store_index += 1
    current = values[right]
    values[right] = values[store_index]
    values[store_index] = current
    return store_index


__all__ = public_names_from_objects(
    AutomaticSmoothingFilterSizeStrategy,
    AverageSmoothingPlaneStrategy,
    CalculationScope,
    ConvexHullSmoothingBackendStrategy,
    ConvexHullSmoothingPlaneStrategy,
    DivideIlluminationCorrectionStrategy,
    ExactLevelSetNumpyConvexHullSmoothingBackendStrategy,
    FilterSizeMethod,
    FitPolynomialSmoothingPlaneStrategy,
    GaussianFilterSmoothingPlaneStrategy,
    IlluminationCorrection,
    IlluminationCorrectionInputStack,
    IlluminationCorrectionMethod,
    IlluminationCorrectionRequest,
    IlluminationCorrectionSettingSequence,
    IlluminationCorrectionStrategy,
    IlluminationGaussianFilter,
    IlluminationMask,
    IlluminationStats,
    IntensityChoice,
    LegacyFastNumpyConvexHullSmoothingBackendStrategy,
    ManualSmoothingFilterSizeStrategy,
    MedianFilterSmoothingPlaneStrategy,
    NativeExactLevelSetNumpyConvexHullSmoothingBackendStrategy,
    NativeNumpyRankMedianSmoothingBackendStrategy,
    NoSmoothingPlaneStrategy,
    NumbaNumpyRankMedianSmoothingBackendStrategy,
    ObjectWidthSmoothingFilterSizeStrategy,
    RankMedianSmoothingBackendStrategy,
    RescaleOption,
    SmoothingFilterSizeRequest,
    SmoothingFilterSizeStrategy,
    SmoothingMethod,
    SmoothingPlaneRequest,
    SmoothingPlaneStrategy,
    SplineBgMode,
    SplinesSmoothingPlaneStrategy,
    SubtractIlluminationCorrectionStrategy,
    coerce_illumination_enum,
    correct_illumination_apply,
    correct_illumination_calculate,
    fit_polynomial_surface,
    fit_polynomial_unmasked_gram,
)
