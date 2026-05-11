"""
Converted from CellProfiler: CorrectIlluminationCalculate
Calculates an illumination correction function to correct uneven illumination/lighting/shading.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import time

import numpy as np

from openhcs.core.memory.decorators import numpy
from openhcs.core.runtime_values import (
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    image_payload_with_context,
    project_image_mask_to_data_domain,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
)
from openhcs.processing.backends.cellprofiler.illumination import (
    CalculationScope,
    FilterSizeMethod,
    IlluminationGaussianFilter,
    IlluminationMask,
    IntensityChoice,
    RescaleOption,
    SmoothingFilterSizeRequest,
    SmoothingFilterSizeStrategy,
    SmoothingMethod,
    SmoothingPlaneRequest,
    SmoothingPlaneStrategy,
    SplineBgMode,
)
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.processing.materialization import csv_materializer

from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum

_PROFILE_RUNTIME_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME"
logger = logging.getLogger(__name__)


def _profile_enabled() -> bool:
    return os.environ.get(_PROFILE_RUNTIME_ENV, "").lower() in {"1", "true", "yes"}


def _log_profile(label: str, seconds: float, **fields: object) -> None:
    if not _profile_enabled():
        return
    field_text = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("RUNTIME_PROFILE %s %.6fs %s", label, seconds, field_text)


@dataclass
class IlluminationStats:
    slice_index: int
    min_value: float
    max_value: float
    mean_value: float
    calculation_type: str
    smoothing_method: str


ROBUST_FACTOR = 0.02


def _blockwise_background_minimum(
    pixel_data: np.ndarray,
    mask: np.ndarray | None,
    block_size: int,
    morphology: MorphologyBackendStrategy,
) -> np.ndarray:
    return morphology.blockwise_minimum(pixel_data, mask, block_size)


def _preprocess_for_averaging(
    pixel_data: np.ndarray,
    mask: np.ndarray | None,
    intensity_choice: IntensityChoice,
    smoothing_method: SmoothingMethod,
    block_size: int,
    morphology: MorphologyBackendStrategy,
) -> np.ndarray:
    """Create a version of the image appropriate for averaging."""
    if (
        intensity_choice == IntensityChoice.REGULAR
        or smoothing_method == SmoothingMethod.SPLINES
    ):
        result = pixel_data.copy()
        if mask is not None:
            result[~mask] = 0
        return result
    else:  # BACKGROUND
        return _blockwise_background_minimum(
            pixel_data,
            mask,
            block_size,
            morphology,
        )


def _is_multi_image_stack(
    pixel_data: np.ndarray,
    calculation_scope: CalculationScope | None = None,
) -> bool:
    del calculation_scope
    return pixel_data.ndim >= 3


def _spatial_image_shape(
    pixel_data: np.ndarray,
    calculation_scope: CalculationScope,
) -> tuple[int, ...]:
    if (
        calculation_scope.uses_all_images
        and _is_multi_image_stack(pixel_data, calculation_scope)
    ):
        return tuple(pixel_data.shape[1:])
    return tuple(pixel_data.shape)


def _illumination_average_image(
    pixel_data: np.ndarray,
    mask: np.ndarray | None,
    intensity_choice: IntensityChoice,
    smoothing_method: SmoothingMethod,
    block_size: int,
    morphology: MorphologyBackendStrategy,
    calculation_scope: CalculationScope,
) -> np.ndarray:
    if not (
        calculation_scope.uses_all_images
        and _is_multi_image_stack(pixel_data, calculation_scope)
    ):
        return _preprocess_for_averaging(
            pixel_data,
            mask,
            intensity_choice,
            smoothing_method,
            block_size,
            morphology,
        )

    if pixel_data.shape[0] == 0:
        raise ValueError("All-image illumination calculation requires at least one image.")

    averaged_inputs = [
        _preprocess_for_averaging(
            np.asarray(slice_data),
            IlluminationMask(mask, pixel_data).for_stack_slice(slice_index),
            intensity_choice,
            smoothing_method,
            block_size,
            morphology,
        )
        for slice_index, slice_data in enumerate(pixel_data)
    ]
    return np.mean(np.stack(averaged_inputs, axis=0), axis=0)


def _apply_dilation(
    pixel_data: np.ndarray,
    mask: np.ndarray | None,
    dilate: bool,
    dilation_radius: int,
) -> np.ndarray:
    """Apply dilation using Gaussian convolution."""
    if not dilate:
        return pixel_data

    result = IlluminationGaussianFilter(pixel_data, mask, dilation_radius).apply()
    if mask is not None:
        result[~mask] = 0
    return result


def _apply_scaling(
    pixel_data: np.ndarray,
    mask: np.ndarray | None,
    rescale_option: RescaleOption,
) -> np.ndarray:
    """Rescale the illumination function."""
    if rescale_option == RescaleOption.NO:
        return pixel_data
    mask = project_image_mask_to_data_domain(mask, pixel_data)
    if mask is not None:
        sorted_data = pixel_data[(pixel_data > 0) & mask]
    else:
        sorted_data = pixel_data[pixel_data > 0]
    
    if sorted_data.size == 0:
        return pixel_data
    
    if rescale_option == RescaleOption.YES:
        idx = int(len(sorted_data) * ROBUST_FACTOR)
        robust_minimum = np.partition(sorted_data, idx)[idx]
        result = pixel_data.copy()
        result[result < robust_minimum] = robust_minimum
    else:  # MEDIAN
        idx = len(sorted_data) // 2
        robust_minimum = np.partition(sorted_data, idx)[idx]
        result = pixel_data.copy()
    
    if robust_minimum == 0:
        return result
    
    return result / robust_minimum


def _calculate_illumination_from_pixels(
    pixel_data: np.ndarray,
    *,
    mask: np.ndarray | None,
    intensity_choice: IntensityChoice,
    dilate_objects: bool,
    object_dilation_radius: int,
    block_size: int,
    rescale_option: RescaleOption,
    smoothing_method: SmoothingMethod,
    filter_size_method: FilterSizeMethod,
    object_width: int,
    manual_filter_size: int,
    automatic_splines: bool,
    spline_bg_mode: SplineBgMode,
    spline_points: int,
    spline_threshold: float,
    spline_rescale: float,
    spline_max_iterations: int,
    spline_convergence: float,
    calculation_scope: CalculationScope,
    morphology: MorphologyBackendStrategy,
    convex_hull_backend_provider: CellProfilerBackendProvider | None,
    rank_median_backend_provider: CellProfilerBackendProvider | None,
    slice_index: int = 0,
) -> tuple[np.ndarray, IlluminationStats]:
    total_started_at = time.perf_counter()
    phase_started_at = time.perf_counter()
    filter_size = SmoothingFilterSizeStrategy.for_method(
        filter_size_method,
    ).calculate(
        SmoothingFilterSizeRequest(
            image_shape=_spatial_image_shape(pixel_data, calculation_scope),
            object_width=object_width,
            manual_filter_size=manual_filter_size,
        )
    )
    _log_profile(
        "cic_filter_size",
        time.perf_counter() - phase_started_at,
        function="correct_illumination_calculate",
        method=filter_size_method.value,
        smoothing=smoothing_method.value,
    )

    phase_started_at = time.perf_counter()
    avg_image = _illumination_average_image(
        pixel_data,
        mask,
        intensity_choice,
        smoothing_method,
        block_size,
        morphology,
        calculation_scope,
    )
    _log_profile(
        "cic_average_image",
        time.perf_counter() - phase_started_at,
        function="correct_illumination_calculate",
        method=intensity_choice.value,
        scope=calculation_scope.value,
    )
    phase_started_at = time.perf_counter()
    dilated_image = _apply_dilation(avg_image, mask, dilate_objects, object_dilation_radius)
    _log_profile(
        "cic_dilation",
        time.perf_counter() - phase_started_at,
        function="correct_illumination_calculate",
        enabled=dilate_objects,
    )
    smoothing_request = SmoothingPlaneRequest(
        pixel_data=dilated_image,
        mask=mask,
        smoothing_method=smoothing_method,
        filter_size=filter_size,
        spline_bg_mode=spline_bg_mode,
        spline_points=spline_points,
        spline_threshold=spline_threshold,
        spline_rescale=spline_rescale,
        spline_max_iterations=spline_max_iterations,
        spline_convergence=spline_convergence,
        automatic_splines=automatic_splines,
        morphology=morphology,
        convex_hull_backend_provider=convex_hull_backend_provider,
        rank_median_backend_provider=rank_median_backend_provider,
    )
    phase_started_at = time.perf_counter()
    smoothed_image = SmoothingPlaneStrategy.for_method(smoothing_method).smooth(
        smoothing_request
    )
    _log_profile(
        "cic_smoothing",
        time.perf_counter() - phase_started_at,
        function="correct_illumination_calculate",
        method=smoothing_method.value,
    )

    phase_started_at = time.perf_counter()
    output_image = _apply_scaling(smoothed_image, mask, rescale_option).astype(np.float32)
    _log_profile(
        "cic_scaling",
        time.perf_counter() - phase_started_at,
        function="correct_illumination_calculate",
        method=rescale_option.value,
    )
    phase_started_at = time.perf_counter()
    stats = IlluminationStats(
        slice_index=slice_index,
        min_value=float(np.min(output_image)),
        max_value=float(np.max(output_image)),
        mean_value=float(np.mean(output_image)),
        calculation_type=intensity_choice.value,
        smoothing_method=smoothing_method.value
    )
    _log_profile(
        "cic_stats",
        time.perf_counter() - phase_started_at,
        function="correct_illumination_calculate",
    )
    _log_profile(
        "cic_total",
        time.perf_counter() - total_started_at,
        function="correct_illumination_calculate",
    )
    return output_image, stats


@numpy(contract=ProcessingContract.FLEXIBLE)
@special_outputs(("illumination_stats", csv_materializer(
    fields=["slice_index", "min_value", "max_value", "mean_value", "calculation_type", "smoothing_method"],
    analysis_type="illumination_correction"
)))
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
    """
    Calculate an illumination correction function.
    
    This function calculates an illumination function that can be used to correct
    uneven illumination/lighting/shading in images.
    
    Args:
        image: Input image (H, W)
        intensity_choice: Method for calculating illumination function (REGULAR or BACKGROUND)
        dilate_objects: Whether to dilate objects in the averaged image
        object_dilation_radius: Radius for object dilation
        block_size: Block size for background method
        rescale_option: How to rescale the illumination function
        smoothing_method: Method for smoothing the illumination function
        filter_size_method: How to calculate smoothing filter size
        object_width: Approximate object diameter for filter size calculation
        manual_filter_size: Manual smoothing filter size
        automatic_splines: Whether to automatically calculate spline parameters
        spline_bg_mode: Background mode for spline fitting
        spline_points: Number of spline control points
        spline_threshold: Background threshold for splines
        spline_rescale: Image resampling factor for splines
        spline_max_iterations: Maximum iterations for spline fitting
        spline_convergence: Convergence criterion for splines
        calculation_scope: Calculate each image independently or from all images
        convex_hull_backend_provider: Optional OpenHCS CellProfiler illumination
            backend provider. ``None`` uses the registered default provider.
        rank_median_backend_provider: Optional OpenHCS CellProfiler rank-median
            smoothing backend provider. ``None`` uses the registered default
            provider.
    
    Returns:
        Tuple of (illumination_function, stats)
    """
    intensity_choice = coerce_cellprofiler_enum(IntensityChoice, intensity_choice)
    rescale_option = coerce_cellprofiler_enum(RescaleOption, rescale_option)
    smoothing_method = coerce_cellprofiler_enum(SmoothingMethod, smoothing_method)
    filter_size_method = coerce_cellprofiler_enum(FilterSizeMethod, filter_size_method)
    spline_bg_mode = coerce_cellprofiler_enum(SplineBgMode, spline_bg_mode)
    calculation_scope = coerce_cellprofiler_enum(CalculationScope, calculation_scope)
    from openhcs.processing.backends.cellprofiler.morphology import (
        MorphologyBackendStrategy,
    )

    morphology = MorphologyBackendStrategy.for_callable(
        correct_illumination_calculate,
    )

    pixel_data = np.asarray(image_payload_data(image))
    illumination_mask = IlluminationMask(image_payload_mask(image), pixel_data)
    mask = illumination_mask.normalized
    metadata = image_payload_metadata(image).without_unit_interval_intensity_scale()
    common_kwargs = {
        "intensity_choice": intensity_choice,
        "dilate_objects": dilate_objects,
        "object_dilation_radius": object_dilation_radius,
        "block_size": block_size,
        "rescale_option": rescale_option,
        "smoothing_method": smoothing_method,
        "filter_size_method": filter_size_method,
        "object_width": object_width,
        "manual_filter_size": manual_filter_size,
        "automatic_splines": automatic_splines,
        "spline_bg_mode": spline_bg_mode,
        "spline_points": spline_points,
        "spline_threshold": spline_threshold,
        "spline_rescale": spline_rescale,
        "spline_max_iterations": spline_max_iterations,
        "spline_convergence": spline_convergence,
        "morphology": morphology,
        "convex_hull_backend_provider": convex_hull_backend_provider,
        "rank_median_backend_provider": rank_median_backend_provider,
    }

    if (
        _is_multi_image_stack(pixel_data, calculation_scope)
        and not calculation_scope.uses_all_images
    ):
        slice_results = [
            _calculate_illumination_from_pixels(
                np.asarray(slice_data),
                mask=illumination_mask.for_stack_slice(slice_index),
                calculation_scope=CalculationScope.EACH,
                slice_index=slice_index,
                **common_kwargs,
            )
            for slice_index, slice_data in enumerate(pixel_data)
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

    illumination, stats = _calculate_illumination_from_pixels(
        pixel_data,
        mask=mask,
        calculation_scope=calculation_scope,
        **common_kwargs,
    )
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
