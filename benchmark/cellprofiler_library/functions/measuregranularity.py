"""
Converted from CellProfiler: MeasureGranularity
Original: MeasureGranularity module

Measures granularity spectrum (texture size distribution) of images.
Granularity is measured by iteratively eroding the image and measuring
how much signal is lost at each scale.
"""

import logging
import numpy as np
import os
import time
from typing import Tuple, List
from dataclasses import dataclass
from functools import lru_cache
from collections import OrderedDict
import hashlib
from numba import njit, prange
from openhcs.core.memory.decorators import numpy
from openhcs.core.measurement_schemas import (
    DataclassCompanionSchema,
    DataclassFieldInsertion,
)
from openhcs.core.runtime_values import object_label_dense_array
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_outputs, special_inputs
from openhcs.processing.materialization import csv_materializer

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
class GranularityMeasurement:
    """Granularity spectrum measurements for an image."""
    slice_index: int
    gs1: float
    gs2: float
    gs3: float
    gs4: float
    gs5: float
    gs6: float
    gs7: float
    gs8: float
    gs9: float
    gs10: float
    gs11: float
    gs12: float
    gs13: float
    gs14: float
    gs15: float
    gs16: float


ObjectGranularityMeasurement = DataclassCompanionSchema(
    source_type=GranularityMeasurement,
    companion_name="ObjectGranularityMeasurement",
    insertions=(
        DataclassFieldInsertion("object_id", int, after_field="slice_index"),
    ),
    module_name=__name__,
    doc="Granularity spectrum measurements per object.",
).materialize()


@dataclass(frozen=True)
class _GranularityImageSeries:
    pixels: np.ndarray
    new_shape: np.ndarray
    reconstructions: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class _GranularityImageSeriesCacheEntry:
    series: _GranularityImageSeries


_GRANULARITY_IMAGE_SERIES_CACHE: dict[
    tuple[str, tuple[int, ...], bytes, float, float, int, int],
    _GranularityImageSeriesCacheEntry,
] = OrderedDict()
_GRANULARITY_IMAGE_SERIES_CACHE_MAX_ENTRIES = 16


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(("granularity_measurements", csv_materializer(
    fields=["slice_index", "gs1", "gs2", "gs3", "gs4", "gs5", "gs6", "gs7", "gs8",
            "gs9", "gs10", "gs11", "gs12", "gs13", "gs14", "gs15", "gs16"],
    analysis_type="granularity"
)))
def measure_granularity(
    image: np.ndarray,
    subsample_size: float = 0.25,
    background_subsample_size: float = 0.25,
    element_radius: int = 10,
    spectrum_length: int = 16,
) -> Tuple[np.ndarray, GranularityMeasurement]:
    """
    Measure granularity spectrum of an image.
    
    Granularity is a texture measurement that fits structure elements of
    increasing size into the image texture and outputs a spectrum of measures
    based on how well they fit.
    
    Args:
        image: Input grayscale image (H, W)
        subsample_size: Subsampling factor for granularity measurements (0-1)
        background_subsample_size: Subsampling factor for background reduction (0-1)
        element_radius: Radius of structuring element for background removal
        spectrum_length: Number of granular spectrum components to measure
    
    Returns:
        Tuple of (original image, granularity measurements)
    """
    series = _granularity_image_series(
        image,
        subsample_size,
        background_subsample_size,
        element_radius,
        spectrum_length,
    )
    pixels = series.pixels
    
    # Calculate granular spectrum
    startmean = np.mean(pixels)
    startmean = max(startmean, np.finfo(float).eps)
    currentmean = startmean
    gs_values = []
    
    for i, rec in enumerate(series.reconstructions):
        prevmean = currentmean
        currentmean = np.mean(rec)
        gs = (prevmean - currentmean) * 100 / startmean
        if i > 0 and gs < 0.0:
            gs = 0.0
        gs_values.append(gs)
    
    # Pad with zeros if spectrum_length < 16
    while len(gs_values) < 16:
        gs_values.append(0.0)
    
    measurement = GranularityMeasurement(
        slice_index=0,
        gs1=gs_values[0],
        gs2=gs_values[1],
        gs3=gs_values[2],
        gs4=gs_values[3],
        gs5=gs_values[4],
        gs6=gs_values[5],
        gs7=gs_values[6],
        gs8=gs_values[7],
        gs9=gs_values[8],
        gs10=gs_values[9],
        gs11=gs_values[10],
        gs12=gs_values[11],
        gs13=gs_values[12],
        gs14=gs_values[13],
        gs15=gs_values[14],
        gs16=gs_values[15],
    )
    
    return image, measurement


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(("object_granularity_measurements", csv_materializer(
    fields=["slice_index", "object_id", "gs1", "gs2", "gs3", "gs4", "gs5", "gs6", "gs7", "gs8",
            "gs9", "gs10", "gs11", "gs12", "gs13", "gs14", "gs15", "gs16"],
    analysis_type="object_granularity"
)))
def measure_granularity_objects(
    image: np.ndarray,
    labels: np.ndarray,
    subsample_size: float = 0.25,
    background_subsample_size: float = 0.25,
    element_radius: int = 10,
    spectrum_length: int = 16,
) -> Tuple[np.ndarray, List[ObjectGranularityMeasurement]]:
    """
    Measure granularity spectrum within labeled objects.
    
    Args:
        image: Input grayscale image (H, W)
        labels: Label image from segmentation (H, W)
        subsample_size: Subsampling factor for granularity measurements (0-1)
        background_subsample_size: Subsampling factor for background reduction (0-1)
        element_radius: Radius of structuring element for background removal
        spectrum_length: Number of granular spectrum components to measure
    
    Returns:
        Tuple of (original image, list of per-object granularity measurements)
    """
    total_started_at = time.perf_counter()
    orig_shape = image.shape
    phase_started_at = time.perf_counter()
    object_range = np.unique(labels[labels > 0]).astype(np.int32, copy=False)
    nobjects = int(object_range.size)
    _log_profile(
        "granularity_object_ids",
        time.perf_counter() - phase_started_at,
        function="measure_granularity_objects",
        nobjects=nobjects,
    )

    if nobjects == 0:
        return image, []

    phase_started_at = time.perf_counter()
    series = _granularity_image_series(
        image,
        subsample_size,
        background_subsample_size,
        element_radius,
        spectrum_length,
    )
    pixels = series.pixels
    new_shape = series.new_shape
    _log_profile(
        "granularity_series",
        time.perf_counter() - phase_started_at,
        function="measure_granularity_objects",
        reconstructions=len(series.reconstructions),
    )
    
    labels = object_label_dense_array(labels, dtype=np.int32)
    phase_started_at = time.perf_counter()
    label_to_index = _label_to_index_lookup_numba(object_range)
    pixel_rows, pixel_cols, pixel_object_indices, object_counts = (
        _compact_label_pixels_from_lookup_numba(labels, label_to_index)
    )
    _log_profile(
        "granularity_compact_labels",
        time.perf_counter() - phase_started_at,
        function="measure_granularity_objects",
        foreground_pixels=len(pixel_object_indices),
    )

    # Get initial means per object
    phase_started_at = time.perf_counter()
    current_means = _mean_by_compact_label_pixels_numba(
        np.asarray(image),
        pixel_rows,
        pixel_cols,
        pixel_object_indices,
        object_counts,
    )
    start_means = np.maximum(current_means, np.finfo(float).eps)
    _log_profile(
        "granularity_initial_means",
        time.perf_counter() - phase_started_at,
        function="measure_granularity_objects",
    )
    
    # Store gs values per object: shape (nobjects, spectrum_length)
    gs_per_object = np.zeros((nobjects, 16))
    
    phase_started_at = time.perf_counter()
    for gs_idx, rec in enumerate(series.reconstructions):
        prev_means = current_means.copy()

        if subsample_size < 1:
            row_scale = (
                float(new_shape[0] - 1) / float(orig_shape[0] - 1)
                if orig_shape[0] > 1
                else 0.0
            )
            col_scale = (
                float(new_shape[1] - 1) / float(orig_shape[1] - 1)
                if orig_shape[1] > 1
                else 0.0
            )
            new_means = _mean_by_compact_label_pixels_from_resampled_numba(
                np.asarray(rec),
                pixel_rows,
                pixel_cols,
                pixel_object_indices,
                object_counts,
                row_scale,
                col_scale,
            )
        else:
            new_means = _mean_by_compact_label_pixels_numba(
                np.asarray(rec),
                pixel_rows,
                pixel_cols,
                pixel_object_indices,
                object_counts,
            )
        gs_values = (prev_means - new_means) * 100 / start_means
        if gs_idx > 0:
            np.maximum(gs_values, 0.0, out=gs_values)
        gs_per_object[:, gs_idx] = gs_values
        current_means = new_means
    _log_profile(
        "granularity_spectrum_means",
        time.perf_counter() - phase_started_at,
        function="measure_granularity_objects",
    )
    
    # Create measurement objects
    phase_started_at = time.perf_counter()
    measurements = []
    for obj_idx, object_id in enumerate(object_range):
        gs = gs_per_object[obj_idx]
        measurements.append(ObjectGranularityMeasurement(
            slice_index=0,
            object_id=int(object_id),
            gs1=gs[0], gs2=gs[1], gs3=gs[2], gs4=gs[3],
            gs5=gs[4], gs6=gs[5], gs7=gs[6], gs8=gs[7],
            gs9=gs[8], gs10=gs[9], gs11=gs[10], gs12=gs[11],
            gs13=gs[12], gs14=gs[13], gs15=gs[14], gs16=gs[15],
        ))
    _log_profile(
        "granularity_rows",
        time.perf_counter() - phase_started_at,
        function="measure_granularity_objects",
        rows=len(measurements),
    )
    _log_profile(
        "granularity_total",
        time.perf_counter() - total_started_at,
        function="measure_granularity_objects",
    )
    
    return image, measurements


def _granularity_image_series(
    image: np.ndarray,
    subsample_size: float,
    background_subsample_size: float,
    element_radius: int,
    spectrum_length: int,
) -> _GranularityImageSeries:
    """Return reusable background-corrected reconstruction series for one image."""
    image_array = np.asarray(image)
    phase_started_at = time.perf_counter()
    dtype, shape, digest = _granularity_image_content_key(image_array)
    _log_profile(
        "granularity_series_key",
        time.perf_counter() - phase_started_at,
        function="measure_granularity_objects",
    )
    key = (
        dtype,
        shape,
        digest,
        float(subsample_size),
        float(background_subsample_size),
        int(element_radius),
        int(spectrum_length),
    )
    entry = _GRANULARITY_IMAGE_SERIES_CACHE.get(key)
    if entry is not None:
        _GRANULARITY_IMAGE_SERIES_CACHE.move_to_end(key)
        _log_profile(
            "granularity_series_cache_hit",
            0.0,
            function="measure_granularity_objects",
        )
        return entry.series

    phase_started_at = time.perf_counter()
    pixels, new_shape = _background_corrected_pixels(
        image_array,
        subsample_size,
        background_subsample_size,
        element_radius,
    )
    _log_profile(
        "granularity_background_correct",
        time.perf_counter() - phase_started_at,
        function="measure_granularity_objects",
        shape=tuple(int(value) for value in pixels.shape),
    )
    phase_started_at = time.perf_counter()
    reconstructions = _granularity_reconstruction_series(
        pixels,
        spectrum_length,
    )
    _log_profile(
        "granularity_reconstruction_series",
        time.perf_counter() - phase_started_at,
        function="measure_granularity_objects",
        reconstructions=len(reconstructions),
    )
    series = _GranularityImageSeries(
        pixels=pixels,
        new_shape=new_shape,
        reconstructions=reconstructions,
    )
    _GRANULARITY_IMAGE_SERIES_CACHE[key] = _GranularityImageSeriesCacheEntry(
        series=series,
    )
    _GRANULARITY_IMAGE_SERIES_CACHE.move_to_end(key)
    while len(_GRANULARITY_IMAGE_SERIES_CACHE) > _GRANULARITY_IMAGE_SERIES_CACHE_MAX_ENTRIES:
        _GRANULARITY_IMAGE_SERIES_CACHE.popitem(last=False)
    return series


def _granularity_image_content_key(
    image: np.ndarray,
) -> tuple[str, tuple[int, ...], bytes]:
    """Return an exact value key for deterministic granularity series reuse."""
    contiguous = np.ascontiguousarray(image)
    digest = hashlib.blake2b(contiguous.view(np.uint8), digest_size=16).digest()
    return str(contiguous.dtype), tuple(int(value) for value in contiguous.shape), digest


def _prepare_granularity_backend() -> None:
    """Compile Numba kernels used by the granularity backend before execution."""
    image = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape((64, 64))
    labels = np.zeros((64, 64), dtype=np.int32)
    labels[8:24, 8:24] = 1
    labels[32:56, 32:56] = 2
    measure_granularity.__wrapped__(
        image,
        subsample_size=1.0,
        background_subsample_size=0.25,
        element_radius=10,
        spectrum_length=5,
    )
    measure_granularity_objects.__wrapped__(
        image,
        labels,
        subsample_size=1.0,
        background_subsample_size=0.25,
        element_radius=10,
        spectrum_length=5,
    )


def _granularity_reconstruction_series(
    pixels: np.ndarray,
    spectrum_length: int,
) -> tuple[np.ndarray, ...]:
    """Compute the erosion/reconstruction images shared across object sets."""
    ero = pixels.copy()
    cross_offsets = _disk_offsets(1)
    reconstructions = []
    erosion_seconds = 0.0
    reconstruction_seconds = 0.0
    for index in range(int(spectrum_length)):
        phase_started_at = time.perf_counter()
        ero = _gray_erosion_offsets_reflect_numba(ero, cross_offsets)
        erosion_seconds += time.perf_counter() - phase_started_at
        phase_started_at = time.perf_counter()
        reconstruction = _reconstruct_dilation_cross_numba(ero, pixels)
        reconstruction_seconds += time.perf_counter() - phase_started_at
        _log_profile(
            "granularity_reconstruction_iteration",
            time.perf_counter() - phase_started_at,
            function="measure_granularity_objects",
            iteration=index + 1,
            shape=tuple(int(value) for value in pixels.shape),
        )
        reconstructions.append(reconstruction)
    _log_profile(
        "granularity_reconstruction_erosion_total",
        erosion_seconds,
        function="measure_granularity_objects",
        reconstructions=len(reconstructions),
    )
    _log_profile(
        "granularity_reconstruction_dilation_total",
        reconstruction_seconds,
        function="measure_granularity_objects",
        reconstructions=len(reconstructions),
    )
    return tuple(reconstructions)


def _background_corrected_pixels(
    image: np.ndarray,
    subsample_size: float,
    background_subsample_size: float,
    element_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return CP-style background-subtracted granularity pixels."""
    image = np.asarray(image, dtype=np.float64)
    orig_shape = image.shape

    if subsample_size < 1:
        new_shape = (np.asarray(orig_shape) * subsample_size).astype(np.int64)
        new_shape = np.maximum(new_shape, 1)
        pixels = _resample_bilinear_numba(
            image,
            int(new_shape[0]),
            int(new_shape[1]),
            1.0 / float(subsample_size),
            1.0 / float(subsample_size),
        )
    else:
        pixels = image.copy()
        new_shape = np.asarray(orig_shape, dtype=np.int64)

    if background_subsample_size < 1:
        back_shape = (new_shape * background_subsample_size).astype(np.int64)
        back_shape = np.maximum(back_shape, 1)
        back_pixels = _resample_bilinear_numba(
            pixels,
            int(back_shape[0]),
            int(back_shape[1]),
            1.0 / float(background_subsample_size),
            1.0 / float(background_subsample_size),
        )
    else:
        back_pixels = pixels.copy()
        back_shape = new_shape

    footprint_offsets = _disk_offsets(element_radius)
    back_pixels = _gray_erosion_offsets_reflect_numba(back_pixels, footprint_offsets)
    back_pixels = _gray_dilation_offsets_reflect_numba(back_pixels, footprint_offsets)

    if background_subsample_size < 1:
        row_scale = (
            float(back_shape[0] - 1) / float(new_shape[0] - 1)
            if new_shape[0] > 1
            else 0.0
        )
        col_scale = (
            float(back_shape[1] - 1) / float(new_shape[1] - 1)
            if new_shape[1] > 1
            else 0.0
        )
        back_pixels = _resample_bilinear_numba(
            back_pixels,
            int(new_shape[0]),
            int(new_shape[1]),
            row_scale,
            col_scale,
        )

    pixels = pixels - back_pixels
    _clip_negative_inplace_numba(pixels)
    return pixels, new_shape


@lru_cache(maxsize=None)
def _disk_offsets(radius: int) -> np.ndarray:
    """Return integer offsets for skimage.morphology.disk(radius)."""
    radius = int(radius)
    coords = []
    radius_sq = radius * radius
    for row in range(-radius, radius + 1):
        for col in range(-radius, radius + 1):
            if row * row + col * col <= radius_sq:
                coords.append((row, col))
    return np.asarray(coords, dtype=np.int32)


@njit(cache=True, parallel=True)
def _resample_bilinear_numba(
    image: np.ndarray,
    output_height: int,
    output_width: int,
    row_scale: float,
    col_scale: float,
) -> np.ndarray:
    result = np.empty((output_height, output_width), dtype=np.float64)
    for row in prange(output_height):
        sample_row = row * row_scale
        for col in range(output_width):
            result[row, col] = _bilinear_sample_numba(
                image,
                sample_row,
                col * col_scale,
            )
    return result


@njit(cache=True, parallel=True)
def _clip_negative_inplace_numba(image: np.ndarray) -> None:
    height, width = image.shape
    for row in prange(height):
        for col in range(width):
            if image[row, col] < 0.0:
                image[row, col] = 0.0


@njit(cache=True, parallel=True)
def _gray_erosion_offsets_reflect_numba(
    image: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    height, width = image.shape
    result = np.empty((height, width), dtype=np.float64)
    for row in prange(height):
        for col in range(width):
            best = np.inf
            for offset_index in range(offsets.shape[0]):
                sample_row = _reflect_index(row + int(offsets[offset_index, 0]), height)
                sample_col = _reflect_index(col + int(offsets[offset_index, 1]), width)
                value = float(image[sample_row, sample_col])
                if value < best:
                    best = value
            result[row, col] = best
    return result


@njit(cache=True, parallel=True)
def _gray_dilation_offsets_reflect_numba(
    image: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    height, width = image.shape
    result = np.empty((height, width), dtype=np.float64)
    for row in prange(height):
        for col in range(width):
            best = -np.inf
            for offset_index in range(offsets.shape[0]):
                sample_row = _reflect_index(row + int(offsets[offset_index, 0]), height)
                sample_col = _reflect_index(col + int(offsets[offset_index, 1]), width)
                value = float(image[sample_row, sample_col])
                if value > best:
                    best = value
            result[row, col] = best
    return result


@njit(cache=True)
def _reflect_index(index: int, size: int) -> int:
    if size <= 1:
        return 0
    while index < 0 or index >= size:
        if index < 0:
            index = -index - 1
        else:
            index = 2 * size - index - 1
    return index


@njit(cache=True)
def _reconstruct_dilation_cross_numba(
    seed: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Morphological reconstruction by dilation with disk(1) connectivity."""
    height, width = seed.shape
    result = seed.copy()

    for row in range(height):
        for col in range(width):
            value = result[row, col]
            if row > 0 and result[row - 1, col] > value:
                value = result[row - 1, col]
            if col > 0 and result[row, col - 1] > value:
                value = result[row, col - 1]
            mask_value = mask[row, col]
            if value > mask_value:
                value = mask_value
            result[row, col] = value

    for row in range(height - 1, -1, -1):
        for col in range(width - 1, -1, -1):
            value = result[row, col]
            if row + 1 < height and result[row + 1, col] > value:
                value = result[row + 1, col]
            if col + 1 < width and result[row, col + 1] > value:
                value = result[row, col + 1]
            mask_value = mask[row, col]
            if value > mask_value:
                value = mask_value
            result[row, col] = value

    total_pixels = height * width
    queue_rows = np.empty(total_pixels, dtype=np.int64)
    queue_cols = np.empty(total_pixels, dtype=np.int64)
    queued = np.zeros((height, width), dtype=np.bool_)
    head = 0
    tail = 0
    queue_count = 0

    for row in range(height):
        for col in range(width):
            value = result[row, col]
            if row > 0:
                tail, queue_count = _enqueue_reconstruct_neighbor(
                    result,
                    mask,
                    queued,
                    queue_rows,
                    queue_cols,
                    tail,
                    queue_count,
                    row - 1,
                    col,
                    value,
                )
            if row + 1 < height:
                tail, queue_count = _enqueue_reconstruct_neighbor(
                    result,
                    mask,
                    queued,
                    queue_rows,
                    queue_cols,
                    tail,
                    queue_count,
                    row + 1,
                    col,
                    value,
                )
            if col > 0:
                tail, queue_count = _enqueue_reconstruct_neighbor(
                    result,
                    mask,
                    queued,
                    queue_rows,
                    queue_cols,
                    tail,
                    queue_count,
                    row,
                    col - 1,
                    value,
                )
            if col + 1 < width:
                tail, queue_count = _enqueue_reconstruct_neighbor(
                    result,
                    mask,
                    queued,
                    queue_rows,
                    queue_cols,
                    tail,
                    queue_count,
                    row,
                    col + 1,
                    value,
                )

    while queue_count > 0:
        row = queue_rows[head]
        col = queue_cols[head]
        head += 1
        if head == total_pixels:
            head = 0
        queue_count -= 1
        queued[row, col] = False
        value = result[row, col]
        if row > 0:
            tail, queue_count = _propagate_reconstruct_neighbor(
                result,
                mask,
                queued,
                queue_rows,
                queue_cols,
                tail,
                queue_count,
                row - 1,
                col,
                value,
            )
        if row + 1 < height:
            tail, queue_count = _propagate_reconstruct_neighbor(
                result,
                mask,
                queued,
                queue_rows,
                queue_cols,
                tail,
                queue_count,
                row + 1,
                col,
                value,
            )
        if col > 0:
            tail, queue_count = _propagate_reconstruct_neighbor(
                result,
                mask,
                queued,
                queue_rows,
                queue_cols,
                tail,
                queue_count,
                row,
                col - 1,
                value,
            )
        if col + 1 < width:
            tail, queue_count = _propagate_reconstruct_neighbor(
                result,
                mask,
                queued,
                queue_rows,
                queue_cols,
                tail,
                queue_count,
                row,
                col + 1,
                value,
            )

    return result


@njit(cache=True)
def _enqueue_reconstruct_neighbor(
    result: np.ndarray,
    mask: np.ndarray,
    queued: np.ndarray,
    queue_rows: np.ndarray,
    queue_cols: np.ndarray,
    tail: int,
    queue_count: int,
    row: int,
    col: int,
    source_value: float,
) -> tuple[int, int]:
    if (
        result[row, col] < source_value
        and result[row, col] < mask[row, col]
        and not queued[row, col]
    ):
        queue_rows[tail] = row
        queue_cols[tail] = col
        queued[row, col] = True
        tail += 1
        if tail == queue_rows.shape[0]:
            tail = 0
        queue_count += 1
    return tail, queue_count


@njit(cache=True)
def _propagate_reconstruct_neighbor(
    result: np.ndarray,
    mask: np.ndarray,
    queued: np.ndarray,
    queue_rows: np.ndarray,
    queue_cols: np.ndarray,
    tail: int,
    queue_count: int,
    row: int,
    col: int,
    source_value: float,
) -> tuple[int, int]:
    if result[row, col] >= source_value or result[row, col] >= mask[row, col]:
        return tail, queue_count
    new_value = source_value
    if new_value > mask[row, col]:
        new_value = mask[row, col]
    if new_value <= result[row, col]:
        return tail, queue_count
    result[row, col] = new_value
    if not queued[row, col]:
        queue_rows[tail] = row
        queue_cols[tail] = col
        queued[row, col] = True
        tail += 1
        if tail == queue_rows.shape[0]:
            tail = 0
        queue_count += 1
    return tail, queue_count


@njit(cache=True)
def _label_to_index_lookup_numba(object_ids: np.ndarray) -> np.ndarray:
    """Return dense label-id to object-index lookup for repeated scans."""
    max_label = 0
    for index in range(len(object_ids)):
        object_id = int(object_ids[index])
        if object_id > max_label:
            max_label = object_id
    label_to_index = np.full(max_label + 1, -1, dtype=np.int64)
    for index in range(len(object_ids)):
        object_id = int(object_ids[index])
        if object_id > 0:
            label_to_index[object_id] = index
    return label_to_index


@njit(cache=True)
def _label_counts_from_lookup_numba(
    labels: np.ndarray,
    label_to_index: np.ndarray,
) -> np.ndarray:
    """Count object pixels once for all granularity spectrum iterations."""
    object_count = 0
    for label_id in range(len(label_to_index)):
        index = int(label_to_index[label_id])
        if index >= object_count:
            object_count = index + 1
    object_counts = np.zeros(object_count, dtype=np.int64)
    max_label = len(label_to_index) - 1
    height, width = labels.shape
    for row in range(height):
        for col in range(width):
            label_id = int(labels[row, col])
            if label_id <= 0 or label_id > max_label:
                continue
            index = int(label_to_index[label_id])
            if index >= 0:
                object_counts[index] += 1
    return object_counts


@njit(cache=True)
def _compact_label_pixels_from_lookup_numba(
    labels: np.ndarray,
    label_to_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return foreground label pixels once so spectrum iterations skip background."""
    object_count = 0
    for label_id in range(len(label_to_index)):
        index = int(label_to_index[label_id])
        if index >= object_count:
            object_count = index + 1

    max_label = len(label_to_index) - 1
    height, width = labels.shape
    foreground_count = 0
    object_counts = np.zeros(object_count, dtype=np.int64)
    for row in range(height):
        for col in range(width):
            label_id = int(labels[row, col])
            if label_id <= 0 or label_id > max_label:
                continue
            index = int(label_to_index[label_id])
            if index < 0:
                continue
            foreground_count += 1
            object_counts[index] += 1

    pixel_rows = np.empty(foreground_count, dtype=np.int64)
    pixel_cols = np.empty(foreground_count, dtype=np.int64)
    pixel_object_indices = np.empty(foreground_count, dtype=np.int64)
    out_index = 0
    for row in range(height):
        for col in range(width):
            label_id = int(labels[row, col])
            if label_id <= 0 or label_id > max_label:
                continue
            index = int(label_to_index[label_id])
            if index < 0:
                continue
            pixel_rows[out_index] = row
            pixel_cols[out_index] = col
            pixel_object_indices[out_index] = index
            out_index += 1
    return pixel_rows, pixel_cols, pixel_object_indices, object_counts


@njit(cache=True)
def _mean_by_compact_label_pixels_numba(
    image: np.ndarray,
    pixel_rows: np.ndarray,
    pixel_cols: np.ndarray,
    pixel_object_indices: np.ndarray,
    object_counts: np.ndarray,
) -> np.ndarray:
    """Return object means by scanning only cached foreground label pixels."""
    object_count = len(object_counts)
    means = np.empty(object_count, dtype=np.float64)
    sums = np.zeros(object_count, dtype=np.float64)
    for pixel_index in range(len(pixel_object_indices)):
        object_index = int(pixel_object_indices[pixel_index])
        sums[object_index] += float(
            image[int(pixel_rows[pixel_index]), int(pixel_cols[pixel_index])]
        )

    for index in range(object_count):
        count = object_counts[index]
        means[index] = sums[index] / count if count > 0 else np.nan
    return means


@njit(cache=True)
def _mean_by_label_with_counts_numba(
    image: np.ndarray,
    labels: np.ndarray,
    label_to_index: np.ndarray,
    object_counts: np.ndarray,
) -> np.ndarray:
    """Return means using precomputed label lookup and pixel counts."""
    object_count = len(object_counts)
    means = np.empty(object_count, dtype=np.float64)
    sums = np.zeros(object_count, dtype=np.float64)
    max_label = len(label_to_index) - 1
    height, width = labels.shape
    for row in range(height):
        for col in range(width):
            label_id = int(labels[row, col])
            if label_id <= 0 or label_id > max_label:
                continue
            index = int(label_to_index[label_id])
            if index >= 0:
                sums[index] += float(image[row, col])

    for index in range(object_count):
        count = object_counts[index]
        means[index] = sums[index] / count if count > 0 else np.nan
    return means


@njit(cache=True)
def _mean_by_label_numba(
    image: np.ndarray,
    labels: np.ndarray,
    object_ids: np.ndarray,
) -> np.ndarray:
    """Return means for explicit label IDs without SciPy object loops."""
    object_count = len(object_ids)
    means = np.empty(object_count, dtype=np.float64)
    for index in range(object_count):
        means[index] = np.nan
    if object_count == 0:
        return means

    max_label = 0
    for index in range(object_count):
        object_id = int(object_ids[index])
        if object_id > max_label:
            max_label = object_id
    label_to_index = np.full(max_label + 1, -1, dtype=np.int64)
    for index in range(object_count):
        object_id = int(object_ids[index])
        if object_id > 0:
            label_to_index[object_id] = index

    sums = np.zeros(object_count, dtype=np.float64)
    counts = np.zeros(object_count, dtype=np.int64)
    height, width = labels.shape
    for row in range(height):
        for col in range(width):
            label_id = int(labels[row, col])
            if label_id <= 0 or label_id > max_label:
                continue
            index = label_to_index[label_id]
            if index < 0:
                continue
            sums[index] += float(image[row, col])
            counts[index] += 1

    for index in range(object_count):
        if counts[index] > 0:
            means[index] = sums[index] / counts[index]
    return means


def _mean_by_label_from_resampled_numba(
    image: np.ndarray,
    labels: np.ndarray,
    object_ids: np.ndarray,
    row_scale: float,
    col_scale: float,
) -> np.ndarray:
    """Compatibility wrapper using precomputed lookup/count kernels."""
    label_to_index = _label_to_index_lookup_numba(object_ids)
    object_counts = _label_counts_from_lookup_numba(labels, label_to_index)
    return _mean_by_label_from_resampled_with_counts_numba(
        image,
        labels,
        label_to_index,
        object_counts,
        row_scale,
        col_scale,
    )


@njit(cache=True)
def _mean_by_compact_label_pixels_from_resampled_numba(
    image: np.ndarray,
    pixel_rows: np.ndarray,
    pixel_cols: np.ndarray,
    pixel_object_indices: np.ndarray,
    object_counts: np.ndarray,
    row_scale: float,
    col_scale: float,
) -> np.ndarray:
    """Mean resampled values using cached foreground label pixels only."""
    object_count = len(object_counts)
    means = np.empty(object_count, dtype=np.float64)
    sums = np.zeros(object_count, dtype=np.float64)
    for pixel_index in range(len(pixel_object_indices)):
        object_index = int(pixel_object_indices[pixel_index])
        sums[object_index] += _bilinear_sample_numba(
            image,
            float(pixel_rows[pixel_index]) * row_scale,
            float(pixel_cols[pixel_index]) * col_scale,
        )

    for index in range(object_count):
        count = object_counts[index]
        means[index] = sums[index] / count if count > 0 else np.nan
    return means


@njit(cache=True)
def _mean_by_label_from_resampled_with_counts_numba(
    image: np.ndarray,
    labels: np.ndarray,
    label_to_index: np.ndarray,
    object_counts: np.ndarray,
    row_scale: float,
    col_scale: float,
) -> np.ndarray:
    """Mean label values after order-1 coordinate sampling, without materializing."""
    object_count = len(object_counts)
    means = np.empty(object_count, dtype=np.float64)
    if object_count == 0:
        return means

    sums = np.zeros(object_count, dtype=np.float64)
    max_label = len(label_to_index) - 1
    height, width = labels.shape
    for row in range(height):
        sample_row = row * row_scale
        for col in range(width):
            label_id = int(labels[row, col])
            if label_id <= 0 or label_id > max_label:
                continue
            index = label_to_index[label_id]
            if index < 0:
                continue
            sums[index] += _bilinear_sample_numba(image, sample_row, col * col_scale)

    for index in range(object_count):
        count = object_counts[index]
        means[index] = sums[index] / count if count > 0 else np.nan
    return means


@njit(cache=True)
def _bilinear_sample_numba(
    image: np.ndarray,
    row_coord: float,
    col_coord: float,
) -> float:
    height, width = image.shape
    row0 = int(np.floor(row_coord))
    col0 = int(np.floor(col_coord))
    if row0 < 0:
        row0 = 0
    if col0 < 0:
        col0 = 0
    if row0 >= height:
        row0 = height - 1
    if col0 >= width:
        col0 = width - 1
    row1 = row0 + 1
    col1 = col0 + 1
    if row1 >= height:
        row1 = row0
    if col1 >= width:
        col1 = col0

    row_weight = row_coord - row0
    col_weight = col_coord - col0
    top = (
        float(image[row0, col0]) * (1.0 - col_weight)
        + float(image[row0, col1]) * col_weight
    )
    bottom = (
        float(image[row1, col0]) * (1.0 - col_weight)
        + float(image[row1, col1]) * col_weight
    )
    return top * (1.0 - row_weight) + bottom * row_weight


measure_granularity.__openhcs_prepare__ = _prepare_granularity_backend
measure_granularity_objects.__openhcs_prepare__ = _prepare_granularity_backend
