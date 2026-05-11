"""Granularity numerics for CellProfiler-compatible texture measurement."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import logging
import os
import time

import numpy as np
from numba import njit, prange

_PROFILE_RUNTIME_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME"
logger = logging.getLogger(__name__)


def profile_enabled() -> bool:
    """Return whether per-function granularity runtime profiling is enabled."""
    return os.environ.get(_PROFILE_RUNTIME_ENV, "").lower() in {"1", "true", "yes"}


def log_profile(label: str, seconds: float, **fields: object) -> None:
    """Emit one granularity runtime profile event when enabled."""
    if not profile_enabled():
        return
    field_text = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("RUNTIME_PROFILE %s %.6fs %s", label, seconds, field_text)


@dataclass(frozen=True, slots=True)
class GranularityImageSeries:
    """Background-corrected image and reconstruction series."""

    pixels: np.ndarray
    new_shape: np.ndarray
    reconstructions: tuple[np.ndarray, ...]


@dataclass(frozen=True, slots=True)
class GranularityImageSeriesCacheEntry:
    """Cache entry for one exact granularity image-series key."""

    series: GranularityImageSeries


@dataclass(frozen=True, slots=True)
class GranularityImageSeriesRequest:
    """Request for reusable background-corrected granularity reconstructions."""

    image: np.ndarray
    subsample_size: float
    background_subsample_size: float
    element_radius: int
    spectrum_length: int
    profile_function: str

    def series(self) -> GranularityImageSeries:
        image_array = np.asarray(self.image)
        phase_started_at = time.perf_counter()
        dtype, shape, digest = granularity_image_content_key(image_array)
        log_profile(
            "granularity_series_key",
            time.perf_counter() - phase_started_at,
            function=self.profile_function,
        )
        key = (
            dtype,
            shape,
            digest,
            float(self.subsample_size),
            float(self.background_subsample_size),
            int(self.element_radius),
            int(self.spectrum_length),
        )
        entry = GRANULARITY_IMAGE_SERIES_CACHE.get(key)
        if entry is not None:
            GRANULARITY_IMAGE_SERIES_CACHE.move_to_end(key)
            log_profile(
                "granularity_series_cache_hit",
                0.0,
                function=self.profile_function,
            )
            return entry.series

        phase_started_at = time.perf_counter()
        pixels, new_shape = background_corrected_pixels(
            image_array,
            self.subsample_size,
            self.background_subsample_size,
            self.element_radius,
        )
        log_profile(
            "granularity_background_correct",
            time.perf_counter() - phase_started_at,
            function=self.profile_function,
            shape=tuple(int(value) for value in pixels.shape),
        )
        phase_started_at = time.perf_counter()
        reconstructions = granularity_reconstruction_series(
            pixels,
            self.spectrum_length,
        )
        log_profile(
            "granularity_reconstruction_series",
            time.perf_counter() - phase_started_at,
            function=self.profile_function,
            reconstructions=len(reconstructions),
        )
        series = GranularityImageSeries(
            pixels=pixels,
            new_shape=new_shape,
            reconstructions=reconstructions,
        )
        GRANULARITY_IMAGE_SERIES_CACHE[key] = GranularityImageSeriesCacheEntry(
            series=series,
        )
        GRANULARITY_IMAGE_SERIES_CACHE.move_to_end(key)
        while len(GRANULARITY_IMAGE_SERIES_CACHE) > GRANULARITY_IMAGE_SERIES_CACHE_MAX_ENTRIES:
            GRANULARITY_IMAGE_SERIES_CACHE.popitem(last=False)
        return series


GRANULARITY_IMAGE_SERIES_CACHE: dict[
    tuple[str, tuple[int, ...], bytes, float, float, int, int],
    GranularityImageSeriesCacheEntry,
] = OrderedDict()
GRANULARITY_IMAGE_SERIES_CACHE_MAX_ENTRIES = 16


def granularity_image_content_key(
    image: np.ndarray,
) -> tuple[str, tuple[int, ...], bytes]:
    """Return an exact value key for deterministic granularity series reuse."""
    contiguous = np.ascontiguousarray(image)
    digest = hashlib.blake2b(contiguous.view(np.uint8), digest_size=16).digest()
    return str(contiguous.dtype), tuple(int(value) for value in contiguous.shape), digest


def granularity_reconstruction_series(
    pixels: np.ndarray,
    spectrum_length: int,
) -> tuple[np.ndarray, ...]:
    """Compute the erosion/reconstruction images shared across object sets."""
    ero = pixels.copy()
    cross_offsets = disk_offsets(1)
    reconstructions = []
    erosion_seconds = 0.0
    reconstruction_seconds = 0.0
    for index in range(int(spectrum_length)):
        phase_started_at = time.perf_counter()
        ero = gray_erosion_offsets_reflect_numba(ero, cross_offsets)
        erosion_seconds += time.perf_counter() - phase_started_at
        phase_started_at = time.perf_counter()
        reconstruction = reconstruct_dilation_cross_numba(ero, pixels)
        reconstruction_seconds += time.perf_counter() - phase_started_at
        log_profile(
            "granularity_reconstruction_iteration",
            time.perf_counter() - phase_started_at,
            function="measure_granularity_objects",
            iteration=index + 1,
            shape=tuple(int(value) for value in pixels.shape),
        )
        reconstructions.append(reconstruction)
    log_profile(
        "granularity_reconstruction_erosion_total",
        erosion_seconds,
        function="measure_granularity_objects",
        reconstructions=len(reconstructions),
    )
    log_profile(
        "granularity_reconstruction_dilation_total",
        reconstruction_seconds,
        function="measure_granularity_objects",
        reconstructions=len(reconstructions),
    )
    return tuple(reconstructions)


def background_corrected_pixels(
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
        pixels = resample_bilinear_numba(
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
        back_pixels = resample_bilinear_numba(
            pixels,
            int(back_shape[0]),
            int(back_shape[1]),
            1.0 / float(background_subsample_size),
            1.0 / float(background_subsample_size),
        )
    else:
        back_pixels = pixels.copy()
        back_shape = new_shape

    footprint_offsets = disk_offsets(element_radius)
    back_pixels = gray_erosion_offsets_reflect_numba(back_pixels, footprint_offsets)
    back_pixels = gray_dilation_offsets_reflect_numba(back_pixels, footprint_offsets)

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
        back_pixels = resample_bilinear_numba(
            back_pixels,
            int(new_shape[0]),
            int(new_shape[1]),
            row_scale,
            col_scale,
        )

    pixels = pixels - back_pixels
    clip_negative_inplace_numba(pixels)
    return pixels, new_shape


def object_granularity_values(
    image: np.ndarray,
    labels: np.ndarray,
    object_range: np.ndarray,
    series: GranularityImageSeries,
    *,
    subsample_size: float,
    spectrum_length: int,
) -> np.ndarray:
    """Return CP granularity spectrum values for each object id."""
    orig_shape = image.shape
    pixels = series.pixels
    new_shape = series.new_shape
    label_to_index = label_to_index_lookup_numba(object_range)
    pixel_rows, pixel_cols, pixel_object_indices, object_counts = (
        compact_label_pixels_from_lookup_numba(labels, label_to_index)
    )
    current_means = mean_by_compact_label_pixels_numba(
        np.asarray(image),
        pixel_rows,
        pixel_cols,
        pixel_object_indices,
        object_counts,
    )
    start_means = np.maximum(current_means, np.finfo(float).eps)
    gs_per_object = np.zeros((int(object_range.size), 16))
    for gs_idx, rec in enumerate(series.reconstructions[: int(spectrum_length)]):
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
            new_means = mean_by_compact_label_pixels_from_resampled_numba(
                np.asarray(rec),
                pixel_rows,
                pixel_cols,
                pixel_object_indices,
                object_counts,
                row_scale,
                col_scale,
            )
        else:
            new_means = mean_by_compact_label_pixels_numba(
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
    return gs_per_object


@lru_cache(maxsize=None)
def disk_offsets(radius: int) -> np.ndarray:
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
def resample_bilinear_numba(
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
            result[row, col] = bilinear_sample_numba(
                image,
                sample_row,
                col * col_scale,
            )
    return result


@njit(cache=True, parallel=True)
def clip_negative_inplace_numba(image: np.ndarray) -> None:
    height, width = image.shape
    for row in prange(height):
        for col in range(width):
            if image[row, col] < 0.0:
                image[row, col] = 0.0


@njit(cache=True, parallel=True)
def gray_erosion_offsets_reflect_numba(
    image: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    height, width = image.shape
    result = np.empty((height, width), dtype=np.float64)
    for row in prange(height):
        for col in range(width):
            best = np.inf
            for offset_index in range(offsets.shape[0]):
                sample_row = reflect_index(row + int(offsets[offset_index, 0]), height)
                sample_col = reflect_index(col + int(offsets[offset_index, 1]), width)
                value = float(image[sample_row, sample_col])
                if value < best:
                    best = value
            result[row, col] = best
    return result


@njit(cache=True, parallel=True)
def gray_dilation_offsets_reflect_numba(
    image: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    height, width = image.shape
    result = np.empty((height, width), dtype=np.float64)
    for row in prange(height):
        for col in range(width):
            best = -np.inf
            for offset_index in range(offsets.shape[0]):
                sample_row = reflect_index(row + int(offsets[offset_index, 0]), height)
                sample_col = reflect_index(col + int(offsets[offset_index, 1]), width)
                value = float(image[sample_row, sample_col])
                if value > best:
                    best = value
            result[row, col] = best
    return result


@njit(cache=True)
def reflect_index(index: int, size: int) -> int:
    if size <= 1:
        return 0
    while index < 0 or index >= size:
        if index < 0:
            index = -index - 1
        else:
            index = 2 * size - index - 1
    return index


@njit(cache=True)
def reconstruct_dilation_cross_numba(
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
                tail, queue_count = enqueue_reconstruct_neighbor(
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
                tail, queue_count = enqueue_reconstruct_neighbor(
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
                tail, queue_count = enqueue_reconstruct_neighbor(
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
                tail, queue_count = enqueue_reconstruct_neighbor(
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
            tail, queue_count = propagate_reconstruct_neighbor(
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
            tail, queue_count = propagate_reconstruct_neighbor(
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
            tail, queue_count = propagate_reconstruct_neighbor(
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
            tail, queue_count = propagate_reconstruct_neighbor(
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
def enqueue_reconstruct_neighbor(
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
def propagate_reconstruct_neighbor(
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
def label_to_index_lookup_numba(object_ids: np.ndarray) -> np.ndarray:
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
def compact_label_pixels_from_lookup_numba(
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
def mean_by_compact_label_pixels_numba(
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
def mean_by_compact_label_pixels_from_resampled_numba(
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
        sums[object_index] += bilinear_sample_numba(
            image,
            float(pixel_rows[pixel_index]) * row_scale,
            float(pixel_cols[pixel_index]) * col_scale,
        )

    for index in range(object_count):
        count = object_counts[index]
        means[index] = sums[index] / count if count > 0 else np.nan
    return means


@njit(cache=True)
def bilinear_sample_numba(
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


__all__ = [
    "GRANULARITY_IMAGE_SERIES_CACHE",
    "GRANULARITY_IMAGE_SERIES_CACHE_MAX_ENTRIES",
    "GranularityImageSeries",
    "GranularityImageSeriesCacheEntry",
    "GranularityImageSeriesRequest",
    "background_corrected_pixels",
    "bilinear_sample_numba",
    "compact_label_pixels_from_lookup_numba",
    "disk_offsets",
    "granularity_image_content_key",
    "granularity_reconstruction_series",
    "gray_dilation_offsets_reflect_numba",
    "gray_erosion_offsets_reflect_numba",
    "label_to_index_lookup_numba",
    "log_profile",
    "mean_by_compact_label_pixels_from_resampled_numba",
    "mean_by_compact_label_pixels_numba",
    "object_granularity_values",
    "reconstruct_dilation_cross_numba",
]
