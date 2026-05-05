"""Alignment backends for CellProfiler-compatible processing."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit

from openhcs.constants.constants import MemoryType
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    cellprofiler_backend_key,
)


class AlignmentBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Alignment operations keyed by OpenHCS memory type/provider."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def mutual_information_offset(
        self,
        reference_pixels: np.ndarray,
        moving_pixels: np.ndarray,
        reference_mask: np.ndarray,
        moving_mask: np.ndarray,
    ) -> tuple[int, int]:
        """Return column/row offset maximizing mutual information."""


class NumbaNumpyAlignmentBackendStrategy(AlignmentBackendStrategy):
    """Numba-accelerated NumPy alignment primitives."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def mutual_information_offset(
        self,
        reference_pixels: np.ndarray,
        moving_pixels: np.ndarray,
        reference_mask: np.ndarray,
        moving_mask: np.ndarray,
    ) -> tuple[int, int]:
        max_shape = np.maximum(reference_pixels.shape, moving_pixels.shape)
        reshaped_reference_pixels = _reshape_image(reference_pixels, max_shape)
        reshaped_moving_pixels = _reshape_image(moving_pixels, max_shape)
        reshaped_reference_mask = _reshape_image(reference_mask, max_shape)
        reshaped_moving_mask = _reshape_image(moving_mask, max_shape)

        if bool(np.all(reshaped_reference_mask)) and bool(np.all(reshaped_moving_mask)):
            return _mutual_information_offset_unmasked_numba(
                np.asarray(reshaped_reference_pixels, dtype=np.float64),
                np.asarray(reshaped_moving_pixels, dtype=np.float64),
            )

        return _mutual_information_offset_numba(
            np.asarray(reshaped_reference_pixels, dtype=np.float64),
            np.asarray(reshaped_moving_pixels, dtype=np.float64),
            np.asarray(reshaped_reference_mask, dtype=np.bool_),
            np.asarray(reshaped_moving_mask, dtype=np.bool_),
        )


def alignment_backend(
    *,
    backend_provider: BackendProviderInput | None = None,
) -> AlignmentBackendStrategy:
    """Return the selected CellProfiler alignment backend."""
    return AlignmentBackendStrategy.for_memory_type(
        backend_provider=backend_provider,
    )


def _reshape_image(source: np.ndarray, new_shape: tuple[int, int]) -> np.ndarray:
    if tuple(source.shape) == tuple(new_shape):
        return source
    result = np.zeros(new_shape, source.dtype)
    result[: source.shape[0], : source.shape[1]] = source
    return result


@njit(cache=True)
def _mutual_information_offset_unmasked_numba(
    reference_pixels: np.ndarray,
    moving_pixels: np.ndarray,
) -> tuple[int, int]:
    (
        reference_global_min,
        reference_global_max,
        moving_global_min,
        moving_global_max,
    ) = _global_min_max_pair_numba(reference_pixels, moving_pixels)
    reference_bins = _global_uint8_bins_numba(
        reference_pixels,
        reference_global_min,
        reference_global_max,
    )
    moving_bins = _global_uint8_bins_numba(
        moving_pixels,
        moving_global_min,
        moving_global_max,
    )
    ref_hist = np.zeros(256, dtype=np.int64)
    moving_hist = np.zeros(256, dtype=np.int64)
    joint_hist = np.zeros(65536, dtype=np.int64)
    used_ref_bins = np.empty(256, dtype=np.int64)
    used_moving_bins = np.empty(256, dtype=np.int64)
    used_joint_bins = np.empty(65536, dtype=np.int64)
    best = _mutual_information_for_offset_unmasked_numba(
        reference_pixels,
        moving_pixels,
        0,
        0,
        ref_hist,
        moving_hist,
        joint_hist,
        used_ref_bins,
        used_moving_bins,
        used_joint_bins,
        reference_bins,
        moving_bins,
        reference_global_min,
        reference_global_max,
        moving_global_min,
        moving_global_max,
    )
    row_offset = 0
    column_offset = 0
    while True:
        previous_row_offset = row_offset
        previous_column_offset = column_offset
        for candidate_row in range(previous_row_offset - 1, previous_row_offset + 2):
            for candidate_column in range(
                previous_column_offset - 1,
                previous_column_offset + 2,
            ):
                if candidate_row == 0 and candidate_column == 0:
                    continue
                information = _mutual_information_for_offset_unmasked_numba(
                    reference_pixels,
                    moving_pixels,
                    candidate_row,
                    candidate_column,
                    ref_hist,
                    moving_hist,
                    joint_hist,
                    used_ref_bins,
                    used_moving_bins,
                    used_joint_bins,
                    reference_bins,
                    moving_bins,
                    reference_global_min,
                    reference_global_max,
                    moving_global_min,
                    moving_global_max,
                )
                if information > best:
                    best = information
                    row_offset = candidate_row
                    column_offset = candidate_column
        if row_offset == previous_row_offset and column_offset == previous_column_offset:
            return int(column_offset), int(row_offset)


@njit(cache=True)
def _mutual_information_for_offset_unmasked_numba(
    reference_pixels: np.ndarray,
    moving_pixels: np.ndarray,
    row_offset: int,
    column_offset: int,
    ref_hist: np.ndarray,
    moving_hist: np.ndarray,
    joint_hist: np.ndarray,
    used_ref_bins: np.ndarray,
    used_moving_bins: np.ndarray,
    used_joint_bins: np.ndarray,
    reference_bins: np.ndarray,
    moving_bins: np.ndarray,
    reference_global_min: float,
    reference_global_max: float,
    moving_global_min: float,
    moving_global_max: float,
) -> float:
    rows = reference_pixels.shape[0]
    cols = reference_pixels.shape[1]
    if row_offset < 0:
        moving_row_start = -row_offset
        reference_row_start = 0
        height = rows + row_offset
    else:
        moving_row_start = 0
        reference_row_start = row_offset
        height = rows - row_offset
    if column_offset < 0:
        moving_col_start = -column_offset
        reference_col_start = 0
        width = cols + column_offset
    else:
        moving_col_start = 0
        reference_col_start = column_offset
        width = cols - column_offset
    if height <= 0 or width <= 0:
        return 0.0

    count = height * width
    ref_min = reference_pixels[reference_row_start, reference_col_start]
    ref_max = ref_min
    moving_min = moving_pixels[moving_row_start, moving_col_start]
    moving_max = moving_min
    for y in range(height):
        ref_y = reference_row_start + y
        mov_y = moving_row_start + y
        for x in range(width):
            ref_value = reference_pixels[ref_y, reference_col_start + x]
            moving_value = moving_pixels[mov_y, moving_col_start + x]
            if ref_value < ref_min:
                ref_min = ref_value
            if ref_value > ref_max:
                ref_max = ref_value
            if moving_value < moving_min:
                moving_min = moving_value
            if moving_value > moving_max:
                moving_max = moving_value

    used_ref_count = 0
    used_moving_count = 0
    used_joint_count = 0
    ref_span = ref_max - ref_min
    moving_span = moving_max - moving_min
    if (
        ref_min == reference_global_min
        and ref_max == reference_global_max
        and moving_min == moving_global_min
        and moving_max == moving_global_max
    ):
        return _mutual_information_for_offset_prebinned_unmasked_numba(
            reference_bins,
            moving_bins,
            reference_row_start,
            reference_col_start,
            moving_row_start,
            moving_col_start,
            height,
            width,
            ref_hist,
            moving_hist,
            joint_hist,
            used_ref_bins,
            used_moving_bins,
            used_joint_bins,
        )
    for y in range(height):
        ref_y = reference_row_start + y
        mov_y = moving_row_start + y
        for x in range(width):
            ref_x = reference_col_start + x
            mov_x = moving_col_start + x
            ref_bin = 0
            moving_bin = 0
            if ref_span > 0.0:
                ref_bin = int((reference_pixels[ref_y, ref_x] - ref_min) * 255.0 / ref_span)
                if ref_bin < 0:
                    ref_bin = 0
                elif ref_bin > 255:
                    ref_bin = 255
            if moving_span > 0.0:
                moving_bin = int((moving_pixels[mov_y, mov_x] - moving_min) * 255.0 / moving_span)
                if moving_bin < 0:
                    moving_bin = 0
                elif moving_bin > 255:
                    moving_bin = 255
            if ref_hist[ref_bin] == 0:
                used_ref_bins[used_ref_count] = ref_bin
                used_ref_count += 1
            ref_hist[ref_bin] += 1
            if moving_hist[moving_bin] == 0:
                used_moving_bins[used_moving_count] = moving_bin
                used_moving_count += 1
            moving_hist[moving_bin] += 1
            joint_bin = ref_bin * 256 + moving_bin
            if joint_hist[joint_bin] == 0:
                used_joint_bins[used_joint_count] = joint_bin
                used_joint_count += 1
            joint_hist[joint_bin] += 1

    information = (
        _used_histogram_entropy_numba(ref_hist, used_ref_bins, used_ref_count, count)
        + _used_histogram_entropy_numba(
            moving_hist,
            used_moving_bins,
            used_moving_count,
            count,
        )
        - _used_histogram_entropy_numba(
            joint_hist,
            used_joint_bins,
            used_joint_count,
            count,
        )
    )
    _clear_used_histogram_bins_numba(ref_hist, used_ref_bins, used_ref_count)
    _clear_used_histogram_bins_numba(moving_hist, used_moving_bins, used_moving_count)
    _clear_used_histogram_bins_numba(joint_hist, used_joint_bins, used_joint_count)
    return information


@njit(cache=True)
def _mutual_information_for_offset_prebinned_unmasked_numba(
    reference_bins: np.ndarray,
    moving_bins: np.ndarray,
    reference_row_start: int,
    reference_col_start: int,
    moving_row_start: int,
    moving_col_start: int,
    height: int,
    width: int,
    ref_hist: np.ndarray,
    moving_hist: np.ndarray,
    joint_hist: np.ndarray,
    used_ref_bins: np.ndarray,
    used_moving_bins: np.ndarray,
    used_joint_bins: np.ndarray,
) -> float:
    count = height * width
    used_ref_count = 0
    used_moving_count = 0
    used_joint_count = 0
    for y in range(height):
        ref_y = reference_row_start + y
        mov_y = moving_row_start + y
        for x in range(width):
            ref_bin = int(reference_bins[ref_y, reference_col_start + x])
            moving_bin = int(moving_bins[mov_y, moving_col_start + x])
            if ref_hist[ref_bin] == 0:
                used_ref_bins[used_ref_count] = ref_bin
                used_ref_count += 1
            ref_hist[ref_bin] += 1
            if moving_hist[moving_bin] == 0:
                used_moving_bins[used_moving_count] = moving_bin
                used_moving_count += 1
            moving_hist[moving_bin] += 1
            joint_bin = ref_bin * 256 + moving_bin
            if joint_hist[joint_bin] == 0:
                used_joint_bins[used_joint_count] = joint_bin
                used_joint_count += 1
            joint_hist[joint_bin] += 1

    information = (
        _used_histogram_entropy_numba(ref_hist, used_ref_bins, used_ref_count, count)
        + _used_histogram_entropy_numba(
            moving_hist,
            used_moving_bins,
            used_moving_count,
            count,
        )
        - _used_histogram_entropy_numba(
            joint_hist,
            used_joint_bins,
            used_joint_count,
            count,
        )
    )
    _clear_used_histogram_bins_numba(ref_hist, used_ref_bins, used_ref_count)
    _clear_used_histogram_bins_numba(moving_hist, used_moving_bins, used_moving_count)
    _clear_used_histogram_bins_numba(joint_hist, used_joint_bins, used_joint_count)
    return information


@njit(cache=True)
def _global_min_max_pair_numba(
    reference_pixels: np.ndarray,
    moving_pixels: np.ndarray,
) -> tuple[float, float, float, float]:
    rows, cols = reference_pixels.shape
    reference_min = reference_pixels[0, 0]
    reference_max = reference_min
    moving_min = moving_pixels[0, 0]
    moving_max = moving_min
    for y in range(rows):
        for x in range(cols):
            reference_value = reference_pixels[y, x]
            moving_value = moving_pixels[y, x]
            if reference_value < reference_min:
                reference_min = reference_value
            if reference_value > reference_max:
                reference_max = reference_value
            if moving_value < moving_min:
                moving_min = moving_value
            if moving_value > moving_max:
                moving_max = moving_value
    return reference_min, reference_max, moving_min, moving_max


@njit(cache=True)
def _global_uint8_bins_numba(
    pixels: np.ndarray,
    pixel_min: float,
    pixel_max: float,
) -> np.ndarray:
    rows, cols = pixels.shape
    bins = np.empty((rows, cols), dtype=np.uint8)
    span = pixel_max - pixel_min
    for y in range(rows):
        for x in range(cols):
            bin_value = 0
            if span > 0.0:
                bin_value = int((pixels[y, x] - pixel_min) * 255.0 / span)
                if bin_value < 0:
                    bin_value = 0
                elif bin_value > 255:
                    bin_value = 255
            bins[y, x] = bin_value
    return bins


@njit(cache=True)
def _mutual_information_offset_numba(
    reference_pixels: np.ndarray,
    moving_pixels: np.ndarray,
    reference_mask: np.ndarray,
    moving_mask: np.ndarray,
) -> tuple[int, int]:
    ref_hist = np.zeros(256, dtype=np.int64)
    moving_hist = np.zeros(256, dtype=np.int64)
    joint_hist = np.zeros(65536, dtype=np.int64)
    used_ref_bins = np.empty(256, dtype=np.int64)
    used_moving_bins = np.empty(256, dtype=np.int64)
    used_joint_bins = np.empty(65536, dtype=np.int64)
    best = _mutual_information_for_offset_numba(
        reference_pixels,
        moving_pixels,
        reference_mask,
        moving_mask,
        0,
        0,
        ref_hist,
        moving_hist,
        joint_hist,
        used_ref_bins,
        used_moving_bins,
        used_joint_bins,
    )
    row_offset = 0
    column_offset = 0
    while True:
        previous_row_offset = row_offset
        previous_column_offset = column_offset
        for candidate_row in range(previous_row_offset - 1, previous_row_offset + 2):
            for candidate_column in range(
                previous_column_offset - 1,
                previous_column_offset + 2,
            ):
                if candidate_row == 0 and candidate_column == 0:
                    continue
                information = _mutual_information_for_offset_numba(
                    reference_pixels,
                    moving_pixels,
                    reference_mask,
                    moving_mask,
                    candidate_row,
                    candidate_column,
                    ref_hist,
                    moving_hist,
                    joint_hist,
                    used_ref_bins,
                    used_moving_bins,
                    used_joint_bins,
                )
                if information > best:
                    best = information
                    row_offset = candidate_row
                    column_offset = candidate_column
        if row_offset == previous_row_offset and column_offset == previous_column_offset:
            return int(column_offset), int(row_offset)


@njit(cache=True)
def _mutual_information_for_offset_numba(
    reference_pixels: np.ndarray,
    moving_pixels: np.ndarray,
    reference_mask: np.ndarray,
    moving_mask: np.ndarray,
    row_offset: int,
    column_offset: int,
    ref_hist: np.ndarray,
    moving_hist: np.ndarray,
    joint_hist: np.ndarray,
    used_ref_bins: np.ndarray,
    used_moving_bins: np.ndarray,
    used_joint_bins: np.ndarray,
) -> float:
    rows = reference_pixels.shape[0]
    cols = reference_pixels.shape[1]
    if row_offset < 0:
        moving_row_start = -row_offset
        reference_row_start = 0
        height = rows + row_offset
    else:
        moving_row_start = 0
        reference_row_start = row_offset
        height = rows - row_offset
    if column_offset < 0:
        moving_col_start = -column_offset
        reference_col_start = 0
        width = cols + column_offset
    else:
        moving_col_start = 0
        reference_col_start = column_offset
        width = cols - column_offset
    if height <= 0 or width <= 0:
        return 0.0

    count = 0
    ref_min = 0.0
    ref_max = 0.0
    moving_min = 0.0
    moving_max = 0.0
    first = True
    for y in range(height):
        ref_y = reference_row_start + y
        mov_y = moving_row_start + y
        for x in range(width):
            ref_x = reference_col_start + x
            mov_x = moving_col_start + x
            if reference_mask[ref_y, ref_x] and moving_mask[mov_y, mov_x]:
                ref_value = reference_pixels[ref_y, ref_x]
                moving_value = moving_pixels[mov_y, mov_x]
                if first:
                    ref_min = ref_value
                    ref_max = ref_value
                    moving_min = moving_value
                    moving_max = moving_value
                    first = False
                else:
                    if ref_value < ref_min:
                        ref_min = ref_value
                    if ref_value > ref_max:
                        ref_max = ref_value
                    if moving_value < moving_min:
                        moving_min = moving_value
                    if moving_value > moving_max:
                        moving_max = moving_value
                count += 1
    if count <= 0:
        return 0.0

    used_ref_count = 0
    used_moving_count = 0
    used_joint_count = 0
    ref_span = ref_max - ref_min
    moving_span = moving_max - moving_min
    for y in range(height):
        ref_y = reference_row_start + y
        mov_y = moving_row_start + y
        for x in range(width):
            ref_x = reference_col_start + x
            mov_x = moving_col_start + x
            if reference_mask[ref_y, ref_x] and moving_mask[mov_y, mov_x]:
                ref_bin = 0
                moving_bin = 0
                if ref_span > 0.0:
                    ref_bin = int((reference_pixels[ref_y, ref_x] - ref_min) * 255.0 / ref_span)
                    if ref_bin < 0:
                        ref_bin = 0
                    elif ref_bin > 255:
                        ref_bin = 255
                if moving_span > 0.0:
                    moving_bin = int((moving_pixels[mov_y, mov_x] - moving_min) * 255.0 / moving_span)
                    if moving_bin < 0:
                        moving_bin = 0
                    elif moving_bin > 255:
                        moving_bin = 255
                if ref_hist[ref_bin] == 0:
                    used_ref_bins[used_ref_count] = ref_bin
                    used_ref_count += 1
                ref_hist[ref_bin] += 1
                if moving_hist[moving_bin] == 0:
                    used_moving_bins[used_moving_count] = moving_bin
                    used_moving_count += 1
                moving_hist[moving_bin] += 1
                joint_bin = ref_bin * 256 + moving_bin
                if joint_hist[joint_bin] == 0:
                    used_joint_bins[used_joint_count] = joint_bin
                    used_joint_count += 1
                joint_hist[joint_bin] += 1

    information = (
        _used_histogram_entropy_numba(ref_hist, used_ref_bins, used_ref_count, count)
        + _used_histogram_entropy_numba(
            moving_hist,
            used_moving_bins,
            used_moving_count,
            count,
        )
        - _used_histogram_entropy_numba(
            joint_hist,
            used_joint_bins,
            used_joint_count,
            count,
        )
    )
    _clear_used_histogram_bins_numba(ref_hist, used_ref_bins, used_ref_count)
    _clear_used_histogram_bins_numba(moving_hist, used_moving_bins, used_moving_count)
    _clear_used_histogram_bins_numba(joint_hist, used_joint_bins, used_joint_count)
    return information


@njit(cache=True)
def _used_histogram_entropy_numba(
    histogram: np.ndarray,
    used_bins: np.ndarray,
    used_count: int,
    count: int,
) -> float:
    if count <= 0:
        return 0.0
    weighted = 0.0
    for index in range(used_count):
        value = histogram[used_bins[index]]
        if value > 0:
            weighted += value * np.log2(value)
    return float(np.log2(count) - weighted / count)


@njit(cache=True)
def _clear_used_histogram_bins_numba(
    histogram: np.ndarray,
    used_bins: np.ndarray,
    used_count: int,
) -> None:
    for index in range(used_count):
        histogram[used_bins[index]] = 0


__all__ = [
    "AlignmentBackendStrategy",
    "NumbaNumpyAlignmentBackendStrategy",
    "alignment_backend",
]
