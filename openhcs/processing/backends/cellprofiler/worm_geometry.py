"""Shared geometry helpers for absorbed CellProfiler worm modules."""

from __future__ import annotations

from dataclasses import dataclass

import centrosome.cpmorphology
import numpy as np
from numba import njit
from scipy.interpolate import interp1d
from scipy.ndimage import distance_transform_edt, label


def eight_connectivity() -> np.ndarray:
    """Return an 8-connectivity structuring element."""
    return np.ones((3, 3), bool)


def skeletonize_worm_mask(binary_image: np.ndarray) -> np.ndarray:
    """Skeletonize a worm mask using CellProfiler's ordered thinning semantics."""
    return _cellprofiler_skeletonize(binary_image > 0)


@dataclass(frozen=True, slots=True)
class CellProfilerLookupTableProjection:
    """One centrosome-compatible 3x3 lookup-table projection."""

    table: np.ndarray
    border_value: bool

    def apply(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image, dtype=bool)
        indexer = np.zeros(image.shape, dtype=np.int16)
        indexer[1:, 1:] += image[:-1, :-1] * 2**0
        indexer[1:, :] += image[:-1, :] * 2**1
        indexer[1:, :-1] += image[:-1, 1:] * 2**2
        indexer[:, 1:] += image[:, :-1] * 2**3
        indexer[:, :] += image[:, :] * 2**4
        indexer[:, :-1] += image[:, 1:] * 2**5
        indexer[:-1, 1:] += image[1:, :-1] * 2**6
        indexer[:-1, :] += image[1:, :] * 2**7
        indexer[:-1, :-1] += image[1:, 1:] * 2**8
        if self.border_value:
            indexer[0, :] |= 2**0 + 2**1 + 2**2
            indexer[-1, :] |= 2**6 + 2**7 + 2**8
            indexer[:, 0] |= 2**0 + 2**3 + 2**6
            indexer[:, -1] |= 2**2 + 2**5 + 2**8
        return self.table[indexer]


@dataclass(frozen=True, slots=True)
class CellProfilerLookupPattern:
    """Boolean 3x3 neighborhood represented by one centrosome table index."""

    index: int

    @property
    def array(self) -> np.ndarray:
        index = self.index
        return np.array(
            [
                [index & 2**0, index & 2**1, index & 2**2],
                [index & 2**3, index & 2**4, index & 2**5],
                [index & 2**6, index & 2**7, index & 2**8],
            ],
            bool,
        )


def _cellprofiler_skeletonize(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    ordering: np.ndarray | None = None,
) -> np.ndarray:
    """Port of ``centrosome.cpmorphology.skeletonize`` with a numba erosion loop."""
    if mask is None:
        masked_image = np.asarray(image, dtype=bool)
    else:
        masked_image = np.asarray(image, dtype=bool).copy()
        masked_image[~np.asarray(mask, dtype=bool)] = False

    if not np.any(masked_image):
        return masked_image.copy()

    if ordering is None:
        distance = distance_transform_edt(masked_image)
    else:
        distance = np.asarray(ordering)

    corner_score = CellProfilerLookupTableProjection(
        _CELLPROFILER_CORNERNESS_TABLE,
        border_value=False,
    ).apply(masked_image)
    rows, cols = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    result_mask = masked_image
    result = np.ascontiguousarray(result_mask, dtype=np.uint8)
    active_rows = np.ascontiguousarray(rows[result_mask], dtype=np.int32)
    active_cols = np.ascontiguousarray(cols[result_mask], dtype=np.int32)

    rng = np.random.RandomState(0)
    tiebreaker = rng.permutation(np.arange(int(np.prod(masked_image.shape))))
    tiebreaker.shape = masked_image.shape
    order = np.lexsort(
        (
            tiebreaker[result_mask],
            corner_score[result_mask],
            distance[result_mask],
        )
    )
    _skeletonize_loop_numba(
        result,
        active_rows,
        active_cols,
        np.ascontiguousarray(order, dtype=np.int32),
        _cellprofiler_skeletonize_table_uint8(),
    )

    skeleton = result.astype(bool)
    if mask is not None:
        skeleton[~mask] = image[~mask]
    return skeleton


def _cellprofiler_cornerness_table() -> np.ndarray:
    return np.array(
        [9 - np.sum(CellProfilerLookupPattern(index).array) for index in range(512)]
    )


def _cellprofiler_skeletonize_table_uint8() -> np.ndarray:
    return _CELLPROFILER_SKELETONIZE_TABLE.astype(np.uint8)


def _cellprofiler_skeletonize_table() -> np.ndarray:
    isolated_center = _make_table(
        True,
        np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], bool),
        np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], bool),
    )
    return isolated_center & np.array(
        [
            label(CellProfilerLookupPattern(index).array, eight_connectivity())[1]
            != label(
                CellProfilerLookupPattern(index & ~(2**4)).array,
                eight_connectivity(),
            )[1]
            or np.sum(CellProfilerLookupPattern(index).array) < 3
            for index in range(512)
        ],
        dtype=bool,
    )


def _cellprofiler_branchpoints_table() -> np.ndarray:
    four_connectivity = np.array(
        [[False, True, False], [True, True, True], [False, True, False]],
        dtype=bool,
    )
    return np.array(
        [
            CellProfilerLookupPattern(index).array[1, 1]
            and label(
                CellProfilerLookupPattern(index - 2**4).array,
                four_connectivity,
            )[1] > 2
            for index in range(512)
        ],
        dtype=bool,
    )


def _cellprofiler_endpoints_table() -> np.ndarray:
    return np.array(
        [
            CellProfilerLookupPattern(index).array[1, 1]
            and np.sum(CellProfilerLookupPattern(index).array) <= 2
            for index in range(512)
        ],
        dtype=bool,
    )


def _make_table(
    value: bool,
    pattern: np.ndarray,
    care: np.ndarray,
) -> np.ndarray:
    def matches(index: int, bit: int, row: int, column: int) -> bool:
        return (((index & 2**bit) > 0) == pattern[row, column]) or not care[
            row,
            column,
        ]

    return np.array(
        [
            value
            if (
                matches(index, 0, 0, 0)
                and matches(index, 1, 0, 1)
                and matches(index, 2, 0, 2)
                and matches(index, 3, 1, 0)
                and matches(index, 4, 1, 1)
                and matches(index, 5, 1, 2)
                and matches(index, 6, 2, 0)
                and matches(index, 7, 2, 1)
                and matches(index, 8, 2, 2)
            )
            else not value
            for index in range(512)
        ],
        bool,
    )


@njit(cache=True)
def _skeletonize_loop_numba(
    result: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    order: np.ndarray,
    table: np.ndarray,
) -> None:
    height, width = result.shape
    for order_index in range(order.shape[0]):
        pixel_index = order[order_index]
        row = rows[pixel_index]
        col = cols[pixel_index]
        if result[row, col] == 0:
            continue

        table_index = 0
        bit = 0
        for row_delta in range(-1, 2):
            neighbor_row = row + row_delta
            for col_delta in range(-1, 2):
                neighbor_col = col + col_delta
                if (
                    0 <= neighbor_row < height
                    and 0 <= neighbor_col < width
                    and result[neighbor_row, neighbor_col] != 0
                ):
                    table_index |= 1 << bit
                bit += 1

        if table[table_index] == 0:
            result[row, col] = 0


def branchpoints(skeleton: np.ndarray) -> np.ndarray:
    """Find branchpoints in a skeleton using CellProfiler's lookup semantics."""
    return CellProfilerLookupTableProjection(
        _CELLPROFILER_BRANCHPOINTS_TABLE,
        border_value=False,
    ).apply(skeleton)


def endpoints(skeleton: np.ndarray) -> np.ndarray:
    """Find endpoints in a skeleton using CellProfiler's lookup semantics."""
    return CellProfilerLookupTableProjection(
        _CELLPROFILER_ENDPOINTS_TABLE,
        border_value=False,
    ).apply(skeleton)


def trace_skeleton_path(skeleton: np.ndarray) -> np.ndarray:
    """Trace a stable path through a skeleton."""
    if not np.any(skeleton):
        return np.zeros((0, 2), dtype=int)

    endpoint_coords = np.argwhere(endpoints(skeleton))
    start = endpoint_coords[0] if len(endpoint_coords) else np.argwhere(skeleton)[0]
    path = [tuple(start)]
    visited = set(path)
    current = start

    while True:
        neighbors = tuple(
            (current[0] + row_delta, current[1] + column_delta)
            for row_delta in (-1, 0, 1)
            for column_delta in (-1, 0, 1)
            if (row_delta, column_delta) != (0, 0)
        )
        next_points = tuple(
            point
            for point in neighbors
            if (
                0 <= point[0] < skeleton.shape[0]
                and 0 <= point[1] < skeleton.shape[1]
                and skeleton[point]
                and point not in visited
            )
        )
        if not next_points:
            break
        current = np.array(next_points[0])
        path.append(tuple(current))
        visited.add(tuple(current))

    return np.array(path)


def calculate_cumulative_lengths(path_coords: np.ndarray) -> np.ndarray:
    """Return cumulative path length for Nx2 path coordinates."""
    if len(path_coords) < 2:
        return np.zeros(len(path_coords))
    diffs = path_coords[1:] - path_coords[:-1]
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    return np.hstack(([0.0], np.cumsum(segment_lengths)))


def sample_control_points(
    path_coords: np.ndarray,
    cumul_lengths: np.ndarray,
    num_control_points: int,
) -> np.ndarray:
    """Sample exactly N control points using CellProfiler's path indexing."""
    if num_control_points <= 0:
        raise ValueError("num_control_points must be positive.")
    if len(path_coords) == 0:
        return np.zeros((num_control_points, 2), dtype=float)
    if len(path_coords) == 1:
        return np.repeat(path_coords.astype(float), num_control_points, axis=0)

    path_coords = path_coords.astype(float)
    cumul_lengths = cumul_lengths.astype(float)
    unique_mask = np.hstack(([True], cumul_lengths[1:] != cumul_lengths[:-1]))
    path_coords = path_coords[unique_mask]
    cumul_lengths = cumul_lengths[unique_mask]
    if len(path_coords) == 1 or cumul_lengths[-1] <= 0:
        return np.repeat(path_coords[:1], num_control_points, axis=0)
    if num_control_points == 1:
        return path_coords[:1]

    first = float(cumul_lengths[-1]) / float(num_control_points - 1)
    last = float(cumul_lengths[-1]) - first
    if num_control_points == 2:
        return path_coords[[0, -1]]

    path_index_for_distance = interp1d(
        cumul_lengths,
        np.linspace(0.0, float(len(path_coords) - 1), len(path_coords)),
    )
    fractional_indexes = path_index_for_distance(
        np.linspace(first, last, num_control_points - 2)
    )
    indexes = fractional_indexes.astype(int)
    fractions = fractional_indexes - indexes
    sampled = (
        path_coords[indexes, :] * (1 - fractions[:, np.newaxis])
        + path_coords[indexes + 1, :] * fractions[:, np.newaxis]
    )
    return np.vstack((path_coords[:1, :], sampled, path_coords[-1:, :]))


def rebuild_worm_from_control_points_approx(
    control_coords: np.ndarray,
    worm_radii: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild a worm using CellProfiler's canonical line rasterization."""
    if len(control_coords) < 2:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    control_coords = np.asarray(control_coords, dtype=np.float64)
    radii = np.asarray(worm_radii, dtype=np.float64)
    if len(radii) < len(control_coords):
        radii = np.pad(radii, (0, len(control_coords) - len(radii)), mode="edge")

    index, count, rows, columns = centrosome.cpmorphology.get_line_pts(
        control_coords[:-1, 0],
        control_coords[:-1, 1],
        control_coords[1:, 0],
        control_coords[1:, 1],
    )
    rows = np.delete(rows, index[1:])
    columns = np.delete(columns, index[1:])
    index = index - np.arange(len(index))
    count -= 1

    nonempty_segments = count != 0
    index = index[nonempty_segments]
    count = count[nonempty_segments]

    segment_labels = np.zeros(len(rows), dtype=int)
    segment_labels[index[1:]] = 1
    segment_labels = np.cumsum(segment_labels)
    order_within_segment = np.arange(len(rows)) - index[segment_labels]
    fractions = order_within_segment.astype(float) / count[segment_labels].astype(
        float
    )
    point_radii = (
        radii[segment_labels] * (1.0 - fractions)
        + radii[segment_labels + 1] * fractions
    )

    max_radius = int(np.max(np.ceil(point_radii)))
    row_offsets, column_offsets = np.mgrid[
        -max_radius : max_radius + 1,
        -max_radius : max_radius + 1,
    ]
    distances = np.sqrt((row_offsets**2 + column_offsets**2).astype(float))
    disk_mask = row_offsets**2 + column_offsets**2 <= max_radius**2
    row_offsets = row_offsets[disk_mask]
    column_offsets = column_offsets[disk_mask]
    distances = distances[disk_mask]

    rows = (rows[:, np.newaxis] + row_offsets[np.newaxis, :]).flatten()
    columns = (columns[:, np.newaxis] + column_offsets[np.newaxis, :]).flatten()
    radius_mask = (
        point_radii[:, np.newaxis] >= distances[np.newaxis, :]
    ).flatten()
    rows = rows[radius_mask]
    columns = columns[radius_mask]

    order = np.lexsort((rows, columns))
    rows = rows[order]
    columns = columns[order]
    unique_mask = np.hstack(
        ([True], (rows[:-1] != rows[1:]) | (columns[:-1] != columns[1:]))
    )
    rows = rows[unique_mask]
    columns = columns[unique_mask]

    height, width = shape
    in_bounds = (
        (rows >= 0)
        & (columns >= 0)
        & (rows < int(height))
        & (columns < int(width))
    )
    return rows[in_bounds], columns[in_bounds]


def control_points_for_label_image(
    labels: np.ndarray,
    num_control_points: int,
) -> np.ndarray:
    """Derive CellProfiler-style control points from a label image."""
    label_image = np.asarray(labels)
    object_numbers = np.unique(label_image)
    object_numbers = object_numbers[object_numbers > 0]
    if len(object_numbers) == 0:
        return np.zeros((0, 2, num_control_points), dtype=float)

    return np.stack(
        tuple(
            _control_points_for_object(label_image == object_number, num_control_points)
            for object_number in object_numbers
        ),
        axis=0,
    )


def _control_points_for_object(
    mask: np.ndarray,
    num_control_points: int,
) -> np.ndarray:
    path_coords = trace_skeleton_path(skeletonize_worm_mask(mask))
    if len(path_coords) < 2:
        path_coords = _fallback_object_path(mask)
    cumul_lengths = calculate_cumulative_lengths(path_coords)
    return sample_control_points(
        path_coords,
        cumul_lengths,
        num_control_points,
    ).T


def _fallback_object_path(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return np.zeros((1, 2), dtype=float)
    if len(coords) == 1:
        return coords.astype(float)

    centered = coords - np.mean(coords, axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ vh[0]
    order = np.argsort(projection)
    return coords[order[[0, -1]]].astype(float)


_CELLPROFILER_SKELETONIZE_TABLE = _cellprofiler_skeletonize_table()
_CELLPROFILER_CORNERNESS_TABLE = _cellprofiler_cornerness_table()
_CELLPROFILER_BRANCHPOINTS_TABLE = _cellprofiler_branchpoints_table()
_CELLPROFILER_ENDPOINTS_TABLE = _cellprofiler_endpoints_table()
