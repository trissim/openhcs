"""Shared geometry helpers for absorbed CellProfiler worm modules."""

from __future__ import annotations

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

    corner_score = _table_lookup_once(
        masked_image,
        _CELLPROFILER_CORNERNESS_TABLE,
        border_value=False,
    )
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


def _table_lookup_once(
    image: np.ndarray,
    table: np.ndarray,
    *,
    border_value: bool,
) -> np.ndarray:
    """Apply a centrosome-compatible 3x3 lookup table for one iteration."""
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
    if border_value:
        indexer[0, :] |= 2**0 + 2**1 + 2**2
        indexer[-1, :] |= 2**6 + 2**7 + 2**8
        indexer[:, 0] |= 2**0 + 2**3 + 2**6
        indexer[:, -1] |= 2**2 + 2**5 + 2**8
    return table[indexer]


def _cellprofiler_cornerness_table() -> np.ndarray:
    return np.array([9 - np.sum(_pattern_of(index)) for index in range(512)])


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
            label(_pattern_of(index), eight_connectivity())[1]
            != label(_pattern_of(index & ~(2**4)), eight_connectivity())[1]
            or np.sum(_pattern_of(index)) < 3
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
            _pattern_of(index)[1, 1]
            and label(_pattern_of(index - 2**4), four_connectivity)[1] > 2
            for index in range(512)
        ],
        dtype=bool,
    )


def _cellprofiler_endpoints_table() -> np.ndarray:
    return np.array(
        [
            _pattern_of(index)[1, 1] and np.sum(_pattern_of(index)) <= 2
            for index in range(512)
        ],
        dtype=bool,
    )


def _pattern_of(index: int) -> np.ndarray:
    return np.array(
        [
            [index & 2**0, index & 2**1, index & 2**2],
            [index & 2**3, index & 2**4, index & 2**5],
            [index & 2**6, index & 2**7, index & 2**8],
        ],
        bool,
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
    return _table_lookup_once(
        np.asarray(skeleton, dtype=bool),
        _CELLPROFILER_BRANCHPOINTS_TABLE,
        border_value=False,
    )


def endpoints(skeleton: np.ndarray) -> np.ndarray:
    """Find endpoints in a skeleton using CellProfiler's lookup semantics."""
    return _table_lookup_once(
        np.asarray(skeleton, dtype=bool),
        _CELLPROFILER_ENDPOINTS_TABLE,
        border_value=False,
    )


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
    """Rebuild a CP-style worm mask from control points and trained radii."""
    if len(control_coords) < 2:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    radii = np.asarray(worm_radii, dtype=np.float64)
    if len(radii) < len(control_coords):
        radii = np.pad(radii, (0, len(control_coords) - len(radii)), mode="edge")
    return _rebuild_worm_from_control_points_numba(
        np.ascontiguousarray(control_coords, dtype=np.float64),
        np.ascontiguousarray(radii, dtype=np.float64),
        int(shape[0]),
        int(shape[1]),
    )


@njit(cache=True)
def _rebuild_worm_from_control_points_numba(
    control_coords: np.ndarray,
    worm_radii: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    segment_count = control_coords.shape[0] - 1
    if segment_count <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    total_line_points = 0
    max_radius = 0
    for segment_index in range(segment_count):
        row0 = int(control_coords[segment_index, 0])
        col0 = int(control_coords[segment_index, 1])
        row1 = int(control_coords[segment_index + 1, 0])
        col1 = int(control_coords[segment_index + 1, 1])
        count = max(abs(row0 - row1), abs(col0 - col1)) + 1
        denominator = count - 1
        if denominator == 0:
            continue
        emitted_count = count if segment_index == segment_count - 1 else count - 1
        total_line_points += emitted_count
        radius0 = worm_radii[segment_index]
        radius1 = worm_radii[segment_index + 1]
        for order in range(emitted_count):
            fraction = float(order) / float(denominator)
            radius = radius0 * (1.0 - fraction) + radius1 * fraction
            radius_int = int(np.ceil(radius))
            if radius_int > max_radius:
                max_radius = radius_int

    if total_line_points == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    disk_count = 0
    for row_delta in range(-max_radius, max_radius + 1):
        for col_delta in range(-max_radius, max_radius + 1):
            if row_delta * row_delta + col_delta * col_delta <= max_radius * max_radius:
                disk_count += 1
    disk_rows = np.empty(disk_count, dtype=np.int64)
    disk_cols = np.empty(disk_count, dtype=np.int64)
    disk_distances = np.empty(disk_count, dtype=np.float64)
    disk_index = 0
    for row_delta in range(-max_radius, max_radius + 1):
        for col_delta in range(-max_radius, max_radius + 1):
            distance2 = row_delta * row_delta + col_delta * col_delta
            if distance2 <= max_radius * max_radius:
                disk_rows[disk_index] = row_delta
                disk_cols[disk_index] = col_delta
                disk_distances[disk_index] = np.sqrt(float(distance2))
                disk_index += 1

    expanded_count = 0
    for segment_index in range(segment_count):
        expanded_count += _count_expanded_segment_pixels(
            control_coords,
            worm_radii,
            segment_index,
            disk_distances,
        )
    if expanded_count == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    expanded_rows = np.empty(expanded_count, dtype=np.int64)
    expanded_cols = np.empty(expanded_count, dtype=np.int64)
    write_index = 0
    for segment_index in range(segment_count):
        write_index = _write_expanded_segment_pixels(
            control_coords,
            worm_radii,
            segment_index,
            disk_rows,
            disk_cols,
            disk_distances,
            expanded_rows,
            expanded_cols,
            write_index,
        )

    keys = expanded_cols * height + expanded_rows
    order = np.argsort(keys)
    unique_count = 0
    previous_key = -1
    for order_index in range(order.shape[0]):
        row = expanded_rows[order[order_index]]
        col = expanded_cols[order[order_index]]
        if row < 0 or col < 0 or row >= height or col >= width:
            continue
        key = col * height + row
        if key == previous_key:
            continue
        unique_count += 1
        previous_key = key
    rows = np.empty(unique_count, dtype=np.int64)
    cols = np.empty(unique_count, dtype=np.int64)
    previous_key = -1
    output_index = 0
    for order_index in range(order.shape[0]):
        row = expanded_rows[order[order_index]]
        col = expanded_cols[order[order_index]]
        if row < 0 or col < 0 or row >= height or col >= width:
            continue
        key = col * height + row
        if key == previous_key:
            continue
        rows[output_index] = row
        cols[output_index] = col
        output_index += 1
        previous_key = key
    return rows, cols


@njit(cache=True)
def _count_expanded_segment_pixels(
    control_coords: np.ndarray,
    worm_radii: np.ndarray,
    segment_index: int,
    disk_distances: np.ndarray,
) -> int:
    row0 = int(control_coords[segment_index, 0])
    col0 = int(control_coords[segment_index, 1])
    row1 = int(control_coords[segment_index + 1, 0])
    col1 = int(control_coords[segment_index + 1, 1])
    diff_i = abs(row0 - row1)
    diff_j = abs(col0 - col1)
    count = max(diff_i, diff_j) + 1
    denominator = count - 1
    if denominator == 0:
        return 0
    segment_count = control_coords.shape[0] - 1
    emitted_count = count if segment_index == segment_count - 1 else count - 1
    total = 0
    for order in range(emitted_count):
        fraction = float(order) / float(denominator)
        radius = (
            worm_radii[segment_index] * (1.0 - fraction)
            + worm_radii[segment_index + 1] * fraction
        )
        for disk_index in range(disk_distances.shape[0]):
            if radius >= disk_distances[disk_index]:
                total += 1
    return total


@njit(cache=True)
def _write_expanded_segment_pixels(
    control_coords: np.ndarray,
    worm_radii: np.ndarray,
    segment_index: int,
    disk_rows: np.ndarray,
    disk_cols: np.ndarray,
    disk_distances: np.ndarray,
    expanded_rows: np.ndarray,
    expanded_cols: np.ndarray,
    write_index: int,
) -> int:
    row0 = int(control_coords[segment_index, 0])
    col0 = int(control_coords[segment_index, 1])
    row1 = int(control_coords[segment_index + 1, 0])
    col1 = int(control_coords[segment_index + 1, 1])
    diff_i = abs(row0 - row1)
    diff_j = abs(col0 - col1)
    count = max(diff_i, diff_j) + 1
    denominator = count - 1
    if denominator == 0:
        return write_index
    step_i = 1 if row1 > row0 else -1
    step_j = 1 if col1 > col0 else -1
    current_i = row0
    current_j = col0
    for point_index in range(count):
        if point_index > 0:
            if diff_i >= diff_j:
                # CP/centrosome Bresenham branch where row changes every step.
                remainder = diff_j * 2 - diff_i
                current_i = row0
                current_j = col0
                for n in range(1, point_index + 1):
                    if remainder >= 0:
                        current_j += step_j
                        remainder -= diff_i * 2
                    current_i += step_i
                    remainder += diff_j * 2
            else:
                remainder = diff_i * 2 - diff_j
                current_i = row0
                current_j = col0
                for n in range(1, point_index + 1):
                    if remainder >= 0:
                        current_i += step_i
                        remainder -= diff_j * 2
                    current_j += step_j
                    remainder += diff_i * 2
        if (
            segment_index != control_coords.shape[0] - 2
            and point_index == count - 1
        ):
            continue
        order = point_index
        fraction = float(order) / float(denominator)
        radius = (
            worm_radii[segment_index] * (1.0 - fraction)
            + worm_radii[segment_index + 1] * fraction
        )
        for disk_index in range(disk_distances.shape[0]):
            if radius >= disk_distances[disk_index]:
                expanded_rows[write_index] = current_i + disk_rows[disk_index]
                expanded_cols[write_index] = current_j + disk_cols[disk_index]
                write_index += 1
    return write_index


def _rebuild_worm_from_control_points_python(
    control_coords: np.ndarray,
    worm_radii: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Reference CP-style implementation kept for parity debugging."""

    index, count, rows, cols = _cellprofiler_line_points(
        control_coords[:-1, 0],
        control_coords[:-1, 1],
        control_coords[1:, 0],
        control_coords[1:, 1],
    )
    if len(rows) == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)

    rows = np.delete(rows, index[1:])
    cols = np.delete(cols, index[1:])
    index = index - np.arange(len(index))
    count = count - 1
    segment_keep = count != 0
    index = index[segment_keep]
    count = count[segment_keep]
    if len(index) == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)

    segment_indexes = np.zeros(len(rows), dtype=int)
    segment_indexes[index[1:]] = 1
    segment_indexes = np.cumsum(segment_indexes)
    order = np.arange(len(rows)) - index[segment_indexes]
    fractions = order.astype(float) / count[segment_indexes].astype(float)

    radii = np.asarray(worm_radii, dtype=float)
    if len(radii) < len(control_coords):
        radii = np.pad(radii, (0, len(control_coords) - len(radii)), mode="edge")
    radius = (
        radii[segment_indexes] * (1.0 - fractions)
        + radii[segment_indexes + 1] * fractions
    )
    max_radius = int(np.max(np.ceil(radius))) if len(radius) else 0
    if max_radius <= 0:
        valid = (rows >= 0) & (cols >= 0) & (rows < shape[0]) & (cols < shape[1])
        return rows[valid], cols[valid]

    delta_rows, delta_cols = np.mgrid[
        -max_radius : max_radius + 1,
        -max_radius : max_radius + 1,
    ]
    distances = np.sqrt((delta_rows * delta_rows + delta_cols * delta_cols).astype(float))
    disk = distances <= max_radius
    delta_rows = delta_rows[disk]
    delta_cols = delta_cols[disk]
    distances = distances[disk]

    expanded_rows = (rows[:, np.newaxis] + delta_rows[np.newaxis, :]).ravel()
    expanded_cols = (cols[:, np.newaxis] + delta_cols[np.newaxis, :]).ravel()
    keep = (radius[:, np.newaxis] >= distances[np.newaxis, :]).ravel()
    expanded_rows = expanded_rows[keep]
    expanded_cols = expanded_cols[keep]
    valid = (
        (expanded_rows >= 0)
        & (expanded_cols >= 0)
        & (expanded_rows < shape[0])
        & (expanded_cols < shape[1])
    )
    order = np.lexsort((expanded_rows, expanded_cols))
    expanded_rows = expanded_rows[order]
    expanded_cols = expanded_cols[order]
    unique = np.hstack(
        (
            [True],
            (expanded_rows[:-1] != expanded_rows[1:])
            | (expanded_cols[:-1] != expanded_cols[1:]),
        )
    )
    expanded_rows = expanded_rows[unique]
    expanded_cols = expanded_cols[unique]
    valid = (
        (expanded_rows >= 0)
        & (expanded_cols >= 0)
        & (expanded_rows < shape[0])
        & (expanded_cols < shape[1])
    )
    coords = np.column_stack((expanded_rows[valid], expanded_cols[valid]))
    if len(coords) == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    return coords[:, 0], coords[:, 1]


def _cellprofiler_line_points(
    pt0i: np.ndarray,
    pt0j: np.ndarray,
    pt1i: np.ndarray,
    pt1j: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return CP/centrosome Bresenham line points for vectorized segments."""
    pt0i = np.asarray(pt0i, dtype=int)
    pt0j = np.asarray(pt0j, dtype=int)
    pt1i = np.asarray(pt1i, dtype=int)
    pt1j = np.asarray(pt1j, dtype=int)
    if len(pt0i) == 0:
        empty = np.zeros(0, dtype=int)
        return empty, empty, empty, empty

    diff_i = np.abs(pt0i - pt1i)
    diff_j = np.abs(pt0j - pt1j)
    count = np.maximum(diff_i, diff_j).astype(int) + 1
    index = np.cumsum(count) - count
    step_i = (pt1i > pt0i).astype(int) * 2 - 1
    step_j = (pt1j > pt0j).astype(int) * 2 - 1
    n_points = int(index[-1] + count[-1])
    rows = np.zeros(n_points, dtype=int)
    cols = np.zeros(n_points, dtype=int)
    rows[index] = pt0i
    cols[index] = pt0j

    mask = diff_i >= diff_j
    if np.any(mask):
        _fill_line_axis_major(
            rows,
            cols,
            index[mask],
            count[mask],
            pt0i[mask],
            pt0j[mask],
            diff_i[mask],
            diff_j[mask],
            step_i[mask],
            step_j[mask],
            row_major=True,
        )

    mask = diff_j > diff_i
    if np.any(mask):
        _fill_line_axis_major(
            rows,
            cols,
            index[mask],
            count[mask],
            pt0i[mask],
            pt0j[mask],
            diff_i[mask],
            diff_j[mask],
            step_i[mask],
            step_j[mask],
            row_major=False,
        )
    return index, count, rows, cols


def _fill_line_axis_major(
    rows: np.ndarray,
    cols: np.ndarray,
    index: np.ndarray,
    count: np.ndarray,
    current_i: np.ndarray,
    current_j: np.ndarray,
    diff_i: np.ndarray,
    diff_j: np.ndarray,
    step_i: np.ndarray,
    step_j: np.ndarray,
    *,
    row_major: bool,
) -> None:
    last_n = int(np.max(count))
    current_i = current_i.copy()
    current_j = current_j.copy()
    remainder = (diff_j * 2 - diff_i) if row_major else (diff_i * 2 - diff_j)
    for n in range(1, last_n + 1):
        active = count > n
        remainder = remainder[active]
        current_i = current_i[active]
        current_j = current_j[active]
        index = index[active]
        count = count[active]
        diff_i = diff_i[active]
        diff_j = diff_j[active]
        step_i = step_i[active]
        step_j = step_j[active]

        remainder_mask = remainder >= 0
        if row_major:
            current_j[remainder_mask] += step_j[remainder_mask]
            remainder[remainder_mask] -= diff_i[remainder_mask] * 2
            current_i += step_i
            remainder += diff_j * 2
        else:
            current_i[remainder_mask] += step_i[remainder_mask]
            remainder[remainder_mask] -= diff_j[remainder_mask] * 2
            current_j += step_j
            remainder += diff_i * 2
        rows[index + n] = current_i
        cols[index + n] = current_j


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
