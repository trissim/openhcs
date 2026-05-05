"""Shared geometry helpers for absorbed CellProfiler worm modules."""

from __future__ import annotations

import numpy as np
from skimage.draw import line
from scipy.interpolate import interp1d
from scipy.ndimage import convolve


def eight_connectivity() -> np.ndarray:
    """Return an 8-connectivity structuring element."""
    return np.ones((3, 3), bool)


def skeletonize_worm_mask(binary_image: np.ndarray) -> np.ndarray:
    """Skeletonize a worm mask using morphological thinning."""
    from skimage.morphology import skeletonize

    return skeletonize(binary_image > 0)


def branchpoints(skeleton: np.ndarray) -> np.ndarray:
    """Find branchpoints in a skeleton."""
    neighbors = convolve(skeleton.astype(int), _NEIGHBOR_KERNEL, mode="constant")
    return skeleton & (neighbors - _NEIGHBOR_CENTER_WEIGHT > 2)


def endpoints(skeleton: np.ndarray) -> np.ndarray:
    """Find endpoints in a skeleton."""
    neighbors = convolve(skeleton.astype(int), _NEIGHBOR_KERNEL, mode="constant")
    return skeleton & ((neighbors - _NEIGHBOR_CENTER_WEIGHT) == 1)


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

    line_rows: list[np.ndarray] = []
    line_cols: list[np.ndarray] = []
    line_segment_indexes: list[np.ndarray] = []
    line_segment_fractions: list[np.ndarray] = []
    for segment_index, (start, stop) in enumerate(
        zip(control_coords[:-1], control_coords[1:], strict=True)
    ):
        rows, cols = line(
            int(round(start[0])),
            int(round(start[1])),
            int(round(stop[0])),
            int(round(stop[1])),
        )
        if segment_index:
            rows = rows[1:]
            cols = cols[1:]
        if len(rows) == 0:
            continue
        denominator = max(len(rows) - 1, 1)
        fractions = np.arange(len(rows), dtype=float) / float(denominator)
        line_rows.append(rows)
        line_cols.append(cols)
        line_segment_indexes.append(np.full(len(rows), segment_index, dtype=int))
        line_segment_fractions.append(fractions)
    if not line_rows:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)

    rows = np.concatenate(line_rows)
    cols = np.concatenate(line_cols)
    segment_indexes = np.concatenate(line_segment_indexes)
    fractions = np.concatenate(line_segment_fractions)
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
    coords = np.unique(
        np.column_stack((expanded_rows[valid], expanded_cols[valid])),
        axis=0,
    )
    if len(coords) == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    return coords[:, 0], coords[:, 1]


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


_NEIGHBOR_CENTER_WEIGHT = 10
_NEIGHBOR_KERNEL = np.array(
    [
        [1, 1, 1],
        [1, _NEIGHBOR_CENTER_WEIGHT, 1],
        [1, 1, 1],
    ]
)
