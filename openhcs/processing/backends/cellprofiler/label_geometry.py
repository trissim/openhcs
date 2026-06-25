"""Native label-geometry kernels shared by CellProfiler-compatible backends."""

from __future__ import annotations

import numpy as np
from numba import njit


def feret_diameters_from_labels(
    labels: np.ndarray,
    label_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return CP-compatible minimum and maximum Feret diameters."""
    label_array = np.asarray(labels, dtype=np.int32)
    if label_array.ndim != 2:
        raise ValueError(
            f"Feret diameters require 2-D labels, got {label_array.ndim}D."
        )
    return _feret_diameters_from_labels_numba(
        np.ascontiguousarray(label_array),
        np.asarray(label_ids, dtype=np.int32),
    )


def minimum_enclosing_circle_from_labels(
    labels: np.ndarray,
    label_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return CP-compatible minimum enclosing circle centers and radii."""
    label_array = np.asarray(labels, dtype=np.int32)
    if label_array.ndim != 2:
        raise ValueError(
            "Minimum enclosing circle requires 2-D labels, got "
            f"{label_array.ndim}D."
        )
    return _minimum_enclosing_circle_from_labels_numba(
        np.ascontiguousarray(label_array),
        np.asarray(label_ids, dtype=np.int32),
    )


@njit(cache=True)
def _feret_diameters_from_labels_numba(
    labels: np.ndarray,
    label_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    object_count = label_ids.size
    min_diameters = np.zeros(object_count, dtype=np.float64)
    max_diameters = np.zeros(object_count, dtype=np.float64)
    if object_count == 0:
        return min_diameters, max_diameters

    counts, offsets, point_y, point_x = _outline_points_by_label_numba(
        labels,
        label_ids,
    )
    if point_y.size == 0:
        return min_diameters, max_diameters

    for object_index in range(object_count):
        count = counts[object_index]
        if count <= 1:
            continue
        hull_y = np.empty(count * 2, dtype=np.int64)
        hull_x = np.empty(count * 2, dtype=np.int64)
        hull_count = _monotone_label_hull_numba(
            point_y,
            point_x,
            offsets[object_index],
            count,
            hull_y,
            hull_x,
        )
        min_diameter, max_diameter = _feret_diameters_from_hull_numba(
            hull_y,
            hull_x,
            hull_count,
        )
        min_diameters[object_index] = min_diameter
        max_diameters[object_index] = max_diameter
    return min_diameters, max_diameters


@njit(cache=True)
def _minimum_enclosing_circle_from_labels_numba(
    labels: np.ndarray,
    label_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    object_count = label_ids.size
    centers = np.zeros((object_count, 2), dtype=np.float64)
    radii = np.zeros(object_count, dtype=np.float64)
    if object_count == 0:
        return centers, radii

    counts, offsets, point_y, point_x = _outline_points_by_label_numba(
        labels,
        label_ids,
    )
    for object_index in range(object_count):
        count = counts[object_index]
        if count == 0:
            centers[object_index, 0] = np.nan
            centers[object_index, 1] = np.nan
            continue
        hull_y = np.empty(max(1, count * 2), dtype=np.int64)
        hull_x = np.empty(max(1, count * 2), dtype=np.int64)
        hull_count = _monotone_label_hull_numba(
            point_y,
            point_x,
            offsets[object_index],
            count,
            hull_y,
            hull_x,
        )
        center_y, center_x, radius = _minimum_enclosing_circle_from_hull_numba(
            hull_y,
            hull_x,
            hull_count,
        )
        centers[object_index, 0] = center_y
        centers[object_index, 1] = center_x
        radii[object_index] = radius
    return centers, radii


@njit(cache=True)
def _outline_points_by_label_numba(
    labels: np.ndarray,
    label_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    object_count = label_ids.size
    counts = np.zeros(object_count, dtype=np.int64)
    offsets = np.zeros(object_count + 1, dtype=np.int64)
    if object_count == 0:
        return (
            counts,
            offsets,
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
        )

    max_label = 0
    for object_index in range(object_count):
        label_id = int(label_ids[object_index])
        if label_id > max_label:
            max_label = label_id
    height, width = labels.shape
    for y in range(height):
        for x in range(width):
            label_id = int(labels[y, x])
            if label_id > max_label:
                max_label = label_id
    if max_label <= 0:
        return (
            counts,
            offsets,
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
        )

    label_to_output = np.full(max_label + 1, -1, dtype=np.int64)
    for object_index in range(object_count):
        label_id = int(label_ids[object_index])
        if label_id > 0 and label_id <= max_label:
            label_to_output[label_id] = object_index

    for y in range(height):
        for x in range(width):
            label_id = int(labels[y, x])
            if label_id <= 0 or label_id > max_label:
                continue
            object_index = label_to_output[label_id]
            if object_index < 0:
                continue
            if _is_label_outline_pixel_numba(labels, y, x, label_id):
                counts[object_index] += 1

    for object_index in range(object_count):
        offsets[object_index + 1] = offsets[object_index] + counts[object_index]
    point_total = offsets[object_count]
    point_y = np.empty(point_total, dtype=np.int64)
    point_x = np.empty(point_total, dtype=np.int64)
    cursor = offsets.copy()
    for y in range(height):
        for x in range(width):
            label_id = int(labels[y, x])
            if label_id <= 0 or label_id > max_label:
                continue
            object_index = label_to_output[label_id]
            if object_index < 0:
                continue
            if not _is_label_outline_pixel_numba(labels, y, x, label_id):
                continue
            point_index = cursor[object_index]
            point_y[point_index] = y
            point_x[point_index] = x
            cursor[object_index] += 1
    return counts, offsets, point_y, point_x


@njit(cache=True)
def _is_label_outline_pixel_numba(
    labels: np.ndarray,
    y: int,
    x: int,
    label_id: int,
) -> bool:
    if y == 0 or x == 0 or y == labels.shape[0] - 1 or x == labels.shape[1] - 1:
        return True
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dy == 0 and dx == 0:
                continue
            if int(labels[y + dy, x + dx]) != label_id:
                return True
    return False


@njit(cache=True)
def _monotone_label_hull_numba(
    point_y: np.ndarray,
    point_x: np.ndarray,
    start: int,
    count: int,
    hull_y: np.ndarray,
    hull_x: np.ndarray,
) -> int:
    if count <= 1:
        if count == 1:
            hull_y[0] = point_y[start]
            hull_x[0] = point_x[start]
        return count

    hull_count = 0
    for offset in range(count):
        py = point_y[start + offset]
        px = point_x[start + offset]
        while hull_count >= 2 and _cross_label_hull_points_numba(
            hull_y[hull_count - 2],
            hull_x[hull_count - 2],
            hull_y[hull_count - 1],
            hull_x[hull_count - 1],
            py,
            px,
        ) <= 0:
            hull_count -= 1
        hull_y[hull_count] = py
        hull_x[hull_count] = px
        hull_count += 1

    lower_count = hull_count
    for offset in range(count - 2, -1, -1):
        py = point_y[start + offset]
        px = point_x[start + offset]
        while hull_count > lower_count and _cross_label_hull_points_numba(
            hull_y[hull_count - 2],
            hull_x[hull_count - 2],
            hull_y[hull_count - 1],
            hull_x[hull_count - 1],
            py,
            px,
        ) <= 0:
            hull_count -= 1
        hull_y[hull_count] = py
        hull_x[hull_count] = px
        hull_count += 1

    if hull_count > 1:
        hull_count -= 1
    return hull_count


@njit(cache=True)
def _cross_label_hull_points_numba(
    ay: int,
    ax: int,
    by: int,
    bx: int,
    cy: int,
    cx: int,
) -> int:
    return (by - ay) * (cx - ax) - (bx - ax) * (cy - ay)


@njit(cache=True)
def _feret_diameters_from_hull_numba(
    hull_y: np.ndarray,
    hull_x: np.ndarray,
    hull_count: int,
) -> tuple[float, float]:
    if hull_count <= 1:
        return 0.0, 0.0

    max_distance_sq = 0.0
    for first_index in range(hull_count):
        first_y = float(hull_y[first_index])
        first_x = float(hull_x[first_index])
        for second_index in range(first_index + 1, hull_count):
            dy = float(hull_y[second_index]) - first_y
            dx = float(hull_x[second_index]) - first_x
            distance_sq = dy * dy + dx * dx
            if distance_sq > max_distance_sq:
                max_distance_sq = distance_sq

    if hull_count == 2:
        return 0.0, np.sqrt(max_distance_sq)

    min_width_sq = np.inf
    for edge_index in range(hull_count):
        next_index = 0 if edge_index == hull_count - 1 else edge_index + 1
        y0 = float(hull_y[edge_index])
        x0 = float(hull_x[edge_index])
        edge_y = float(hull_y[next_index]) - y0
        edge_x = float(hull_x[next_index]) - x0
        edge_length_sq = edge_y * edge_y + edge_x * edge_x
        if edge_length_sq == 0.0:
            continue

        edge_width_sq = 0.0
        for point_index in range(hull_count):
            point_y = float(hull_y[point_index]) - y0
            point_x = float(hull_x[point_index]) - x0
            cross = edge_y * point_x - edge_x * point_y
            distance_sq = (cross * cross) / edge_length_sq
            if distance_sq > edge_width_sq:
                edge_width_sq = distance_sq
        if edge_width_sq < min_width_sq:
            min_width_sq = edge_width_sq

    if not np.isfinite(min_width_sq):
        min_width_sq = 0.0
    return np.sqrt(min_width_sq), np.sqrt(max_distance_sq)


@njit(cache=True)
def _minimum_enclosing_circle_from_hull_numba(
    hull_y: np.ndarray,
    hull_x: np.ndarray,
    hull_count: int,
) -> tuple[float, float, float]:
    if hull_count <= 0:
        return np.nan, np.nan, 0.0
    if hull_count == 1:
        return float(hull_y[0]), float(hull_x[0]), 0.0

    best_y = 0.0
    best_x = 0.0
    best_radius_sq = np.inf
    for first_index in range(hull_count):
        y0 = float(hull_y[first_index])
        x0 = float(hull_x[first_index])
        for second_index in range(first_index + 1, hull_count):
            center_y = 0.5 * (y0 + float(hull_y[second_index]))
            center_x = 0.5 * (x0 + float(hull_x[second_index]))
            dy = y0 - center_y
            dx = x0 - center_x
            radius_sq = dy * dy + dx * dx
            if radius_sq >= best_radius_sq:
                continue
            if _circle_covers_hull_numba(
                hull_y,
                hull_x,
                hull_count,
                center_y,
                center_x,
                radius_sq,
            ):
                best_y = center_y
                best_x = center_x
                best_radius_sq = radius_sq

    for first_index in range(hull_count):
        for second_index in range(first_index + 1, hull_count):
            for third_index in range(second_index + 1, hull_count):
                valid, center_y, center_x, radius_sq = _circle_from_three_points_numba(
                    float(hull_y[first_index]),
                    float(hull_x[first_index]),
                    float(hull_y[second_index]),
                    float(hull_x[second_index]),
                    float(hull_y[third_index]),
                    float(hull_x[third_index]),
                )
                if not valid or radius_sq >= best_radius_sq:
                    continue
                if _circle_covers_hull_numba(
                    hull_y,
                    hull_x,
                    hull_count,
                    center_y,
                    center_x,
                    radius_sq,
                ):
                    best_y = center_y
                    best_x = center_x
                    best_radius_sq = radius_sq

    if not np.isfinite(best_radius_sq):
        return float(hull_y[0]), float(hull_x[0]), 0.0
    return best_y, best_x, np.sqrt(best_radius_sq)


@njit(cache=True)
def _circle_from_three_points_numba(
    ay: float,
    ax: float,
    by: float,
    bx: float,
    cy: float,
    cx: float,
) -> tuple[bool, float, float, float]:
    determinant = 2.0 * (
        ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)
    )
    if abs(determinant) < 1e-12:
        return False, 0.0, 0.0, 0.0
    center_x = (
        (ax * ax + ay * ay) * (by - cy)
        + (bx * bx + by * by) * (cy - ay)
        + (cx * cx + cy * cy) * (ay - by)
    ) / determinant
    center_y = (
        (ax * ax + ay * ay) * (cx - bx)
        + (bx * bx + by * by) * (ax - cx)
        + (cx * cx + cy * cy) * (bx - ax)
    ) / determinant
    dy = ay - center_y
    dx = ax - center_x
    return True, center_y, center_x, dy * dy + dx * dx


@njit(cache=True)
def _circle_covers_hull_numba(
    hull_y: np.ndarray,
    hull_x: np.ndarray,
    hull_count: int,
    center_y: float,
    center_x: float,
    radius_sq: float,
) -> bool:
    limit = radius_sq + 1e-8
    for point_index in range(hull_count):
        dy = float(hull_y[point_index]) - center_y
        dx = float(hull_x[point_index]) - center_x
        if dy * dy + dx * dx > limit:
            return False
    return True


__all__ = [
    "feret_diameters_from_labels",
    "minimum_enclosing_circle_from_labels",
]
