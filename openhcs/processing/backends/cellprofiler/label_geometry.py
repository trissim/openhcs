"""Native label-geometry kernels shared by CellProfiler-compatible backends."""

from __future__ import annotations

import math

from llvmlite import ir
import numpy as np
from numba import njit, types
from numba.extending import intrinsic

from openhcs.processing.backends.numpy_runtime import (
    numpy_avx512_skx_svml_symbol_available,
)

_NUMPY_124_SVML_ACOS_AVAILABLE = numpy_avx512_skx_svml_symbol_available(
    "__svml_acos8"
)


@intrinsic
def _numpy_124_scalar_arccos(typing_context, value):
    """Emit the standard SVML arccos operation used by NumPy 1.24."""

    del typing_context, value
    signature = types.float64(types.float64)

    def codegen(context, builder, resolved_signature, arguments):
        del context, resolved_signature
        double_type = ir.DoubleType()
        vector_type = ir.VectorType(double_type, 8)
        function_type = ir.FunctionType(vector_type, (vector_type,))
        if "__svml_acos8" in builder.module.globals:
            arccos_function = builder.module.globals["__svml_acos8"]
        else:
            arccos_function = ir.Function(
                builder.module,
                function_type,
                name="__svml_acos8",
            )
        input_vector = ir.Constant(
            vector_type,
            (ir.Constant(double_type, 0.0),) * 8,
        )
        lane_zero = ir.Constant(ir.IntType(32), 0)
        input_vector = builder.insert_element(
            input_vector,
            arguments[0],
            lane_zero,
        )
        result = builder.call(arccos_function, (input_vector,))
        return builder.extract_element(result, lane_zero)

    return signature, codegen


@njit(cache=True)
def _numpy_124_svml_arccos(values: np.ndarray) -> np.ndarray:
    """Return NumPy 1.24 standard-SVML arccos values for a flat array."""
    result = np.empty(values.size, dtype=np.float64)
    for index in range(values.size):
        result[index] = _numpy_124_scalar_arccos(values[index])
    return result


@njit(cache=True)
def _numpy_124_portable_arccos(values: np.ndarray) -> np.ndarray:
    """Return NumPy 1.24 scalar-libm arccos values for a flat array."""
    result = np.empty(values.size, dtype=np.float64)
    for index in range(values.size):
        result[index] = math.acos(values[index])
    return result


def _numpy_124_arccos(values: np.ndarray) -> np.ndarray:
    """Return the NumPy 1.24 angle primitive available on this architecture."""
    value_array = np.asarray(values, dtype=np.float64)
    if _NUMPY_124_SVML_ACOS_AVAILABLE:
        return _numpy_124_svml_arccos(value_array)
    return _numpy_124_portable_arccos(value_array)


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
    """Return circles using CellProfiler 4.2.8.1 vertex ordering."""
    import centrosome.cpmorphology
    import scipy.ndimage

    label_array = np.asarray(labels, dtype=np.int32)
    if label_array.ndim != 2:
        raise ValueError(
            "Minimum enclosing circle requires 2-D labels, got "
            f"{label_array.ndim}D."
        )
    indexes = np.asarray(label_ids, dtype=np.int32)
    if indexes.size == 0:
        return np.zeros((0, 2), dtype=float), np.zeros(0, dtype=float)

    hull, point_count = centrosome.cpmorphology.convex_hull(label_array, indexes)
    centers = np.zeros((indexes.size, 2), dtype=float)
    radii = np.zeros(indexes.size, dtype=float)
    point_index = np.zeros(indexes.size, dtype=int)
    point_index[1:] = np.cumsum(point_count[:-1])

    centers[point_count == 0] = np.nan
    if np.all(point_count == 0):
        return centers, radii
    singletons = point_count == 1
    centers[singletons] = hull[point_index[singletons], 1:]
    pairs = point_count == 2
    centers[pairs] = (
        hull[point_index[pairs], 1:] + hull[point_index[pairs] + 1, 1:]
    ) / 2
    distance = centers[pairs] - hull[point_index[pairs], 1:]
    radii[pairs] = np.sqrt(distance[:, 0] ** 2 + distance[:, 1] ** 2)

    active = point_count > 2
    s0_idx = point_index.copy()
    s1_idx = s0_idx + 1
    anti_indexes = np.zeros(int(np.max(indexes)) + 1, dtype=int)
    anti_indexes[indexes] = np.arange(indexes.size)
    anti_indexes_per_point = anti_indexes[hull[:, 0]]
    within_label_indexes = np.arange(hull.shape[0]) - point_index[
        anti_indexes_per_point
    ]

    while np.any(active):
        labels_to_consider = indexes[active]
        anti_indexes_to_consider = np.zeros(
            int(np.max(labels_to_consider)) + 1,
            dtype=int,
        )
        anti_indexes_to_consider[labels_to_consider] = np.arange(
            labels_to_consider.size
        )
        active_vertices = active[anti_indexes_per_point] & (
            within_label_indexes >= 2
        )
        vertices = hull[active_vertices, 1:]
        vertex_labels = hull[active_vertices, 0]
        vertex_indexes = np.flatnonzero(active_vertices).astype(np.int32)
        considered_vertex_indexes = anti_indexes_to_consider[vertex_labels]
        s0 = hull[s0_idx[active], 1:][considered_vertex_indexes]
        s1 = hull[s1_idx[active], 1:][considered_vertex_indexes]

        s01 = (s0 - s1).astype(float)
        vs0 = (vertices - s0).astype(float)
        vs1 = (vertices - s1).astype(float)
        angle_vs1s0 = np.abs(
            _numpy_124_arccos(
                np.sum(s01 * vs1, axis=1)
                / np.sqrt(np.sum(s01**2, axis=1) * np.sum(vs1**2, axis=1))
            )
        )
        angle_vs0s1 = np.abs(
            _numpy_124_arccos(
                np.sum(-s01 * vs0, axis=1)
                / np.sqrt(np.sum(s01**2, axis=1) * np.sum(vs0**2, axis=1))
            )
        )
        angle_s0vs1 = np.pi - angle_vs1s0 - angle_vs0s1
        if np.any(angle_s0vs1 < 0):
            raise RuntimeError("Minimum enclosing circle produced a negative angle.")

        min_angle = np.asarray(
            scipy.ndimage.minimum(
                angle_s0vs1,
                vertex_labels,
                labels_to_consider,
            )
        ).reshape(-1)
        min_position = _grouped_minimum_positions(
            angle_s0vs1,
            vertex_labels,
            indexes,
        )
        vertex_counts = np.asarray(
            scipy.ndimage.sum(
                active_vertices,
                hull[:, 0],
                labels_to_consider,
            )
        ).reshape(-1)

        case_1 = (min_angle >= np.pi / 2) | (vertex_counts == 0)
        if np.any(case_1):
            finished = np.zeros(indexes.size, dtype=bool)
            finished[anti_indexes[labels_to_consider[case_1]]] = True
            finished_s0 = hull[s0_idx[finished], 1:].astype(float)
            finished_s1 = hull[s1_idx[finished], 1:].astype(float)
            centers[finished] = (finished_s0 + finished_s1) / 2
            radii[finished] = (
                np.sqrt(np.sum((finished_s0 - finished_s1) ** 2, axis=1)) / 2
            )
            active[finished] = False

        case_2 = active.copy()
        case_2[angle_vs1s0[min_position] > np.pi / 2] = False
        case_2[angle_vs0s1[min_position] > np.pi / 2] = False
        case_2[angle_s0vs1[min_position] > np.pi / 2] = False
        if np.any(case_2):
            case_s0 = hull[s0_idx[case_2], 1:].astype(float)
            case_s1 = hull[s1_idx[case_2], 1:].astype(float)
            case_vertex = vertices[min_position[case_2]].astype(float)
            y_axis, x_axis = 0, 1
            denominator = 2 * (
                case_s0[:, x_axis] * (case_s1[:, y_axis] - case_vertex[:, y_axis])
                + case_s1[:, x_axis]
                * (case_vertex[:, y_axis] - case_s0[:, y_axis])
                + case_vertex[:, x_axis]
                * (case_s0[:, y_axis] - case_s1[:, y_axis])
            )
            centers[case_2, x_axis] = (
                np.sum(case_s0**2, axis=1)
                * (case_s1[:, y_axis] - case_vertex[:, y_axis])
                + np.sum(case_s1**2, axis=1)
                * (case_vertex[:, y_axis] - case_s0[:, y_axis])
                + np.sum(case_vertex**2, axis=1)
                * (case_s0[:, y_axis] - case_s1[:, y_axis])
            ) / denominator
            centers[case_2, y_axis] = (
                np.sum(case_s0**2, axis=1)
                * (case_vertex[:, x_axis] - case_s1[:, x_axis])
                + np.sum(case_s1**2, axis=1)
                * (case_s0[:, x_axis] - case_vertex[:, x_axis])
                + np.sum(case_vertex**2, axis=1)
                * (case_s1[:, x_axis] - case_s0[:, x_axis])
            ) / denominator
            radii[case_2] = np.sqrt(
                np.sum((case_s0 - centers[case_2]) ** 2, axis=1)
            )
            active[case_2] = False

        if np.any(active):
            labels_to_consider = indexes[active]
            indexes_to_consider = anti_indexes[labels_to_consider]
            obtuse_vertex_indexes = vertex_indexes[min_position[active]]
            angle_vs0s1_to_consider = angle_vs0s1[min_position[active]]
            s0_is_obtuse = angle_vs0s1_to_consider > np.pi / 2
            if np.any(s0_is_obtuse):
                selected_vertices = obtuse_vertex_indexes[s0_is_obtuse]
                selected_s0 = s0_idx[indexes_to_consider[s0_is_obtuse]]
                within_label_indexes[selected_s0] = within_label_indexes[
                    selected_vertices
                ]
                s0_idx[indexes_to_consider[s0_is_obtuse]] = selected_vertices
                within_label_indexes[selected_vertices] = 0
            s1_is_obtuse = ~s0_is_obtuse
            if np.any(s1_is_obtuse):
                selected_vertices = obtuse_vertex_indexes[s1_is_obtuse]
                selected_s1 = s1_idx[indexes_to_consider[s1_is_obtuse]]
                within_label_indexes[selected_s1] = within_label_indexes[
                    selected_vertices
                ]
                s1_idx[indexes_to_consider[s1_is_obtuse]] = selected_vertices
                within_label_indexes[selected_vertices] = 1
    return centers, radii


def _grouped_minimum_positions(
    values: np.ndarray,
    labels: np.ndarray,
    indexes: np.ndarray,
) -> np.ndarray:
    """Return SciPy minimum positions with NumPy 1.24 quicksort tie ordering."""
    value_array = np.asarray(values)
    label_array = np.asarray(labels)
    index_array = np.asarray(indexes, dtype=int)
    max_label = int(np.max(label_array)) if label_array.size else 0
    requested = index_array.ravel().copy()
    found = (requested >= 0) & (requested <= max_label)
    requested[~found] = max_label + 1
    order = _numpy_124_scalar_argsort(value_array.ravel())
    sorted_labels = label_array.ravel()[order]
    sorted_positions = np.arange(value_array.size, dtype=int)[order]
    minimum_positions = np.zeros(max_label + 2, dtype=int)
    minimum_positions[sorted_labels[::-1]] = sorted_positions[::-1]
    return minimum_positions[requested].reshape(index_array.shape)


def _numpy_124_scalar_argsort(values: np.ndarray) -> np.ndarray:
    """Return the NumPy 1.24 scalar quicksort permutation for float data."""
    order = np.arange(values.size, dtype=int)
    if order.size <= 1:
        return order

    stack: list[tuple[int, int, int]] = []
    left = 0
    right = int(order.size - 1)
    depth = (int(order.size).bit_length() - 1) * 2
    while True:
        if depth < 0:
            suborder = order[left : right + 1]
            order[left : right + 1] = suborder[
                np.argsort(values[suborder], kind="heapsort")
            ]
            if not stack:
                break
            left, right, depth = stack.pop()
            continue

        while (right - left) > 15:
            middle = left + ((right - left) >> 1)
            if values[order[middle]] < values[order[left]]:
                order[middle], order[left] = order[left], order[middle]
            if values[order[right]] < values[order[middle]]:
                order[right], order[middle] = order[middle], order[right]
            if values[order[middle]] < values[order[left]]:
                order[middle], order[left] = order[left], order[middle]
            pivot = values[order[middle]]
            lower = left
            upper = right - 1
            order[middle], order[upper] = order[upper], order[middle]
            while True:
                lower += 1
                while values[order[lower]] < pivot:
                    lower += 1
                upper -= 1
                while pivot < values[order[upper]]:
                    upper -= 1
                if lower >= upper:
                    break
                order[lower], order[upper] = order[upper], order[lower]
            pivot_index = right - 1
            order[lower], order[pivot_index] = order[pivot_index], order[lower]
            if (lower - left) < (right - lower):
                stack.append((lower + 1, right, depth - 1))
                right = lower - 1
            else:
                stack.append((left, lower - 1, depth - 1))
                left = lower + 1
            depth -= 1

        for lower in range(left + 1, right + 1):
            value_index = int(order[lower])
            pivot = values[value_index]
            upper = lower
            previous = lower - 1
            while upper > left and pivot < values[order[previous]]:
                order[upper] = order[previous]
                upper -= 1
                previous -= 1
            order[upper] = value_index

        if not stack:
            break
        left, right, depth = stack.pop()
    return order


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


__all__ = [
    "feret_diameters_from_labels",
    "minimum_enclosing_circle_from_labels",
]
