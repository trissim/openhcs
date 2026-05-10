"""
Converted from CellProfiler: UntangleWorms
Original: UntangleWorms module for untangling overlapping worms

This module untangles overlapping worms using a trained worm model.
It takes a binary image and labels the worms, untangling them and
associating all of a worm's pieces together.
"""

import numpy as np
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from heapq import heappop, heappush
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_opening,
    find_objects,
    label,
)

from openhcs.core.memory.decorators import numpy
from openhcs.interop.cellprofiler.worm_measurements import (
    WormControlPointMeasurementSchema,
    control_points_from_worm_measurement_rows,
)
from openhcs.core.runtime_semantics import ObjectLabelRepresentation
from openhcs.core.runtime_values import (
    ObjectLabelPayload,
    ObjectLabelSet,
    SparseIJVLabelRows,
    object_label_payload_from_source_image,
    object_label_set_from_source_image,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.processing.materialization import csv_materializer, segmentation_mask_rois

from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from benchmark.cellprofiler_library.functions.worm_geometry import (
    branchpoints,
    calculate_cumulative_lengths,
    endpoints,
    eight_connectivity,
    rebuild_worm_from_control_points_approx,
    sample_control_points,
    skeletonize_worm_mask,
    trace_skeleton_path,
)


class OverlapStyle(str, Enum):
    WITH_OVERLAP = "with_overlap"
    WITHOUT_OVERLAP = "without_overlap"
    BOTH = "both"


@dataclass(frozen=True, slots=True)
class WormLabelOutputRequest:
    sparse_overlapping: ObjectLabelSet
    overlapping: ObjectLabelPayload
    nonoverlapping: ObjectLabelPayload


class WormLabelOutputStrategy(
    EnumKeyedStrategyMixin[OverlapStyle],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Select UntangleWorms label outputs for one CellProfiler overlap style."""

    __registry_key__ = "overlap_style_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "overlap_style"
    __enum_label_attr__ = "overlap_style_label"
    overlap_style_label: ClassVar[str | None] = None
    overlap_style: ClassVar[OverlapStyle | None] = None

    @classmethod
    def for_overlap_style(cls, overlap_style: OverlapStyle) -> "WormLabelOutputStrategy":
        return cls.for_enum_member(overlap_style)

    @abstractmethod
    def outputs(
        self,
        request: WormLabelOutputRequest,
    ) -> tuple[ObjectLabelSet | ObjectLabelPayload, ObjectLabelPayload]:
        """Return the public overlapping and nonoverlapping label payloads."""


class WithOverlapWormLabelOutputStrategy(WormLabelOutputStrategy):
    overlap_style = OverlapStyle.WITH_OVERLAP

    def outputs(
        self,
        request: WormLabelOutputRequest,
    ) -> tuple[ObjectLabelSet | ObjectLabelPayload, ObjectLabelPayload]:
        return request.sparse_overlapping, request.overlapping


class WithoutOverlapWormLabelOutputStrategy(WormLabelOutputStrategy):
    overlap_style = OverlapStyle.WITHOUT_OVERLAP

    def outputs(
        self,
        request: WormLabelOutputRequest,
    ) -> tuple[ObjectLabelSet | ObjectLabelPayload, ObjectLabelPayload]:
        return request.nonoverlapping, request.nonoverlapping


class BothOverlapWormLabelOutputStrategy(WormLabelOutputStrategy):
    overlap_style = OverlapStyle.BOTH

    def outputs(
        self,
        request: WormLabelOutputRequest,
    ) -> tuple[ObjectLabelSet | ObjectLabelPayload, ObjectLabelPayload]:
        return request.sparse_overlapping, request.nonoverlapping


def coerce_overlap_style(value: str | OverlapStyle) -> OverlapStyle:
    """Normalize CellProfiler overlap-style literals into the typed enum."""
    if isinstance(value, OverlapStyle):
        return value
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    for style in OverlapStyle:
        literals = (
            style.name.lower(),
            style.value,
            style.value.replace("_", ""),
        )
        if normalized in literals:
            return style
    raise ValueError(
        "overlap_style must be one of "
        f"{', '.join(style.value for style in OverlapStyle)}; got {value!r}."
    )


def _get_angles(control_coords: np.ndarray) -> np.ndarray:
    """Extract angles at each interior control point"""
    if len(control_coords) < 3:
        return np.array([])
    
    segments_delta = control_coords[1:] - control_coords[:-1]
    segment_bearings = np.arctan2(segments_delta[:, 0], segments_delta[:, 1])
    angles = segment_bearings[1:] - segment_bearings[:-1]
    
    # Constrain angles to [-pi, pi]
    angles[angles > np.pi] -= 2 * np.pi
    angles[angles < -np.pi] += 2 * np.pi
    return angles


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(
    ("worm_measurements", csv_materializer(analysis_type="worm_analysis")),
    ("overlapping_labels", segmentation_mask_rois()),
    ("nonoverlapping_labels", segmentation_mask_rois()),
)
def untangle_worms(
    image: np.ndarray,
    overlap_style: OverlapStyle = OverlapStyle.WITHOUT_OVERLAP,
    min_worm_area: float = 100.0,
    max_worm_area: float = 5000.0,
    num_control_points: int = 21,
    cost_threshold: float = 100.0,
    min_path_length: float = 50.0,
    max_path_length: float = 500.0,
    overlap_weight: float = 5.0,
    leftover_weight: float = 10.0,
    median_worm_area: float | None = None,
    max_radius: float | None = None,
    max_skel_length: float | None = None,
    mean_angles: tuple[float, ...] | None = None,
    inv_angles_covariance_matrix: tuple[tuple[float, ...], ...] | None = None,
    radii_from_training: tuple[float, ...] | None = None,
    overlapping_object_name: str = "OverlappingWorms",
    nonoverlapping_object_name: str = "NonOverlappingWorms",
) -> tuple[
    np.ndarray,
    list[dict[str, float | int | str]],
    ObjectLabelSet | ObjectLabelPayload,
    ObjectLabelPayload,
]:
    """
    Untangle overlapping worms in a binary image.
    
    This function takes a binary image where foreground indicates worm shapes
    and attempts to identify and separate individual worms, even when they
    overlap or cross each other.
    
    Args:
        image: Binary input image (H, W) where foreground indicates worms
        overlap_style: How to handle overlapping regions:
            - "with_overlap": Include overlapping regions in both worms
            - "without_overlap": Exclude overlapping regions from both worms
            - "both": Generate both types of output
        min_worm_area: Minimum area for a valid worm (pixels)
        max_worm_area: Maximum area for a single worm (larger = cluster)
        num_control_points: Number of control points for worm shape model
        cost_threshold: Maximum shape cost for accepting a worm
        min_path_length: Minimum skeleton path length for a worm
        max_path_length: Maximum skeleton path length for a worm
        overlap_weight: Penalty weight for overlapping worm regions
        leftover_weight: Penalty weight for uncovered foreground
    
    Returns:
        Tuple of (original_image, measurements, overlapping_labels, nonoverlapping_labels)
    """
    overlap_style = coerce_overlap_style(overlap_style)
    mean_angles_array = _coerce_mean_angles(mean_angles, num_control_points)
    inv_angles_covariance_array = _coerce_inverse_covariance(
        inv_angles_covariance_matrix,
        num_control_points,
    )
    radii_array = _coerce_worm_radii(radii_from_training, num_control_points)

    # Ensure binary
    binary = image > 0
    
    # Label connected components
    labels, count = label(binary, structure=eight_connectivity())
    
    if count == 0:
        empty_labels, empty_nonoverlapping_labels = _worm_label_outputs(
            [],
            source_image=image,
            image_shape=image.shape,
            radii_from_training=radii_array,
            overlap_style=overlap_style,
            overlapping_object_name=overlapping_object_name,
        )
        return image, [], empty_labels, empty_nonoverlapping_labels
    
    # Skeletonize
    skeleton = skeletonize_worm_mask(binary)
    
    # Remove skeleton points at image edges
    eroded = binary_erosion(binary, structure=eight_connectivity())
    skeleton = skeletonize_worm_mask(skeleton & eroded)
    
    areas = np.bincount(labels.ravel())
    component_slices = find_objects(labels)
    all_path_coords: list[np.ndarray] = []
    
    for i, object_slice in enumerate(component_slices, start=1):
        if object_slice is None:
            continue
        component_area = areas[i]
        
        # Skip if too small
        if component_area < min_worm_area:
            continue
        
        row_slice, column_slice = object_slice
        local_labels = labels[object_slice]
        mask = local_labels == i
        component_skeleton = skeleton[object_slice] & mask
        
        if not np.any(component_skeleton):
            continue
        
        if component_area <= max_worm_area:
            path_coords = _longest_worm_graph_path_coords(
                mask,
                component_skeleton,
                max_length=max_path_length,
            )
            
            if len(path_coords) < 2:
                continue
            
            cumul_lengths = calculate_cumulative_lengths(path_coords)
            total_length = cumul_lengths[-1]
            if not _worm_shape_cost_passes(
                path_coords,
                total_length=total_length,
                num_control_points=num_control_points,
                mean_angles=mean_angles_array,
                inv_angles_covariance_matrix=inv_angles_covariance_array,
                cost_threshold=cost_threshold,
            ):
                continue
            
            all_path_coords.append(
                _offset_path_coords(
                    path_coords,
                    row_offset=row_slice.start,
                    column_offset=column_slice.start,
                )
            )
        else:
            graph = _worm_graph_from_binary(
                mask,
                component_skeleton,
                max_radius=max_radius,
                max_skel_length=max_skel_length,
            )
            paths = _all_worm_graph_paths(
                graph,
                min_length=min_path_length,
                max_length=max_path_length,
            )
            all_path_coords.extend(
                _offset_path_coords(
                    path_coords,
                    row_offset=row_slice.start,
                    column_offset=column_slice.start,
                )
                for path_coords in _select_worm_cluster_paths(
                    graph,
                    paths,
                    component_area=int(component_area),
                    median_worm_area=median_worm_area,
                    num_control_points=num_control_points,
                    mean_angles=mean_angles_array,
                    inv_angles_covariance_matrix=inv_angles_covariance_array,
                    cost_threshold=cost_threshold,
                    overlap_weight=overlap_weight,
                    leftover_weight=leftover_weight,
                    min_path_length=min_path_length,
                    max_path_length=max_path_length,
                )
            )
    
    overlapping_labels, nonoverlapping_labels = _worm_label_outputs(
        all_path_coords,
        source_image=image,
        image_shape=image.shape,
        radii_from_training=radii_array,
        overlap_style=overlap_style,
        overlapping_object_name=overlapping_object_name,
    )
    
    measurements = _worm_descriptor_rows(
        all_path_coords,
        num_control_points=num_control_points,
        overlapping_object_name=overlapping_object_name,
        nonoverlapping_object_name=nonoverlapping_object_name,
        overlap_style=overlap_style,
    )
    
    return image, measurements, overlapping_labels, nonoverlapping_labels


@dataclass(frozen=True, slots=True)
class _WormGraphPath:
    segments: tuple[int, ...]
    branch_areas: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _WormGraph:
    segments: tuple[tuple[np.ndarray, np.ndarray], ...]
    segment_lengths: np.ndarray
    incidence_matrix: np.ndarray
    incidence_directions: np.ndarray
    incident_branch_areas: tuple[np.ndarray, ...]
    incident_segments: tuple[np.ndarray, ...]


def _worm_graph_from_binary(
    binary_image: np.ndarray,
    skeleton: np.ndarray,
    *,
    max_radius: float | None,
    max_skel_length: float | None,
) -> _WormGraph:
    """Build CP's branch-area/segment graph without centrosome calls."""
    branch_areas = branchpoints(skeleton)
    if max_radius is not None and max_radius > 0:
        far = binary_erosion(binary_image, structure=_cellprofiler_strel_disk(max_radius))
        far = binary_opening(far, structure=eight_connectivity())
        far_labels, _count = label(far, structure=eight_connectivity())
        if far_labels.size:
            far_counts = np.bincount(
                far_labels.ravel(),
                weights=branch_areas.ravel().astype(float),
            )
            far[far_counts[far_labels] < 2] = False
            branch_areas |= far
    branch_areas = binary_dilation(branch_areas, structure=eight_connectivity())
    segments = skeleton & ~branch_areas
    if max_skel_length is not None and np.any(segments):
        segments, branch_areas = _insert_long_segment_breakpoints(
            segments,
            branch_areas,
            max_skel_length=max(int(max_skel_length), 2),
        )
    return _worm_graph_from_branching_areas(branch_areas, segments)


def _insert_long_segment_breakpoints(
    segments: np.ndarray,
    branch_areas: np.ndarray,
    *,
    max_skel_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    i, j, labels, order, _distance, segment_count = _trace_segments(segments)
    if segment_count == 0:
        return segments, branch_areas
    max_order = np.zeros(segment_count + 1, dtype=int)
    for label_id in range(1, segment_count + 1):
        label_orders = order[labels == label_id]
        if len(label_orders):
            max_order[label_id] = int(np.max(label_orders))
    big_segment = max_order >= max_skel_length
    segment_count_per_label = np.maximum(
        ((max_order + max_skel_length - 1) / max_skel_length).astype(int),
        1,
    )
    segment_length = np.maximum(((max_order + 1) / segment_count_per_label).astype(int), 1)
    new_breakpoints = (
        (order % segment_length[labels] == segment_length[labels] - 1)
        & (order != max_order[labels])
        & big_segment[labels]
    )
    if not np.any(new_breakpoints):
        return segments, branch_areas
    new_branch_areas = np.zeros(segments.shape, dtype=bool)
    new_branch_areas[i[new_breakpoints], j[new_breakpoints]] = True
    new_branch_areas = binary_dilation(
        new_branch_areas,
        structure=eight_connectivity(),
    )
    return segments & ~new_branch_areas, branch_areas | new_branch_areas


def _worm_graph_from_branching_areas(
    branch_areas: np.ndarray,
    segments: np.ndarray,
) -> _WormGraph:
    branch_labels, branch_count = label(branch_areas, structure=eight_connectivity())
    i, j, labels, order, _distance, segment_count = _trace_segments(segments)
    if segment_count == 0:
        empty_incidence = np.zeros((branch_count, 0), dtype=bool)
        return _WormGraph(
            segments=(),
            segment_lengths=np.zeros(0, dtype=float),
            incidence_matrix=empty_incidence,
            incidence_directions=empty_incidence.copy(),
            incident_branch_areas=(),
            incident_segments=tuple(np.zeros(0, dtype=int) for _ in range(branch_count)),
        )

    sort_order = np.lexsort((order, labels))
    i = i[sort_order]
    j = j[sort_order]
    labels = labels[sort_order]
    order = order[sort_order]
    counts = np.bincount(labels)[1:]
    indexes = np.cumsum(counts) - counts
    coords = np.column_stack((i, j))
    graph_segments = tuple(
        (
            coords[indexes[index] : indexes[index] + counts[index]],
            coords[indexes[index] : indexes[index] + counts[index]][::-1],
        )
        for index in range(len(counts))
    )
    start_labels = np.zeros(segments.shape, dtype=int)
    starts = order == 0
    start_labels[i[starts], j[starts]] = labels[starts]
    ends = np.cumsum(counts) - 1
    end_labels = np.zeros(segments.shape, dtype=int)
    end_labels[i[ends], j[ends]] = labels[ends]
    incidence_directions = _incidence_matrix(
        branch_labels,
        branch_count,
        start_labels,
        segment_count,
    )
    incidence_matrix = _incidence_matrix(
        branch_labels,
        branch_count,
        end_labels,
        segment_count,
    )
    incidence_matrix |= incidence_directions
    segment_lengths = np.array(
        [calculate_cumulative_lengths(segment[0])[-1] for segment in graph_segments],
        dtype=float,
    )
    incident_segments = tuple(
        np.flatnonzero(incidence_matrix[branch_index, :])
        for branch_index in range(branch_count)
    )
    incident_branch_areas = tuple(
        np.flatnonzero(incidence_matrix[:, segment_index])
        for segment_index in range(segment_count)
    )
    return _WormGraph(
        segments=graph_segments,
        segment_lengths=segment_lengths,
        incidence_matrix=incidence_matrix,
        incidence_directions=incidence_directions,
        incident_branch_areas=incident_branch_areas,
        incident_segments=incident_segments,
    )


def _cellprofiler_strel_disk(radius: float) -> np.ndarray:
    """Return CellProfiler/centrosome's disk footprint semantics."""
    integer_radius = int(radius)
    rows, columns = np.mgrid[
        -integer_radius : integer_radius + 1,
        -integer_radius : integer_radius + 1,
    ]
    return (rows * rows + columns * columns) <= radius * radius


def _offset_path_coords(
    coords: np.ndarray,
    *,
    row_offset: int,
    column_offset: int,
) -> np.ndarray:
    if len(coords) == 0:
        return coords
    offset = np.array((row_offset, column_offset), dtype=coords.dtype)
    return coords + offset


def _longest_worm_graph_path_coords(
    binary_image: np.ndarray,
    skeleton: np.ndarray,
    *,
    max_length: float,
) -> np.ndarray:
    graph = _worm_graph_from_binary(
        binary_image,
        skeleton,
        max_radius=None,
        max_skel_length=None,
    )
    longest_coords = np.zeros((0, 2), dtype=int)
    longest_length = 0.0
    for path in _all_worm_graph_paths(graph, min_length=0.0, max_length=max_length):
        coords = _graph_path_to_pixel_coords(graph, path)
        path_length = float(calculate_cumulative_lengths(coords)[-1])
        if path_length >= longest_length:
            longest_coords = coords
            longest_length = path_length
    return longest_coords


def _trace_segments(
    segments: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    foreground = np.argwhere(segments)
    if len(foreground) == 0:
        empty_i = np.zeros(0, dtype=int)
        empty_distance = np.zeros(0, dtype=float)
        return empty_i, empty_i, empty_i, empty_i, empty_distance, 0

    row_min, column_min = foreground.min(axis=0)
    row_max, column_max = foreground.max(axis=0) + 1
    local_segments = segments[row_min:row_max, column_min:column_max]
    segment_labels, segment_count = label(local_segments, structure=eight_connectivity())
    if segment_count == 0:
        empty_i = np.zeros(0, dtype=int)
        empty_distance = np.zeros(0, dtype=float)
        return empty_i, empty_i, empty_i, empty_i, empty_distance, 0
    endpoint_mask = endpoints(local_segments)
    traced: list[tuple[int, int, int, float]] = []
    object_slices = find_objects(segment_labels)
    for label_id, object_slice in enumerate(object_slices, start=1):
        if object_slice is None:
            continue
        row_slice, column_slice = object_slice
        local_labels = segment_labels[object_slice]
        segment_mask = local_labels == label_id
        endpoint_coords = np.argwhere(endpoint_mask[object_slice] & segment_mask)
        if len(endpoint_coords):
            start = endpoint_coords[
                np.lexsort((endpoint_coords[:, 1], endpoint_coords[:, 0]))
            ][0]
        else:
            coords = np.argwhere(segment_mask)
            start = coords[np.lexsort((coords[:, 1], coords[:, 0]))][0]
        distances = _segment_geodesic_distances(segment_mask, tuple(start))
        coords = np.argwhere(segment_mask)
        for row, column in coords:
            traced.append(
                (
                    int(row + row_slice.start + row_min),
                    int(column + column_slice.start + column_min),
                    label_id,
                    float(distances[row, column]),
                )
            )
    traced_array = np.array(traced, dtype=float)
    sort_order = np.lexsort((traced_array[:, 3], traced_array[:, 2]))
    traced_array = traced_array[sort_order]
    labels = traced_array[:, 2].astype(int)
    segment_order = np.arange(len(labels), dtype=int)
    areas = np.bincount(labels)
    indexes = np.cumsum(areas) - areas
    segment_order -= indexes[labels]
    return (
        traced_array[:, 0].astype(int),
        traced_array[:, 1].astype(int),
        labels,
        segment_order,
        traced_array[:, 3],
        segment_count,
    )


def _segment_geodesic_distances(
    segment_mask: np.ndarray,
    start: tuple[int, int],
) -> np.ndarray:
    distances = np.full(segment_mask.shape, np.inf, dtype=float)
    distances[start] = 0.0
    queue: list[tuple[float, int, int]] = [(0.0, int(start[0]), int(start[1]))]
    while queue:
        distance, row, column = heappop(queue)
        if distance != distances[row, column]:
            continue
        for row_delta in (-1, 0, 1):
            for column_delta in (-1, 0, 1):
                if row_delta == 0 and column_delta == 0:
                    continue
                next_row = row + row_delta
                next_column = column + column_delta
                if (
                    next_row < 0
                    or next_column < 0
                    or next_row >= segment_mask.shape[0]
                    or next_column >= segment_mask.shape[1]
                    or not segment_mask[next_row, next_column]
                ):
                    continue
                step = float(np.hypot(row_delta, column_delta))
                next_distance = distance + step
                if next_distance < distances[next_row, next_column]:
                    distances[next_row, next_column] = next_distance
                    heappush(queue, (next_distance, next_row, next_column))
    return distances


def _trace_segment_from_start(segment_mask: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    path = [start]
    visited = {start}
    current = start
    while True:
        next_points = []
        row, column = current
        for row_delta in (-1, 0, 1):
            for column_delta in (-1, 0, 1):
                if row_delta == 0 and column_delta == 0:
                    continue
                point = (row + row_delta, column + column_delta)
                if (
                    0 <= point[0] < segment_mask.shape[0]
                    and 0 <= point[1] < segment_mask.shape[1]
                    and segment_mask[point]
                    and point not in visited
                ):
                    next_points.append(point)
        if not next_points:
            break
        next_points.sort()
        current = next_points[0]
        path.append(current)
        visited.add(current)
    if len(visited) != int(np.count_nonzero(segment_mask)):
        remaining = [tuple(coord) for coord in np.argwhere(segment_mask) if tuple(coord) not in visited]
        path.extend(sorted(remaining))
    return np.asarray(path, dtype=int)


def _incidence_matrix(
    branch_labels: np.ndarray,
    branch_count: int,
    endpoint_labels: np.ndarray,
    segment_count: int,
) -> np.ndarray:
    incidence = np.zeros((branch_count, segment_count), dtype=bool)
    if branch_count == 0 or segment_count == 0:
        return incidence
    rows, columns = np.nonzero(branch_labels)
    height, width = branch_labels.shape
    for row, column in zip(rows, columns, strict=True):
        branch_id = int(branch_labels[row, column])
        for row_delta in (-1, 0, 1):
            neighbor_row = row + row_delta
            if neighbor_row < 0 or neighbor_row >= height:
                continue
            for column_delta in (-1, 0, 1):
                neighbor_column = column + column_delta
                if neighbor_column < 0 or neighbor_column >= width:
                    continue
                segment_id = int(endpoint_labels[neighbor_row, neighbor_column])
                if segment_id > 0:
                    incidence[branch_id - 1, segment_id - 1] = True
    return incidence


def _all_worm_graph_paths(
    graph: _WormGraph,
    *,
    min_length: float,
    max_length: float,
) -> list[_WormGraphPath]:
    paths: list[_WormGraphPath] = []
    for segment_index, current_length in enumerate(graph.segment_lengths):
        if current_length >= min_length:
            paths.append(_WormGraphPath((segment_index,), ()))
        unfinished_branches = tuple(
            (int(branch_index),)
            for branch_index in graph.incident_branch_areas[segment_index]
        )
        paths.extend(
            _all_worm_graph_paths_recur(
                graph,
                unfinished_segments=(segment_index,),
                unfinished_branch_areas=unfinished_branches,
                current_length=float(current_length),
                min_length=min_length,
                max_length=max_length,
            )
        )
    return paths


def _all_worm_graph_paths_recur(
    graph: _WormGraph,
    *,
    unfinished_segments: tuple[int, ...],
    unfinished_branch_areas: tuple[tuple[int, ...], ...],
    current_length: float,
    min_length: float,
    max_length: float,
) -> list[_WormGraphPath]:
    if not unfinished_segments:
        return []
    paths: list[_WormGraphPath] = []
    last_segment = unfinished_segments[-1]
    for unfinished_branch in unfinished_branch_areas:
        end_branch = unfinished_branch[-1]
        direction = graph.incidence_directions[end_branch, last_segment]
        last_coord = graph.segments[last_segment][int(direction)][-1]
        for segment_index in graph.incident_segments[end_branch]:
            segment_index = int(segment_index)
            if segment_index in unfinished_segments:
                continue
            direction = not graph.incidence_directions[end_branch, segment_index]
            first_coord = graph.segments[segment_index][int(direction)][0]
            gap_length = float(np.sqrt(np.sum((last_coord - first_coord) ** 2)))
            next_length = current_length + gap_length + graph.segment_lengths[segment_index]
            if next_length > max_length:
                continue
            next_segments = (*unfinished_segments, segment_index)
            if segment_index > unfinished_segments[0] and next_length >= min_length:
                paths.append(_WormGraphPath(next_segments, unfinished_branch))
            next_branches = tuple(
                (*unfinished_branch, int(branch_index))
                for branch_index in graph.incident_branch_areas[segment_index]
                if int(branch_index) != end_branch and int(branch_index) not in unfinished_branch
            )
            paths.extend(
                _all_worm_graph_paths_recur(
                    graph,
                    unfinished_segments=next_segments,
                    unfinished_branch_areas=next_branches,
                    current_length=float(next_length),
                    min_length=min_length,
                    max_length=max_length,
                )
            )
    return paths


def _graph_path_to_pixel_coords(
    graph: _WormGraph,
    path: _WormGraphPath,
) -> np.ndarray:
    if len(path.segments) == 1:
        return graph.segments[path.segments[0]][0]
    direction = graph.incidence_directions[path.branch_areas[0], path.segments[0]]
    result = [graph.segments[path.segments[0]][int(direction)]]
    for branch_area, segment in zip(path.branch_areas, path.segments[1:], strict=True):
        direction = not graph.incidence_directions[branch_area, segment]
        result.append(graph.segments[segment][int(direction)])
    return np.vstack(result)


def _select_worm_cluster_paths(
    graph: _WormGraph,
    paths: list[_WormGraphPath],
    *,
    component_area: int,
    median_worm_area: float | None,
    num_control_points: int,
    mean_angles: np.ndarray,
    inv_angles_covariance_matrix: np.ndarray,
    cost_threshold: float,
    overlap_weight: float,
    leftover_weight: float,
    min_path_length: float,
    max_path_length: float,
) -> list[np.ndarray]:
    paths_and_costs: list[tuple[_WormGraphPath, float]] = []
    for path in paths:
        coords = _graph_path_to_pixel_coords(graph, path)
        total_length = float(calculate_cumulative_lengths(coords)[-1])
        if total_length > max_path_length or total_length < min_path_length:
            continue
        cost = _worm_shape_cost(
            coords,
            total_length=total_length,
            num_control_points=num_control_points,
            mean_angles=mean_angles,
            inv_angles_covariance_matrix=inv_angles_covariance_matrix,
        )
        if cost < cost_threshold:
            paths_and_costs.append((path, cost))
    if not paths_and_costs:
        return []
    costs = np.asarray([cost for _path, cost in paths_and_costs], dtype=float)
    order = np.lexsort([costs])
    if len(order) > 500:
        order = order[:500]
    costs = costs[order]
    path_segment_matrix = np.zeros((len(graph.segments), len(order)), dtype=bool)
    for column, ordered_index in enumerate(order):
        path_segment_matrix[list(paths_and_costs[int(ordered_index)][0].segments), column] = True
    max_worms = _cluster_max_worms(
        component_area,
        median_worm_area=median_worm_area,
    )
    selected_indexes = _fast_select_worm_paths(
        costs,
        path_segment_matrix,
        graph.segment_lengths,
        overlap_weight=overlap_weight,
        leftover_weight=leftover_weight,
        max_worms=max_worms,
    )
    selected_paths = [
        paths_and_costs[int(order[selected_index])][0]
        for selected_index in selected_indexes
    ]
    return [_graph_path_to_pixel_coords(graph, path) for path in selected_paths]


def _cluster_max_worms(
    component_area: int,
    *,
    median_worm_area: float | None,
) -> int:
    if median_worm_area is None or median_worm_area <= 0:
        return 1
    return max(1, int(np.ceil(component_area / median_worm_area)))


def _fast_select_worm_paths(
    costs: np.ndarray,
    path_segment_matrix: np.ndarray,
    segment_lengths: np.ndarray,
    *,
    overlap_weight: float,
    leftover_weight: float,
    max_worms: int,
) -> list[int]:
    current_best_subset: list[int] = []
    current_best_cost = float(np.sum(segment_lengths) * leftover_weight)
    current_path_segment_matrix = path_segment_matrix.astype(int)
    current_path_choices = np.eye(len(costs), dtype=bool)
    for _level in range(min(max_worms, len(costs))):
        (
            current_best_subset,
            current_best_cost,
            current_path_segment_matrix,
            current_path_choices,
        ) = _select_one_worm_path_level(
            costs,
            path_segment_matrix,
            segment_lengths,
            current_best_subset,
            current_best_cost,
            current_path_segment_matrix,
            current_path_choices,
            overlap_weight=overlap_weight,
            leftover_weight=leftover_weight,
        )
        if np.prod(current_path_choices.shape) == 0:
            break
    return current_best_subset


def _select_one_worm_path_level(
    costs: np.ndarray,
    path_segment_matrix: np.ndarray,
    segment_lengths: np.ndarray,
    current_best_subset: list[int],
    current_best_cost: float,
    current_path_segment_matrix: np.ndarray,
    current_path_choices: np.ndarray,
    *,
    overlap_weight: float,
    leftover_weight: float,
) -> tuple[list[int], float, np.ndarray, np.ndarray]:
    partial_costs = (
        np.sum(costs[:, np.newaxis] * current_path_choices, axis=0)
        + np.sum(
            np.maximum(current_path_segment_matrix - 1, 0)
            * segment_lengths[:, np.newaxis],
            axis=0,
        )
        * overlap_weight
    )
    total_costs = (
        partial_costs
        + np.sum(
            (current_path_segment_matrix == 0) * segment_lengths[:, np.newaxis],
            axis=0,
        )
        * leftover_weight
    )
    order = np.lexsort([total_costs])
    if len(order) and total_costs[order[0]] < current_best_cost:
        current_best_subset = np.flatnonzero(
            current_path_choices[:, order[0]]
        ).tolist()
        current_best_cost = float(total_costs[order[0]])
    mask = partial_costs < current_best_cost
    if not np.any(mask):
        return (
            current_best_subset,
            current_best_cost,
            np.zeros((len(costs), 0), dtype=int),
            np.zeros((len(costs), 0), dtype=bool),
        )
    order = order[mask[order]]
    if len(order) * len(costs) > 5000:
        order = order[: (1 + 5000 // len(costs))]
    current_path_segment_matrix = current_path_segment_matrix[:, order]
    current_path_choices = current_path_choices[:, order]
    i, j = np.mgrid[0 : len(costs), 0 : len(costs)]
    disallow = i >= j
    allowed = np.dot(disallow, current_path_choices) == 0
    if not np.any(allowed):
        return (
            current_best_subset,
            current_best_cost,
            np.zeros((len(costs), 0), dtype=int),
            np.zeros((len(costs), 0), dtype=bool),
        )
    i, j = np.argwhere(allowed).transpose()
    return (
        current_best_subset,
        current_best_cost,
        path_segment_matrix[:, i] + current_path_segment_matrix[:, j],
        np.eye(len(costs), dtype=bool)[:, i] | current_path_choices[:, j],
    )


def _worm_label_outputs(
    all_path_coords: list[np.ndarray],
    *,
    source_image: object,
    image_shape: tuple[int, int],
    radii_from_training: np.ndarray,
    overlap_style: OverlapStyle,
    overlapping_object_name: str,
) -> tuple[ObjectLabelSet | ObjectLabelPayload, ObjectLabelPayload]:
    ijv_parts: list[np.ndarray] = []
    overlap_hits = np.zeros(image_shape, dtype=np.int16)
    overlapping = np.zeros(image_shape, dtype=np.int32)
    for object_number, path_coords in enumerate(all_path_coords, start=1):
        rows, cols = _reconstructed_worm_pixels(
            path_coords,
            image_shape=image_shape,
            radii_from_training=radii_from_training,
        )
        if len(rows) == 0:
            continue
        ijv_parts.append(
            np.column_stack(
                (
                    rows.astype(np.int32, copy=False),
                    cols.astype(np.int32, copy=False),
                    np.full(len(rows), object_number, dtype=np.int32),
                )
            )
        )
        overlap_hits[rows, cols] += 1
        overlapping[rows, cols] = object_number
    nonoverlapping = overlapping.copy()
    nonoverlapping[overlap_hits != 1] = 0
    ijv = (
        np.vstack(ijv_parts).astype(np.int32, copy=False)
        if ijv_parts
        else np.zeros((0, 3), dtype=np.int32)
    )
    sparse_overlapping = object_label_set_from_source_image(
        source_image,
        name=overlapping_object_name,
        labels=SparseIJVLabelRows(ijv),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
    )
    overlapping_payload = object_label_payload_from_source_image(
        source_image,
        overlapping,
        declared_object_count=len(all_path_coords),
    )
    nonoverlapping_payload = object_label_payload_from_source_image(
        source_image,
        nonoverlapping,
        declared_object_count=len(all_path_coords),
    )
    return WormLabelOutputStrategy.for_overlap_style(overlap_style).outputs(
        WormLabelOutputRequest(
            sparse_overlapping=sparse_overlapping,
            overlapping=overlapping_payload,
            nonoverlapping=nonoverlapping_payload,
        )
    )


def _reconstructed_worm_pixels(
    path_coords: np.ndarray,
    *,
    image_shape: tuple[int, int],
    radii_from_training: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(path_coords) < 2:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    control_coords = sample_control_points(
        path_coords,
        calculate_cumulative_lengths(path_coords),
        len(radii_from_training),
    )
    return rebuild_worm_from_control_points_approx(
        control_coords,
        radii_from_training,
        image_shape,
    )


def _coerce_mean_angles(
    mean_angles: tuple[float, ...] | None,
    num_control_points: int,
) -> np.ndarray:
    if mean_angles is None:
        return np.zeros(max(num_control_points - 1, 0), dtype=float)
    return np.asarray(mean_angles, dtype=float)


def _coerce_inverse_covariance(
    inv_angles_covariance_matrix: tuple[tuple[float, ...], ...] | None,
    num_control_points: int,
) -> np.ndarray:
    if inv_angles_covariance_matrix is None:
        return np.eye(max(num_control_points - 1, 0), dtype=float)
    return np.asarray(inv_angles_covariance_matrix, dtype=float)


def _coerce_worm_radii(
    radii_from_training: tuple[float, ...] | None,
    num_control_points: int,
) -> np.ndarray:
    if radii_from_training is None:
        return np.ones(num_control_points, dtype=float)
    radii = np.asarray(radii_from_training, dtype=float)
    if len(radii) == num_control_points:
        return radii
    if len(radii) == 0:
        return np.ones(num_control_points, dtype=float)
    if len(radii) < num_control_points:
        return np.pad(radii, (0, num_control_points - len(radii)), mode="edge")
    return radii[:num_control_points]


def _worm_shape_cost_passes(
    path_coords: np.ndarray,
    *,
    total_length: float,
    num_control_points: int,
    mean_angles: np.ndarray,
    inv_angles_covariance_matrix: np.ndarray,
    cost_threshold: float,
) -> bool:
    cost = _worm_shape_cost(
        path_coords,
        total_length=total_length,
        num_control_points=num_control_points,
        mean_angles=mean_angles,
        inv_angles_covariance_matrix=inv_angles_covariance_matrix,
    )
    return cost < cost_threshold


def _worm_shape_cost(
    path_coords: np.ndarray,
    *,
    total_length: float,
    num_control_points: int,
    mean_angles: np.ndarray,
    inv_angles_covariance_matrix: np.ndarray,
) -> float:
    control_coords = sample_control_points(
        path_coords,
        calculate_cumulative_lengths(path_coords),
        num_control_points,
    )
    if len(mean_angles) != num_control_points - 1:
        return 0.0
    if inv_angles_covariance_matrix.shape != (num_control_points - 1, num_control_points - 1):
        return 0.0
    angles = _get_angles(control_coords)
    feature_vector = np.hstack((angles, [total_length])) - mean_angles
    return float(feature_vector @ inv_angles_covariance_matrix @ feature_vector)


def _worm_descriptor_rows(
    all_path_coords: list[np.ndarray],
    *,
    num_control_points: int,
    overlapping_object_name: str,
    nonoverlapping_object_name: str,
    overlap_style: OverlapStyle,
) -> list[dict[str, float | int | str]]:
    """Return CellProfiler-compatible per-object worm descriptor rows."""
    rows: list[dict[str, float | int | str]] = []
    object_names = _measurement_object_names(
        overlap_style,
        overlapping_object_name,
        nonoverlapping_object_name,
    )
    for object_number, path_coords in enumerate(all_path_coords, start=1):
        descriptor = _worm_descriptor_row(
            path_coords,
            object_number=object_number,
            num_control_points=num_control_points,
        )
        for object_name in object_names:
            rows.append({"object_name": object_name, **descriptor})
    return rows


def _measurement_object_names(
    overlap_style: OverlapStyle,
    overlapping_object_name: str,
    nonoverlapping_object_name: str,
) -> tuple[str, ...]:
    if overlap_style is OverlapStyle.WITH_OVERLAP:
        return (overlapping_object_name,)
    if overlap_style is OverlapStyle.WITHOUT_OVERLAP:
        return (nonoverlapping_object_name,)
    return (overlapping_object_name, nonoverlapping_object_name)


def _worm_descriptor_row(
    path_coords: np.ndarray,
    *,
    object_number: int,
    num_control_points: int,
) -> dict[str, float | int]:
    cumul_lengths = calculate_cumulative_lengths(path_coords)
    if len(path_coords) < 2:
        control_coords = np.zeros((num_control_points, 2), dtype=float)
        angles = np.zeros(max(num_control_points - 2, 0), dtype=float)
        length = 0.0
    else:
        control_coords = sample_control_points(
            path_coords,
            cumul_lengths,
            num_control_points,
        )
        angles = _get_angles(control_coords)
        length = float(cumul_lengths[-1])

    row: dict[str, float | int] = {
        "object_number": object_number,
        "worm_length": length,
    }
    for index, angle in enumerate(angles, start=1):
        row[f"worm_angle_{index}"] = float(angle)
    row.update(
        WormControlPointMeasurementSchema(
            num_control_points=num_control_points,
        ).row_fields(control_coords)
    )
    return row
