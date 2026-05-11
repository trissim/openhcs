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
from openhcs.processing.backends.cellprofiler.worm_geometry import (
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

    @abstractmethod
    def measurement_object_names(
        self,
        *,
        overlapping_object_name: str,
        nonoverlapping_object_name: str,
    ) -> tuple[str, ...]:
        """Return object names that should receive UntangleWorms measurements."""


class WithOverlapWormLabelOutputStrategy(WormLabelOutputStrategy):
    overlap_style = OverlapStyle.WITH_OVERLAP

    def outputs(
        self,
        request: WormLabelOutputRequest,
    ) -> tuple[ObjectLabelSet | ObjectLabelPayload, ObjectLabelPayload]:
        return request.sparse_overlapping, request.overlapping

    def measurement_object_names(
        self,
        *,
        overlapping_object_name: str,
        nonoverlapping_object_name: str,
    ) -> tuple[str, ...]:
        return (overlapping_object_name,)


class WithoutOverlapWormLabelOutputStrategy(WormLabelOutputStrategy):
    overlap_style = OverlapStyle.WITHOUT_OVERLAP

    def outputs(
        self,
        request: WormLabelOutputRequest,
    ) -> tuple[ObjectLabelSet | ObjectLabelPayload, ObjectLabelPayload]:
        return request.nonoverlapping, request.nonoverlapping

    def measurement_object_names(
        self,
        *,
        overlapping_object_name: str,
        nonoverlapping_object_name: str,
    ) -> tuple[str, ...]:
        return (nonoverlapping_object_name,)


class BothOverlapWormLabelOutputStrategy(WormLabelOutputStrategy):
    overlap_style = OverlapStyle.BOTH

    def outputs(
        self,
        request: WormLabelOutputRequest,
    ) -> tuple[ObjectLabelSet | ObjectLabelPayload, ObjectLabelPayload]:
        return request.sparse_overlapping, request.nonoverlapping

    def measurement_object_names(
        self,
        *,
        overlapping_object_name: str,
        nonoverlapping_object_name: str,
    ) -> tuple[str, ...]:
        return (overlapping_object_name, nonoverlapping_object_name)


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


@dataclass(frozen=True, slots=True)
class WormControlPointGeometry:
    """CP-compatible geometry derived from sampled worm control points."""

    control_coords: np.ndarray

    @property
    def angles(self) -> np.ndarray:
        """Extract angles at each interior control point."""
        if len(self.control_coords) < 3:
            return np.array([])

        segments_delta = self.control_coords[1:] - self.control_coords[:-1]
        segment_bearings = np.arctan2(segments_delta[:, 0], segments_delta[:, 1])
        angles = segment_bearings[1:] - segment_bearings[:-1]
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
            if not WormShapeCostRequest(
                path_coords=path_coords,
                total_length=total_length,
                num_control_points=num_control_points,
                mean_angles=mean_angles_array,
                inv_angles_covariance_matrix=inv_angles_covariance_array,
            ).passes(cost_threshold):
                continue
            
            all_path_coords.append(
                _offset_path_coords(
                    path_coords,
                    row_offset=row_slice.start,
                    column_offset=column_slice.start,
                )
            )
        else:
            graph = WormGraphFromBinaryRequest(
                binary_image=mask,
                skeleton=component_skeleton,
                max_radius=max_radius,
                max_skel_length=max_skel_length,
            ).build()
            paths = graph.paths_between_lengths(
                min_length=min_path_length,
                max_length=max_path_length,
            )
            all_path_coords.extend(
                _offset_path_coords(
                    path_coords,
                    row_offset=row_slice.start,
                    column_offset=column_slice.start,
                )
                for path_coords in WormClusterPathSelectionPolicy(
                    median_worm_area=median_worm_area,
                    component_area=int(component_area),
                    num_control_points=num_control_points,
                    mean_angles=mean_angles_array,
                    inv_angles_covariance_matrix=inv_angles_covariance_array,
                    cost_threshold=cost_threshold,
                    overlap_weight=overlap_weight,
                    leftover_weight=leftover_weight,
                    min_path_length=min_path_length,
                    max_path_length=max_path_length,
                ).select(graph, paths)
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
class WormGraphPath:
    """Ordered path through a CP worm graph."""

    segments: tuple[int, ...]
    branch_areas: tuple[int, ...]

    def to_pixel_coords(self, graph: "WormGraph") -> np.ndarray:
        if len(self.segments) == 1:
            return graph.segments[self.segments[0]][0]
        direction = graph.incidence_directions[self.branch_areas[0], self.segments[0]]
        result = [graph.segments[self.segments[0]][int(direction)]]
        for branch_area, segment in zip(
            self.branch_areas,
            self.segments[1:],
            strict=True,
        ):
            direction = not graph.incidence_directions[branch_area, segment]
            result.append(graph.segments[segment][int(direction)])
        return np.vstack(result)


@dataclass(frozen=True, slots=True)
class WormGraph:
    """CP worm branch-area graph with path enumeration semantics."""

    segments: tuple[tuple[np.ndarray, np.ndarray], ...]
    segment_lengths: np.ndarray
    incidence_matrix: np.ndarray
    incidence_directions: np.ndarray
    incident_branch_areas: tuple[np.ndarray, ...]
    incident_segments: tuple[np.ndarray, ...]

    def paths_between_lengths(
        self,
        *,
        min_length: float,
        max_length: float,
    ) -> list[WormGraphPath]:
        paths: list[WormGraphPath] = []
        for segment_index, current_length in enumerate(self.segment_lengths):
            if current_length >= min_length:
                paths.append(WormGraphPath((segment_index,), ()))
            unfinished_branches = tuple(
                (int(branch_index),)
                for branch_index in self.incident_branch_areas[segment_index]
            )
            paths.extend(
                self._paths_from(
                    unfinished_segments=(segment_index,),
                    unfinished_branch_areas=unfinished_branches,
                    current_length=float(current_length),
                    min_length=min_length,
                    max_length=max_length,
                )
            )
        return paths

    def _paths_from(
        self,
        *,
        unfinished_segments: tuple[int, ...],
        unfinished_branch_areas: tuple[tuple[int, ...], ...],
        current_length: float,
        min_length: float,
        max_length: float,
    ) -> list[WormGraphPath]:
        if not unfinished_segments:
            return []
        paths: list[WormGraphPath] = []
        last_segment = unfinished_segments[-1]
        for unfinished_branch in unfinished_branch_areas:
            end_branch = unfinished_branch[-1]
            direction = self.incidence_directions[end_branch, last_segment]
            last_coord = self.segments[last_segment][int(direction)][-1]
            for segment_index in self.incident_segments[end_branch]:
                segment_index = int(segment_index)
                if segment_index in unfinished_segments:
                    continue
                direction = not self.incidence_directions[end_branch, segment_index]
                first_coord = self.segments[segment_index][int(direction)][0]
                gap_length = float(np.sqrt(np.sum((last_coord - first_coord) ** 2)))
                next_length = (
                    current_length
                    + gap_length
                    + self.segment_lengths[segment_index]
                )
                if next_length > max_length:
                    continue
                next_segments = (*unfinished_segments, segment_index)
                if segment_index > unfinished_segments[0] and next_length >= min_length:
                    paths.append(WormGraphPath(next_segments, unfinished_branch))
                next_branches = tuple(
                    (*unfinished_branch, int(branch_index))
                    for branch_index in self.incident_branch_areas[segment_index]
                    if int(branch_index) != end_branch
                    and int(branch_index) not in unfinished_branch
                )
                paths.extend(
                    self._paths_from(
                        unfinished_segments=next_segments,
                        unfinished_branch_areas=next_branches,
                        current_length=float(next_length),
                        min_length=min_length,
                        max_length=max_length,
                    )
                )
        return paths


@dataclass(frozen=True, slots=True)
class WormGraphFromBinaryRequest:
    """Inputs for CP-style worm branch-area/segment graph construction."""

    binary_image: np.ndarray
    skeleton: np.ndarray
    max_radius: float | None
    max_skel_length: float | None

    def build(self) -> WormGraph:
        branch_areas = branchpoints(self.skeleton)
        if self.max_radius is not None and self.max_radius > 0:
            far = binary_erosion(
                self.binary_image,
                structure=_cellprofiler_strel_disk(self.max_radius),
            )
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
        segments = self.skeleton & ~branch_areas
        if self.max_skel_length is not None and np.any(segments):
            segments, branch_areas = _insert_long_segment_breakpoints(
                segments,
                branch_areas,
                max_skel_length=max(int(self.max_skel_length), 2),
            )
        return _worm_graph_from_branching_areas(branch_areas, segments)


@dataclass(frozen=True, slots=True)
class WormSegmentTrace:
    """Ordered pixels, labels, and distances for traced worm graph segments."""

    rows: np.ndarray
    columns: np.ndarray
    labels: np.ndarray
    order: np.ndarray
    distance: np.ndarray
    segment_count: int

    @classmethod
    def from_segments(cls, segments: np.ndarray) -> "WormSegmentTrace":
        foreground = np.argwhere(segments)
        if len(foreground) == 0:
            empty_i = np.zeros(0, dtype=int)
            empty_distance = np.zeros(0, dtype=float)
            return cls(empty_i, empty_i, empty_i, empty_i, empty_distance, 0)

        row_min, column_min = foreground.min(axis=0)
        row_max, column_max = foreground.max(axis=0) + 1
        local_segments = segments[row_min:row_max, column_min:column_max]
        segment_labels, segment_count = label(
            local_segments,
            structure=eight_connectivity(),
        )
        if segment_count == 0:
            empty_i = np.zeros(0, dtype=int)
            empty_distance = np.zeros(0, dtype=float)
            return cls(empty_i, empty_i, empty_i, empty_i, empty_distance, 0)
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
        return cls(
            traced_array[:, 0].astype(int),
            traced_array[:, 1].astype(int),
            labels,
            segment_order,
            traced_array[:, 3],
            segment_count,
        )


def _insert_long_segment_breakpoints(
    segments: np.ndarray,
    branch_areas: np.ndarray,
    *,
    max_skel_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    trace = WormSegmentTrace.from_segments(segments)
    if trace.segment_count == 0:
        return segments, branch_areas
    max_order = np.zeros(trace.segment_count + 1, dtype=int)
    for label_id in range(1, trace.segment_count + 1):
        label_orders = trace.order[trace.labels == label_id]
        if len(label_orders):
            max_order[label_id] = int(np.max(label_orders))
    big_segment = max_order >= max_skel_length
    segment_count_per_label = np.maximum(
        ((max_order + max_skel_length - 1) / max_skel_length).astype(int),
        1,
    )
    segment_length = np.maximum(((max_order + 1) / segment_count_per_label).astype(int), 1)
    new_breakpoints = (
        (trace.order % segment_length[trace.labels] == segment_length[trace.labels] - 1)
        & (trace.order != max_order[trace.labels])
        & big_segment[trace.labels]
    )
    if not np.any(new_breakpoints):
        return segments, branch_areas
    new_branch_areas = np.zeros(segments.shape, dtype=bool)
    new_branch_areas[trace.rows[new_breakpoints], trace.columns[new_breakpoints]] = True
    new_branch_areas = binary_dilation(
        new_branch_areas,
        structure=eight_connectivity(),
    )
    return segments & ~new_branch_areas, branch_areas | new_branch_areas


def _worm_graph_from_branching_areas(
    branch_areas: np.ndarray,
    segments: np.ndarray,
) -> WormGraph:
    branch_labels, branch_count = label(branch_areas, structure=eight_connectivity())
    trace = WormSegmentTrace.from_segments(segments)
    if trace.segment_count == 0:
        empty_incidence = np.zeros((branch_count, 0), dtype=bool)
        return WormGraph(
            segments=(),
            segment_lengths=np.zeros(0, dtype=float),
            incidence_matrix=empty_incidence,
            incidence_directions=empty_incidence.copy(),
            incident_branch_areas=(),
            incident_segments=tuple(np.zeros(0, dtype=int) for _ in range(branch_count)),
        )

    sort_order = np.lexsort((trace.order, trace.labels))
    i = trace.rows[sort_order]
    j = trace.columns[sort_order]
    labels = trace.labels[sort_order]
    order = trace.order[sort_order]
    segment_count = trace.segment_count
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
    return WormGraph(
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
    graph = WormGraphFromBinaryRequest(
        binary_image=binary_image,
        skeleton=skeleton,
        max_radius=None,
        max_skel_length=None,
    ).build()
    longest_coords = np.zeros((0, 2), dtype=int)
    longest_length = 0.0
    for path in graph.paths_between_lengths(min_length=0.0, max_length=max_length):
        coords = path.to_pixel_coords(graph)
        path_length = float(calculate_cumulative_lengths(coords)[-1])
        if path_length >= longest_length:
            longest_coords = coords
            longest_length = path_length
    return longest_coords


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


@dataclass(frozen=True, slots=True)
class WormClusterPathSelectionPolicy:
    """Shape and coverage policy for selecting candidate paths in a worm cluster."""

    median_worm_area: float | None
    component_area: int
    num_control_points: int
    mean_angles: np.ndarray
    inv_angles_covariance_matrix: np.ndarray
    cost_threshold: float
    overlap_weight: float
    leftover_weight: float
    min_path_length: float
    max_path_length: float

    def select(self, graph: WormGraph, paths: list[WormGraphPath]) -> list[np.ndarray]:
        paths_and_costs: list[tuple[WormGraphPath, float]] = []
        for path in paths:
            coords = path.to_pixel_coords(graph)
            total_length = float(calculate_cumulative_lengths(coords)[-1])
            if total_length > self.max_path_length or total_length < self.min_path_length:
                continue
            cost = WormShapeCostRequest(
                path_coords=coords,
                total_length=total_length,
                num_control_points=self.num_control_points,
                mean_angles=self.mean_angles,
                inv_angles_covariance_matrix=self.inv_angles_covariance_matrix,
            ).cost
            if cost < self.cost_threshold:
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
            path = paths_and_costs[int(ordered_index)][0]
            path_segment_matrix[list(path.segments), column] = True
        selected_indexes = WormPathSubsetSelectionContext(
            costs=costs,
            path_segment_matrix=path_segment_matrix,
            segment_lengths=graph.segment_lengths,
            overlap_weight=self.overlap_weight,
            leftover_weight=self.leftover_weight,
            max_worms=_cluster_max_worms(
                self.component_area,
                median_worm_area=self.median_worm_area,
            ),
        ).select()
        selected_paths = [
            paths_and_costs[int(order[selected_index])][0]
            for selected_index in selected_indexes
        ]
        return [path.to_pixel_coords(graph) for path in selected_paths]


def _cluster_max_worms(
    component_area: int,
    *,
    median_worm_area: float | None,
) -> int:
    if median_worm_area is None or median_worm_area <= 0:
        return 1
    return max(1, int(np.ceil(component_area / median_worm_area)))


@dataclass(frozen=True, slots=True)
class WormPathSelectionState:
    """Mutable-search state returned immutably between path selection levels."""

    best_subset: list[int]
    best_cost: float
    path_segment_matrix: np.ndarray
    path_choices: np.ndarray


@dataclass(frozen=True, slots=True)
class WormPathSubsetSelectionContext:
    """CP path coverage objective for selecting non-overlapping worm paths."""

    costs: np.ndarray
    path_segment_matrix: np.ndarray
    segment_lengths: np.ndarray
    overlap_weight: float
    leftover_weight: float
    max_worms: int

    def select(self) -> list[int]:
        state = WormPathSelectionState(
            best_subset=[],
            best_cost=float(np.sum(self.segment_lengths) * self.leftover_weight),
            path_segment_matrix=self.path_segment_matrix.astype(int),
            path_choices=np.eye(len(self.costs), dtype=bool),
        )
        for _level in range(min(self.max_worms, len(self.costs))):
            state = self._select_one_level(state)
            if np.prod(state.path_choices.shape) == 0:
                break
        return state.best_subset

    def _select_one_level(self, state: WormPathSelectionState) -> WormPathSelectionState:
        partial_costs = (
            np.sum(self.costs[:, np.newaxis] * state.path_choices, axis=0)
            + np.sum(
                np.maximum(state.path_segment_matrix - 1, 0)
                * self.segment_lengths[:, np.newaxis],
                axis=0,
            )
            * self.overlap_weight
        )
        total_costs = (
            partial_costs
            + np.sum(
                (state.path_segment_matrix == 0) * self.segment_lengths[:, np.newaxis],
                axis=0,
            )
            * self.leftover_weight
        )
        order = np.lexsort([total_costs])
        best_subset = state.best_subset
        best_cost = state.best_cost
        if len(order) and total_costs[order[0]] < best_cost:
            best_subset = np.flatnonzero(state.path_choices[:, order[0]]).tolist()
            best_cost = float(total_costs[order[0]])
        mask = partial_costs < best_cost
        if not np.any(mask):
            return self._empty_state(best_subset, best_cost)
        order = order[mask[order]]
        if len(order) * len(self.costs) > 5000:
            order = order[: (1 + 5000 // len(self.costs))]
        path_segment_matrix = state.path_segment_matrix[:, order]
        path_choices = state.path_choices[:, order]
        i, j = np.mgrid[0 : len(self.costs), 0 : len(self.costs)]
        disallow = i >= j
        allowed = np.dot(disallow, path_choices) == 0
        if not np.any(allowed):
            return self._empty_state(best_subset, best_cost)
        i, j = np.argwhere(allowed).transpose()
        return WormPathSelectionState(
            best_subset=best_subset,
            best_cost=best_cost,
            path_segment_matrix=(
                self.path_segment_matrix[:, i] + path_segment_matrix[:, j]
            ),
            path_choices=np.eye(len(self.costs), dtype=bool)[:, i] | path_choices[:, j],
        )

    def _empty_state(
        self,
        best_subset: list[int],
        best_cost: float,
    ) -> WormPathSelectionState:
        return WormPathSelectionState(
            best_subset=best_subset,
            best_cost=best_cost,
            path_segment_matrix=np.zeros((len(self.costs), 0), dtype=int),
            path_choices=np.zeros((len(self.costs), 0), dtype=bool),
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


@dataclass(frozen=True, slots=True)
class WormShapeCostRequest:
    """Mahalanobis-style CP worm shape cost for one candidate path."""

    path_coords: np.ndarray
    total_length: float
    num_control_points: int
    mean_angles: np.ndarray
    inv_angles_covariance_matrix: np.ndarray

    @property
    def cost(self) -> float:
        control_coords = sample_control_points(
            self.path_coords,
            calculate_cumulative_lengths(self.path_coords),
            self.num_control_points,
        )
        if len(self.mean_angles) != self.num_control_points - 1:
            return 0.0
        expected_shape = (self.num_control_points - 1, self.num_control_points - 1)
        if self.inv_angles_covariance_matrix.shape != expected_shape:
            return 0.0
        angles = WormControlPointGeometry(control_coords).angles
        feature_vector = np.hstack((angles, [self.total_length])) - self.mean_angles
        return float(
            feature_vector
            @ self.inv_angles_covariance_matrix
            @ feature_vector
        )

    def passes(self, cost_threshold: float) -> bool:
        return self.cost < cost_threshold


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
    object_names = WormLabelOutputStrategy.for_overlap_style(
        overlap_style
    ).measurement_object_names(
        overlapping_object_name=overlapping_object_name,
        nonoverlapping_object_name=nonoverlapping_object_name,
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
        angles = WormControlPointGeometry(control_coords).angles
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
