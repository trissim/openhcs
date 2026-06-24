"""
Converted from CellProfiler: UntangleWorms
Original: UntangleWorms module for untangling overlapping worms

This module untangles overlapping worms using a trained worm model.
It takes a binary image and labels the worms, untangling them and
associating all of a worm's pieces together.
"""

import numpy as np
import re
import scipy.ndimage
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from heapq import heappop, heappush
from typing import Any, ClassVar

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
)
from openhcs.core.runtime_semantics import ObjectLabelRepresentation
from openhcs.core.runtime_values import (
    ObjectLabelPayload,
    ObjectLabelSet,
    ObjectLabelValue,
    SourceImageObjectLabelBuildRequest,
    SparseIJVLabelRows,
    object_label_dense_array,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.processing.materialization import csv_materializer, segmentation_mask_rois

from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.processing.backends.cellprofiler.worm_geometry import (
    branchpoints,
    calculate_cumulative_lengths,
    control_points_for_label_image,
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


class FlipMode(Enum):
    """StraightenWorms head/tail alignment policy."""

    NONE = "do_not_align"
    TOP = "top_brightest"
    BOTTOM = "bottom_brightest"
    MANUAL = "flip_manually"


@dataclass(frozen=True, slots=True)
class WormMeasurement:
    """StraightenWorms per-object intensity row."""

    slice_index: int
    object_number: int
    center_x: float
    center_y: float
    mean_intensity: float
    std_intensity: float


@dataclass(frozen=True, slots=True)
class DeadWormStats:
    """IdentifyDeadWorms summary row."""

    slice_index: int
    object_count: int
    mean_center_x: float
    mean_center_y: float
    mean_angle: float


@dataclass(frozen=True, slots=True)
class StraightenedWormPlacement:
    """Source-to-output mapping for one straightened worm block."""

    object_number: int
    output_y: slice
    output_x: slice
    source_y: np.ndarray
    source_x: np.ndarray


@dataclass(frozen=True, slots=True)
class StraightenWormControlPoints:
    """Control-point normalization policy for StraightenWorms."""

    points: np.ndarray | None
    labels: np.ndarray
    num_control_points: int

    @property
    def normalized(self) -> np.ndarray:
        if self.points is None:
            return control_points_for_label_image(self.labels, self.num_control_points)
        points = np.asarray(self.points, dtype=float)
        if points.ndim != 3:
            raise ValueError(
                "StraightenWorms control_points must have shape "
                "(objects, 2, control_points) or (2, control_points, objects)."
            )
        if points.shape[1] == 2:
            normalized = points
        elif points.shape[0] == 2:
            normalized = points.transpose(2, 0, 1)
        else:
            raise ValueError(
                "StraightenWorms control_points must include one coordinate axis "
                "of length 2."
            )
        if normalized.shape[2] != self.num_control_points:
            raise ValueError(
                f"StraightenWorms expected {self.num_control_points} control points; "
                f"got {normalized.shape[2]}."
            )
        return normalized


@dataclass(frozen=True, slots=True)
class StraightenWormsSliceRequest:
    """Executable StraightenWorms request for one 2-D runtime slice."""

    image: np.ndarray
    labels: np.ndarray
    control_points: np.ndarray
    worm_width: int
    num_control_points: int
    flip_mode: FlipMode
    measure_intensity: bool
    slice_index: int

    @property
    def half_width(self) -> int:
        return self.worm_width // 2

    @property
    def output_width(self) -> int:
        return 2 * self.half_width + 1

    @property
    def positive_labels(self) -> np.ndarray:
        labels = np.unique(self.labels)
        return labels[labels > 0]

    def execute(self) -> tuple[np.ndarray, np.ndarray, list[WormMeasurement]]:
        image = self.image
        labels = self.labels
        unique_labels = self.positive_labels
        if len(unique_labels) == 0:
            shape = (self.output_width, self.output_width)
            return np.zeros(shape, dtype=image.dtype), np.zeros(shape, dtype=np.int32), []

        lengths = self.worm_lengths(len(unique_labels))
        if not lengths:
            shape = (self.output_width, self.output_width)
            return np.zeros(shape, dtype=image.dtype), np.zeros(shape, dtype=np.int32), []

        shape = (max(lengths) + self.output_width, len(unique_labels) * self.output_width)
        straightened_image = np.zeros(shape, dtype=image.dtype)
        straightened_labels = np.zeros(shape, dtype=np.int32)
        placements = self.placements(unique_labels, lengths)
        self.apply_placements(straightened_image, straightened_labels, placements)
        return (
            straightened_image,
            straightened_labels,
            self.measurements(straightened_image, straightened_labels, placements),
        )

    def worm_lengths(self, worm_count: int) -> list[int]:
        lengths: list[int] = []
        for index in range(min(worm_count, self.control_points.shape[0])):
            control_point = self.control_points[index]
            lengths.append(int(np.ceil(calculate_cumulative_lengths(control_point.T)[-1])))
        return lengths

    def placements(
        self,
        unique_labels: np.ndarray,
        lengths: list[int],
    ) -> list[StraightenedWormPlacement]:
        placements: list[StraightenedWormPlacement] = []
        for index, object_number in enumerate(unique_labels):
            if index >= len(lengths) or lengths[index] == 0:
                continue
            if index >= self.control_points.shape[0]:
                continue
            placements.append(
                self.placement_for_object(
                    object_number=int(object_number),
                    object_index=index,
                    length=lengths[index],
                )
            )
        return placements

    def placement_for_object(
        self,
        *,
        object_number: int,
        object_index: int,
        length: int,
    ) -> StraightenedWormPlacement:
        control_point = self.control_points[object_index]
        ii = control_point[0]
        jj = control_point[1]
        t_orig = np.linspace(0, length, self.num_control_points)
        t_new = np.arange(0, length + 1)
        ci = np.interp(t_new, t_orig, ii)
        cj = np.interp(t_new, t_orig, jj)

        di = np.diff(ci, prepend=ci[0])
        dj = np.diff(cj, prepend=cj[0])
        di[0] = di[1] if len(di) > 1 else 0
        dj[0] = dj[1] if len(dj) > 1 else 0
        norm = np.sqrt(di**2 + dj**2)
        norm[norm == 0] = 1
        ni = -dj / norm
        nj = di / norm

        half_width = self.half_width
        ci_ext = np.concatenate(
            [
                np.arange(-half_width, 0) * nj[0] + ci[0],
                ci,
                np.arange(1, half_width + 1) * nj[-1] + ci[-1],
            ]
        )
        cj_ext = np.concatenate(
            [
                np.arange(-half_width, 0) * (-ni[0]) + cj[0],
                cj,
                np.arange(1, half_width + 1) * (-ni[-1]) + cj[-1],
            ]
        )
        ni_ext = np.concatenate([[ni[0]] * half_width, ni, [ni[-1]] * half_width])
        nj_ext = np.concatenate([[nj[0]] * half_width, nj, [nj[-1]] * half_width])
        iii, jjj = np.mgrid[0 : len(ci_ext), -half_width : (half_width + 1)]
        source_y = ci_ext[iii] + ni_ext[iii] * jjj
        source_x = cj_ext[iii] + nj_ext[iii] * jjj
        if self.should_flip(object_number, ci_ext, cj_ext, ni_ext, nj_ext, iii, jjj):
            iii_flip = len(ci_ext) - iii - 1
            jjj_flip = -jjj
            source_y = ci_ext[iii_flip] + ni_ext[iii_flip] * jjj_flip
            source_x = cj_ext[iii_flip] + nj_ext[iii_flip] * jjj_flip
        return StraightenedWormPlacement(
            object_number=object_number,
            output_y=slice(0, len(ci_ext)),
            output_x=slice(
                self.output_width * object_index,
                self.output_width * (object_index + 1),
            ),
            source_y=np.ascontiguousarray(source_y, dtype=float),
            source_x=np.ascontiguousarray(source_x, dtype=float),
        )

    def should_flip(
        self,
        object_number: int,
        ci_ext: np.ndarray,
        cj_ext: np.ndarray,
        ni_ext: np.ndarray,
        nj_ext: np.ndarray,
        iii: np.ndarray,
        jjj: np.ndarray,
    ) -> bool:
        if self.flip_mode is FlipMode.NONE:
            return False
        source_y = ci_ext[iii] + ni_ext[iii] * jjj
        source_x = cj_ext[iii] + nj_ext[iii] * jjj
        sampled_image = scipy.ndimage.map_coordinates(
            self.image,
            [source_y, source_x],
            order=1,
            mode="constant",
        )
        sampled_mask = scipy.ndimage.map_coordinates(
            (self.labels == object_number).astype(np.float32),
            [source_y, source_x],
            order=0,
        )
        sampled_image = sampled_image * sampled_mask
        halfway = len(ci_ext) // 2
        area_top = np.sum(sampled_mask[:halfway, :])
        area_bottom = np.sum(sampled_mask[halfway:, :])
        if area_top <= 0 or area_bottom <= 0:
            return False
        top_intensity = np.sum(sampled_image[:halfway, :]) / area_top
        bottom_intensity = np.sum(sampled_image[halfway:, :]) / area_bottom
        return (
            self.flip_mode is FlipMode.TOP
            and top_intensity < bottom_intensity
        ) or (
            self.flip_mode is FlipMode.BOTTOM
            and bottom_intensity < top_intensity
        )

    def apply_placements(
        self,
        straightened_image: np.ndarray,
        straightened_labels: np.ndarray,
        placements: list[StraightenedWormPlacement],
    ) -> None:
        if not placements:
            return
        flat_source_y = np.concatenate([placement.source_y.ravel() for placement in placements])
        flat_source_x = np.concatenate([placement.source_x.ravel() for placement in placements])
        flat_image = scipy.ndimage.map_coordinates(
            self.image,
            [flat_source_y, flat_source_x],
            order=1,
            mode="constant",
        )
        flat_labels = scipy.ndimage.map_coordinates(
            self.labels,
            [flat_source_y, flat_source_x],
            order=0,
            mode="constant",
            cval=0,
        )
        offset = 0
        for placement in placements:
            block_shape = placement.source_y.shape
            block_size = placement.source_y.size
            next_offset = offset + block_size
            image_block = flat_image[offset:next_offset].reshape(block_shape)
            label_block = flat_labels[offset:next_offset].reshape(block_shape)
            straightened_image[placement.output_y, placement.output_x] = image_block
            output_label_block = straightened_labels[placement.output_y, placement.output_x]
            output_label_block[label_block == placement.object_number] = placement.object_number
            offset = next_offset

    def measurements(
        self,
        straightened_image: np.ndarray,
        straightened_labels: np.ndarray,
        placements: list[StraightenedWormPlacement],
    ) -> list[WormMeasurement]:
        if not self.measure_intensity:
            return []
        measurements: list[WormMeasurement] = []
        for placement in placements:
            mask = (
                straightened_labels[placement.output_y, placement.output_x]
                == placement.object_number
            )
            if np.sum(mask) == 0:
                continue
            image_block = straightened_image[placement.output_y, placement.output_x]
            values = image_block[mask]
            center_y, center_x = scipy.ndimage.center_of_mass(mask.astype(float))
            measurements.append(
                WormMeasurement(
                    slice_index=self.slice_index,
                    object_number=placement.object_number,
                    center_x=(
                        float(center_x) + float(placement.output_x.start)
                        if not np.isnan(center_x)
                        else 0.0
                    ),
                    center_y=(
                        float(center_y) + float(placement.output_y.start)
                        if not np.isnan(center_y)
                        else 0.0
                    ),
                    mean_intensity=float(np.mean(values)),
                    std_intensity=float(np.std(values)),
                )
            )
        return measurements


@dataclass(frozen=True, slots=True)
class DeadWormDiamondTemplate:
    """Diamond-shaped dead-worm structuring element at one angle."""

    worm_width: int
    worm_length: int
    angle: float

    def footprint(self) -> np.ndarray:
        from scipy.ndimage import binary_fill_holes

        x0 = int(np.sin(self.angle) * self.worm_length / 2)
        x1 = int(np.cos(self.angle) * self.worm_width / 2)
        x2 = -x0
        x3 = -x1
        y2 = int(np.cos(self.angle) * self.worm_length / 2)
        y1 = int(np.sin(self.angle) * self.worm_width / 2)
        y0 = -y2
        y3 = -y1
        xmax = np.max(np.abs([x0, x1, x2, x3]))
        ymax = np.max(np.abs([y0, y1, y2, y3]))
        footprint = np.zeros((ymax * 2 + 1, xmax * 2 + 1), bool)
        pts_y0 = np.array([y0, y1, y2, y3]) + ymax
        pts_x0 = np.array([x0, x1, x2, x3]) + xmax
        pts_y1 = np.array([y1, y2, y3, y0]) + ymax
        pts_x1 = np.array([x1, x2, x3, x0]) + xmax
        i_pts, j_pts = LineSegments.from_endpoints(
            pts_y0,
            pts_x0,
            pts_y1,
            pts_x1,
        ).points()
        valid = (
            (i_pts >= 0)
            & (i_pts < footprint.shape[0])
            & (j_pts >= 0)
            & (j_pts < footprint.shape[1])
        )
        footprint[i_pts[valid], j_pts[valid]] = True
        return binary_fill_holes(footprint)


@dataclass(frozen=True, slots=True)
class LineSegments:
    """Integer points along one or more line segments."""

    y0: np.ndarray
    x0: np.ndarray
    y1: np.ndarray
    x1: np.ndarray

    @classmethod
    def from_endpoints(
        cls,
        y0: np.ndarray,
        x0: np.ndarray,
        y1: np.ndarray,
        x1: np.ndarray,
    ) -> "LineSegments":
        return cls(y0=y0, x0=x0, y1=y1, x1=x1)

    def points(self) -> tuple[np.ndarray, np.ndarray]:
        all_i: list[int] = []
        all_j: list[int] = []
        for index in range(len(self.y0)):
            dy = abs(self.y1[index] - self.y0[index])
            dx = abs(self.x1[index] - self.x0[index])
            sy = 1 if self.y0[index] < self.y1[index] else -1
            sx = 1 if self.x0[index] < self.x1[index] else -1
            err = dx - dy
            cy = self.y0[index]
            cx = self.x0[index]
            while True:
                all_i.append(cy)
                all_j.append(cx)
                if cy == self.y1[index] and cx == self.x1[index]:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    cx += sx
                if e2 < dx:
                    err += dx
                    cy += sy
        return np.array(all_i), np.array(all_j)


@dataclass(frozen=True, slots=True)
class ConnectedComponentEdges:
    """Union-find connected components for an integer edge list."""

    first: np.ndarray
    second: np.ndarray

    def labels(self) -> np.ndarray:
        if len(self.first) == 0:
            return np.zeros(0, dtype=int)
        vertex_count = max(np.max(self.first), np.max(self.second)) + 1
        labels = np.arange(vertex_count)

        def find(vertex: int) -> int:
            root = vertex
            while labels[root] != root:
                root = labels[root]
            while labels[vertex] != root:
                next_vertex = labels[vertex]
                labels[vertex] = root
                vertex = next_vertex
            return int(root)

        def union(first: int, second: int) -> None:
            first_root = find(first)
            second_root = find(second)
            if first_root != second_root:
                labels[first_root] = second_root

        for first, second in zip(self.first, self.second):
            union(int(first), int(second))
        for index in range(vertex_count):
            labels[index] = find(index)
        unique_labels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        return np.array([label_map[label] for label in labels])


@dataclass(frozen=True, slots=True)
class DeadWormAdjacencyPolicy:
    """CP dead-worm hit grouping policy in spatial/angle space."""

    i: np.ndarray
    j: np.ndarray
    angle: np.ndarray
    space_dist: float
    angle_dist: float

    def edges(self) -> tuple[np.ndarray, np.ndarray]:
        if len(self.i) < 2:
            return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
        order = np.lexsort((self.angle, self.j, self.i))
        i_sorted = self.i[order]
        j_sorted = self.j[order]
        angle_sorted = self.angle[order]
        first: list[int] = []
        second: list[int] = []
        for idx1 in range(len(self.i)):
            for idx2 in range(idx1 + 1, len(self.i)):
                spatial_dist_sq = (
                    (i_sorted[idx1] - i_sorted[idx2]) ** 2
                    + (j_sorted[idx1] - j_sorted[idx2]) ** 2
                )
                if spatial_dist_sq > self.space_dist**2:
                    continue
                angle_diff = abs(angle_sorted[idx1] - angle_sorted[idx2])
                if angle_diff <= self.angle_dist or (np.pi - angle_diff) <= self.angle_dist:
                    first.append(order[idx1])
                    second.append(order[idx2])
        return np.array(first, dtype=int), np.array(second, dtype=int)


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
    ) -> tuple[ObjectLabelValue, ObjectLabelPayload]:
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
    ) -> tuple[ObjectLabelValue, ObjectLabelPayload]:
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
    ) -> tuple[ObjectLabelValue, ObjectLabelPayload]:
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
    ) -> tuple[ObjectLabelValue, ObjectLabelPayload]:
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
    ObjectLabelValue,
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


@numpy
@special_inputs("worm_labels")
@special_outputs(
    ("straightened_labels", None),
    (
        "worm_measurements",
        csv_materializer(
            fields=[
                "slice_index",
                "object_number",
                "center_x",
                "center_y",
                "mean_intensity",
                "std_intensity",
            ],
            analysis_type="worm_measurements",
        ),
    ),
)
def straighten_worms(
    image: np.ndarray,
    worm_labels: np.ndarray,
    control_points: np.ndarray | None = None,
    worm_width: int = 20,
    num_control_points: int = 21,
    flip_mode: FlipMode = FlipMode.NONE,
    number_of_segments: int = 4,
    number_of_stripes: int = 3,
    measure_intensity: bool = True,
) -> tuple[Any, ...]:
    """Straighten labeled worms using sampled or provided control points."""
    del number_of_segments, number_of_stripes
    flip_mode = coerce_cellprofiler_enum(FlipMode, flip_mode)
    if flip_mode is FlipMode.MANUAL:
        raise NotImplementedError("StraightenWorms manual flipping is interactive.")

    image_stack = image[np.newaxis, :, :] if image.ndim == 2 else image
    labels_stack = object_label_dense_array(worm_labels, dtype=np.int32)
    if labels_stack.ndim == 2:
        labels_stack = labels_stack[np.newaxis, :, :]

    straightened_images: list[np.ndarray] = []
    straightened_label_planes: list[np.ndarray] = []
    all_measurements: list[WormMeasurement] = []
    for slice_index in range(image_stack.shape[0]):
        labels_slice = (
            labels_stack[slice_index]
            if slice_index < labels_stack.shape[0]
            else labels_stack[0]
        )
        slice_image, slice_labels, measurements = StraightenWormsSliceRequest(
            image=image_stack[slice_index],
            labels=labels_slice,
            control_points=StraightenWormControlPoints(
                points=control_points,
                labels=labels_slice,
                num_control_points=num_control_points,
            ).normalized,
            worm_width=worm_width,
            num_control_points=num_control_points,
            flip_mode=flip_mode,
            measure_intensity=measure_intensity,
            slice_index=slice_index,
        ).execute()
        straightened_images.append(slice_image)
        straightened_label_planes.append(slice_labels)
        all_measurements.extend(measurements)

    straightened_image_stack = np.stack(straightened_images, axis=0)
    straightened_label_stack = np.stack(straightened_label_planes, axis=0)
    return (
        *tuple(
            straightened_image_stack[index]
            for index in range(straightened_image_stack.shape[0])
        ),
        straightened_label_stack,
        all_measurements,
    )


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(
    (
        "dead_worm_stats",
        csv_materializer(
            fields=[
                "slice_index",
                "object_count",
                "mean_center_x",
                "mean_center_y",
                "mean_angle",
            ],
            analysis_type="dead_worm_identification",
        ),
    ),
    ("labels", segmentation_mask_rois()),
)
def identify_dead_worms(
    image: np.ndarray,
    worm_width: int = 10,
    worm_length: int = 100,
    angle_count: int = 32,
    auto_distance: bool = True,
    space_distance: float = 5.0,
    angular_distance: float = 30.0,
) -> tuple[np.ndarray, DeadWormStats, np.ndarray]:
    """Identify straight dead worms by diamond-template matches across angles."""
    from scipy.ndimage import binary_erosion

    mask = image > 0
    i_coords: list[np.ndarray] = []
    j_coords: list[np.ndarray] = []
    a_coords: list[np.ndarray] = []
    ig, jg = np.mgrid[0 : mask.shape[0], 0 : mask.shape[1]]
    for angle_index in range(angle_count):
        angle = float(angle_index) * np.pi / float(angle_count)
        footprint = DeadWormDiamondTemplate(
            worm_width=worm_width,
            worm_length=worm_length,
            angle=angle,
        ).footprint()
        erosion = binary_erosion(mask, footprint)
        point_count = np.sum(erosion)
        if point_count <= 0:
            continue
        i_coords.append(ig[erosion])
        j_coords.append(jg[erosion])
        a_coords.append(np.ones(point_count) * angle)

    if not i_coords:
        labels = np.zeros(mask.shape, dtype=np.int32)
        return image, DeadWormStats(0, 0, 0.0, 0.0, 0.0), labels

    i = np.concatenate(i_coords)
    j = np.concatenate(j_coords)
    a = np.concatenate(a_coords)
    if auto_distance:
        space_dist = float(worm_width)
        angle_dist = np.arctan2(worm_width, worm_length) + np.pi / angle_count
    else:
        space_dist = space_distance
        angle_dist = angular_distance * np.pi / 180.0

    first, second = DeadWormAdjacencyPolicy(
        i=i,
        j=j,
        angle=a,
        space_dist=space_dist,
        angle_dist=angle_dist,
    ).edges()
    if len(first) > 0:
        ij_labels = ConnectedComponentEdges(first, second).labels() + 1
        label_count = int(np.max(ij_labels))
        label_indexes = np.arange(1, label_count + 1)
        center_x = np.array([np.mean(j[ij_labels == label]) for label in label_indexes])
        center_y = np.array([np.mean(i[ij_labels == label]) for label in label_indexes])
        angles = np.array([np.mean(a[ij_labels == label]) for label in label_indexes])
        labels = np.zeros(mask.shape, dtype=np.int32)
        labels[i, j] = ij_labels
    else:
        label_count = len(i)
        labels = np.zeros(mask.shape, dtype=np.int32)
        if label_count > 0:
            labels[i, j] = np.arange(1, label_count + 1)
            center_x = j.astype(float)
            center_y = i.astype(float)
            angles = a
        else:
            center_x = np.array([])
            center_y = np.array([])
            angles = np.array([])

    stats = DeadWormStats(
        slice_index=0,
        object_count=int(label_count),
        mean_center_x=float(np.mean(center_x)) if len(center_x) > 0 else 0.0,
        mean_center_y=float(np.mean(center_y)) if len(center_y) > 0 else 0.0,
        mean_angle=float(np.mean(angles) * 180 / np.pi) if len(angles) > 0 else 0.0,
    )
    return image, stats, labels


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
) -> tuple[ObjectLabelValue, ObjectLabelPayload]:
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
    sparse_overlapping = SourceImageObjectLabelBuildRequest(
        image=source_image,
        labels=SparseIJVLabelRows(ijv),
    ).label_set(
        name=overlapping_object_name,
        representation=ObjectLabelRepresentation.SPARSE_IJV,
    )
    overlapping_payload = SourceImageObjectLabelBuildRequest(
        image=source_image,
        labels=overlapping,
        declared_object_count=len(all_path_coords),
    ).payload()
    nonoverlapping_payload = SourceImageObjectLabelBuildRequest(
        image=source_image,
        labels=nonoverlapping,
        declared_object_count=len(all_path_coords),
    ).payload()
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
