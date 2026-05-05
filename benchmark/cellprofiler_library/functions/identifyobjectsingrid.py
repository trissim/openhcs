"""Converted from CellProfiler: IdentifyObjectsInGrid

Identifies objects within each section of a grid pattern.
This module creates labeled objects based on grid definitions,
with options for rectangles, circles, or natural shapes.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import ClassVar, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from metaclass_registry import AutoRegisterMeta
from numba import njit
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_outputs, special_inputs
from openhcs.core.runtime_semantics import SpatialGridOrdering
from openhcs.core.runtime_values import (
    ObjectLabelPayload,
    SpatialGrid,
    image_payload_metadata,
)
from openhcs.processing.materialization import csv_materializer, segmentation_mask_rois
from benchmark.cellprofiler_library.functions._enum import _coerce_function_enum


class ShapeChoice(Enum):
    RECTANGLE = "rectangle_forced_location"
    CIRCLE_FORCED = "circle_forced_location"
    CIRCLE_NATURAL = "circle_natural_location"
    NATURAL = "natural_shape_and_location"


class DiameterChoice(Enum):
    AUTOMATIC = "automatic"
    MANUAL = "manual"


@dataclass
class GridDefinition:
    """Grid parameters - typically from DefineGrid module output."""
    rows: int
    columns: int
    x_spacing: float
    y_spacing: float
    x_location_of_lowest_x_spot: float
    y_location_of_lowest_y_spot: float
    x_locations: np.ndarray  # Shape (rows, columns)
    y_locations: np.ndarray  # Shape (rows, columns)
    spot_table: np.ndarray   # Shape (rows, columns) with spot numbers
    image_height: int
    image_width: int
    ordering: SpatialGridOrdering = SpatialGridOrdering.BY_ROWS


@dataclass
class GridObjectStats:
    slice_index: int
    object_count: int
    grid_rows: int
    grid_columns: int
    shape_type: str


@dataclass(frozen=True, slots=True)
class GridShapeRequest:
    """Inputs needed to materialize one grid object shape strategy."""

    grid: GridDefinition
    guiding_labels: np.ndarray | None = None
    filtered_guides: np.ndarray | None = None
    diameter_choice: DiameterChoice = DiameterChoice.MANUAL
    circle_diameter: int = 20


def _fill_grid(grid: GridDefinition) -> np.ndarray:
    """Fill a labels matrix by labeling each rectangle in the grid."""
    i_min = int(grid.y_location_of_lowest_y_spot - grid.y_spacing / 2)
    j_min = int(grid.x_location_of_lowest_x_spot - grid.x_spacing / 2)
    return _fill_grid_numba(
        int(grid.image_height),
        int(grid.image_width),
        float(grid.y_spacing),
        float(grid.x_spacing),
        i_min,
        j_min,
        np.asarray(grid.spot_table, dtype=np.int32),
    )


@njit(cache=True)
def _fill_grid_numba(
    image_height: int,
    image_width: int,
    y_spacing: float,
    x_spacing: float,
    row_origin: int,
    col_origin: int,
    spot_table: np.ndarray,
) -> np.ndarray:
    """Fill rectangular grid labels using the same floor-bin boundaries."""
    labels = np.zeros((image_height, image_width), dtype=np.int32)
    grid_rows, grid_columns = spot_table.shape
    for grid_row in range(grid_rows):
        row_start = int(np.ceil(row_origin + grid_row * y_spacing))
        row_stop = int(np.ceil(row_origin + (grid_row + 1) * y_spacing))
        if row_start < 0:
            row_start = 0
        if row_stop > image_height:
            row_stop = image_height
        if row_start >= row_stop:
            continue
        for grid_col in range(grid_columns):
            col_start = int(np.ceil(col_origin + grid_col * x_spacing))
            col_stop = int(np.ceil(col_origin + (grid_col + 1) * x_spacing))
            if col_start < 0:
                col_start = 0
            if col_stop > image_width:
                col_stop = image_width
            if col_start >= col_stop:
                continue
            label_id = int(spot_table[grid_row, grid_col])
            for row in range(row_start, row_stop):
                for col in range(col_start, col_stop):
                    labels[row, col] = label_id
    return labels


def _grid_labels_for_shape(
    grid: GridDefinition,
    shape: tuple[int, int],
) -> np.ndarray:
    """Return grid labels aligned to the requested output shape."""
    labels = _fill_grid(grid)
    if labels.shape == shape:
        return labels
    result = np.zeros(
        [max(labels.shape[i], shape[i]) for i in range(2)],
        dtype=np.int32,
    )
    result[0:labels.shape[0], 0:labels.shape[1]] = labels
    return result


def _grid_definition(
    *,
    image_shape: tuple[int, int],
    grid: SpatialGrid | None,
    grid_rows: int,
    grid_columns: int,
    x_spacing: float,
    y_spacing: float,
    x_origin: float,
    y_origin: float,
    ordering: SpatialGridOrdering = SpatialGridOrdering.BY_ROWS,
) -> GridDefinition:
    """Build executable grid geometry from a runtime grid or direct kwargs."""
    height, width = image_shape
    if grid is not None:
        grid_rows = grid.rows
        grid_columns = grid.columns
        x_spacing = grid.x_spacing
        y_spacing = grid.y_spacing
        x_origin = grid.x_origin
        y_origin = grid.y_origin
        ordering = grid.ordering
    ordering = _coerce_function_enum(SpatialGridOrdering, ordering)

    i_grid, j_grid = np.mgrid[0:grid_rows, 0:grid_columns]
    y_locations = y_origin + i_grid * y_spacing
    x_locations = x_origin + j_grid * x_spacing
    spot_table = _grid_spot_table(grid_rows, grid_columns, ordering)
    return GridDefinition(
        rows=grid_rows,
        columns=grid_columns,
        x_spacing=x_spacing,
        y_spacing=y_spacing,
        x_location_of_lowest_x_spot=x_origin,
        y_location_of_lowest_y_spot=y_origin,
        x_locations=x_locations,
        y_locations=y_locations,
        spot_table=spot_table,
        image_height=height,
        image_width=width,
        ordering=ordering,
    )


def _grid_spot_table(
    rows: int,
    columns: int,
    ordering: SpatialGridOrdering,
) -> np.ndarray:
    """Return object IDs arranged by the grid's declared numbering order."""
    object_ids = np.arange(1, rows * columns + 1)
    if ordering is SpatialGridOrdering.BY_ROWS:
        return object_ids.reshape(columns, rows).T
    return object_ids.reshape(rows, columns)


def _centers_of_labels(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate centers of mass for each label."""
    max_label = int(labels.max())
    if max_label == 0:
        return np.array([]), np.array([])
    centers_i, centers_j = _centers_of_labels_numba(
        np.asarray(labels, dtype=np.int32),
        max_label,
    )
    return centers_i, centers_j


@njit(cache=True)
def _centers_of_labels_numba(
    labels: np.ndarray,
    max_label: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return row/column centers for dense positive label IDs in one pass."""
    sums_i = np.zeros(max_label + 1, dtype=np.float64)
    sums_j = np.zeros(max_label + 1, dtype=np.float64)
    counts = np.zeros(max_label + 1, dtype=np.int64)
    height, width = labels.shape
    for row in range(height):
        for col in range(width):
            label_id = int(labels[row, col])
            if label_id > 0 and label_id <= max_label:
                sums_i[label_id] += row
                sums_j[label_id] += col
                counts[label_id] += 1

    centers_i = np.empty(max_label, dtype=np.float64)
    centers_j = np.empty(max_label, dtype=np.float64)
    for label_id in range(1, max_label + 1):
        count = counts[label_id]
        if count == 0:
            centers_i[label_id - 1] = np.nan
            centers_j[label_id - 1] = np.nan
        else:
            centers_i[label_id - 1] = sums_i[label_id] / count
            centers_j[label_id - 1] = sums_j[label_id] / count
    return centers_i, centers_j


def _run_circle(
    grid: GridDefinition,
    spot_center_i: np.ndarray,
    spot_center_j: np.ndarray,
    radius: float,
    guiding_labels: Optional[np.ndarray] = None
) -> np.ndarray:
    """Return a labels matrix composed of circles centered on given locations."""
    labels = _fill_grid(grid)

    # Fit labels to guiding objects size if needed
    if guiding_labels is not None:
        if any(guiding_labels.shape[i] > labels.shape[i] for i in range(2)):
            result = np.zeros(
                [max(guiding_labels.shape[i], labels.shape[i]) for i in range(2)],
                dtype=np.int32
            )
            result[0:labels.shape[0], 0:labels.shape[1]] = labels
            labels = result

    center_i_by_label, center_j_by_label = _spot_center_lookup_numba(
        np.asarray(grid.spot_table, dtype=np.int32),
        np.asarray(spot_center_i, dtype=np.float64),
        np.asarray(spot_center_j, dtype=np.float64),
        int(grid.spot_table.max()),
    )
    return _apply_circle_mask_numba(
        np.asarray(labels, dtype=np.int32),
        center_i_by_label,
        center_j_by_label,
        float(radius),
    )


@njit(cache=True)
def _spot_center_lookup_numba(
    spot_table: np.ndarray,
    spot_center_i: np.ndarray,
    spot_center_j: np.ndarray,
    max_label: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return dense label-to-center lookup arrays for grid spot IDs."""
    center_i_by_label = np.empty(max_label + 1, dtype=np.float64)
    center_j_by_label = np.empty(max_label + 1, dtype=np.float64)
    for label_id in range(max_label + 1):
        center_i_by_label[label_id] = np.nan
        center_j_by_label[label_id] = np.nan

    rows, columns = spot_table.shape
    for row in range(rows):
        for col in range(columns):
            label_id = int(spot_table[row, col])
            if label_id <= 0 or label_id > max_label:
                continue
            center_i_by_label[label_id] = float(spot_center_i[row, col])
            center_j_by_label[label_id] = float(spot_center_j[row, col])
    return center_i_by_label, center_j_by_label


@njit(cache=True)
def _apply_circle_mask_numba(
    labels: np.ndarray,
    center_i_by_label: np.ndarray,
    center_j_by_label: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Apply CP's inclusive circle mask without allocating per-pixel centers."""
    radius2 = (radius + 0.5) * (radius + 0.5)
    height, width = labels.shape
    max_label = len(center_i_by_label) - 1
    for row in range(height):
        for col in range(width):
            label_id = int(labels[row, col])
            if label_id <= 0 or label_id > max_label:
                labels[row, col] = 0
                continue
            center_i = center_i_by_label[label_id]
            center_j = center_j_by_label[label_id]
            if np.isnan(center_i) or np.isnan(center_j):
                labels[row, col] = 0
                continue
            delta_i = row - center_i
            delta_j = col - center_j
            if delta_i * delta_i + delta_j * delta_j > radius2:
                labels[row, col] = 0
    return labels


def _run_forced_circle(
    grid: GridDefinition,
    radius: float
) -> np.ndarray:
    """Return a labels matrix composed of circles centered in grid cells."""
    i, j = np.mgrid[0:grid.rows, 0:grid.columns]
    return _run_circle(
        grid,
        grid.y_locations[i, j] if grid.y_locations.ndim == 2 else grid.y_locations[i],
        grid.x_locations[i, j] if grid.x_locations.ndim == 2 else grid.x_locations[j],
        radius
    )


def _filter_labels_by_grid(
    guide_labels: np.ndarray,
    grid: GridDefinition
) -> np.ndarray:
    """Filter guide labels by proximity to edges of grid."""
    labels = _fill_grid(grid)
    lcenters = _guide_label_center_grid_ids(guide_labels, grid, grid_labels=labels)
    return _filter_labels_by_grid_numba(
        np.asarray(guide_labels, dtype=np.int32),
        np.asarray(labels, dtype=np.int32),
        lcenters,
    )


@njit(cache=True)
def _filter_labels_by_grid_numba(
    guide_labels: np.ndarray,
    grid_labels: np.ndarray,
    label_center_grid_ids: np.ndarray,
) -> np.ndarray:
    """Filter guide labels using center-grid membership without mask arrays."""
    filtered = guide_labels.copy()
    guide_height, guide_width = guide_labels.shape
    grid_height, grid_width = grid_labels.shape
    for row in range(guide_height):
        for col in range(guide_width):
            guide_id = int(guide_labels[row, col])
            remove = guide_id == 0
            center_grid_id = 0
            if (
                not remove
                and guide_id >= 0
                and guide_id < len(label_center_grid_ids)
            ):
                center_grid_id = int(label_center_grid_ids[guide_id])
                remove = center_grid_id == 0
            if not remove and row < grid_height and col < grid_width:
                remove = center_grid_id != int(grid_labels[row, col])
            if remove:
                filtered[row, col] = 0
    return filtered


def _guide_label_center_grid_ids(
    guide_labels: np.ndarray,
    grid: GridDefinition,
    *,
    grid_labels: np.ndarray | None = None,
) -> np.ndarray:
    """Map each guide label ID to the grid object ID containing its center."""
    labels = _fill_grid(grid) if grid_labels is None else grid_labels
    return _guide_label_center_grid_ids_numba(
        np.asarray(guide_labels, dtype=np.int32),
        np.asarray(labels, dtype=np.int32),
        int(np.ceil(grid.y_spacing / 10)),
        int(np.ceil(grid.x_spacing / 10)),
    )


@njit(cache=True)
def _guide_label_center_grid_ids_numba(
    guide_labels: np.ndarray,
    grid_labels: np.ndarray,
    y_border: int,
    x_border: int,
) -> np.ndarray:
    """Map guide labels to center grid IDs, preserving CP-style border erasure."""
    max_guide = int(np.max(guide_labels))
    lcenters = np.zeros(max_guide + 1, dtype=np.int32)
    if max_guide == 0:
        return lcenters

    centers_i, centers_j = _centers_of_labels_numba(guide_labels, max_guide)
    height, width = grid_labels.shape
    for guide_id in range(1, max_guide + 1):
        center_i = centers_i[guide_id - 1]
        center_j = centers_j[guide_id - 1]
        if (
            np.isnan(center_i)
            or np.isnan(center_j)
            or center_i >= height
            or center_j >= width
        ):
            continue
        row = int(np.round(center_i))
        col = int(np.round(center_j))
        if row < 0:
            row = 0
        if col < 0:
            col = 0
        if row >= height:
            row = height - 1
        if col >= width:
            col = width - 1
        lcenters[guide_id] = _grid_label_after_border_erase(
            grid_labels,
            row,
            col,
            y_border,
            x_border,
        )
    return lcenters


@njit(cache=True)
def _grid_label_after_border_erase(
    labels: np.ndarray,
    row: int,
    col: int,
    y_border: int,
    x_border: int,
) -> int:
    """Return label value after CP's grid-boundary dead-zone mask."""
    label_id = int(labels[row, col])
    if label_id == 0:
        return 0
    height, width = labels.shape
    if y_border > 0 and height > y_border:
        if row >= y_border and int(labels[row - y_border, col]) != label_id:
            return 0
        if row + y_border < height and int(labels[row + y_border, col]) != label_id:
            return 0
    if x_border > 0 and width > x_border:
        if col >= x_border and int(labels[row, col - x_border]) != label_id:
            return 0
        if col + x_border < width and int(labels[row, col + x_border]) != label_id:
            return 0
    return label_id


def _natural_grid_labels_from_guides(
    guide_labels: np.ndarray,
    grid: GridDefinition,
) -> np.ndarray:
    """Combine accepted guide parts per grid compartment."""
    grid_labels = _grid_labels_for_shape(grid, guide_labels.shape)
    lcenters = _guide_label_center_grid_ids(
        guide_labels,
        grid,
        grid_labels=grid_labels,
    )
    sparse_labels = _natural_grid_labels_from_guides_numba(
        np.asarray(guide_labels, dtype=np.int32),
        np.asarray(grid_labels, dtype=np.int32),
        lcenters,
    )
    return sparse_labels


@njit(cache=True)
def _natural_grid_labels_from_guides_numba(
    guide_labels: np.ndarray,
    grid_labels: np.ndarray,
    label_center_grid_ids: np.ndarray,
) -> np.ndarray:
    """Project accepted guide parts inside their center grid compartment."""
    labels = np.zeros(grid_labels.shape, dtype=np.int32)
    guide_height, guide_width = guide_labels.shape
    for row in range(guide_height):
        for col in range(guide_width):
            guide_id = int(guide_labels[row, col])
            if (
                guide_id <= 0
                or guide_id >= len(label_center_grid_ids)
                or row >= grid_labels.shape[0]
                or col >= grid_labels.shape[1]
            ):
                continue
            projected_id = int(label_center_grid_ids[guide_id])
            if projected_id != 0 and int(grid_labels[row, col]) == projected_id:
                labels[row, col] = projected_id
    return labels


class GridShapeStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal strategy for materializing grid object labels."""

    __registry_key__ = "shape_choice"
    __skip_if_no_key__ = True
    shape_choice: ClassVar[str | None] = None
    requires_guides: ClassVar[bool] = False

    @classmethod
    def for_shape_choice(cls, shape_choice: ShapeChoice | str) -> "GridShapeStrategy":
        resolved = _coerce_function_enum(ShapeChoice, shape_choice)
        strategy_type = cls.__registry__.get(
            resolved.value,
            RectangleGridShapeStrategy,
        )
        return strategy_type()

    @abstractmethod
    def labels(self, request: GridShapeRequest) -> np.ndarray:
        """Return dense labels for one grid shape mode."""


class RectangleGridShapeStrategy(GridShapeStrategy):
    """Fill each grid rectangle with its object label."""

    shape_choice = ShapeChoice.RECTANGLE.value

    def labels(self, request: GridShapeRequest) -> np.ndarray:
        return _fill_grid(request.grid)


class ForcedCircleGridShapeStrategy(GridShapeStrategy):
    """Draw fixed-diameter circles at grid centers."""

    shape_choice = ShapeChoice.CIRCLE_FORCED.value

    def labels(self, request: GridShapeRequest) -> np.ndarray:
        return _run_forced_circle(request.grid, request.circle_diameter / 2.0)


class NaturalCircleGridShapeStrategy(GridShapeStrategy):
    """Draw automatic circles using accepted guide objects for centers/area."""

    shape_choice = ShapeChoice.CIRCLE_NATURAL.value
    requires_guides = True

    def labels(self, request: GridShapeRequest) -> np.ndarray:
        guiding_labels = _require_guiding_labels(request)
        filtered_guides = _require_filtered_guides(request)
        labels = _fill_grid(request.grid)
        labels[filtered_guides[0:labels.shape[0], 0:labels.shape[1]] == 0] = 0
        centers_i, centers_j = _centers_of_labels(labels)

        nmissing = np.max(request.grid.spot_table) - len(centers_i)
        if nmissing > 0:
            centers_i = np.hstack((centers_i, [np.nan] * nmissing))
            centers_j = np.hstack((centers_j, [np.nan] * nmissing))

        spot_centers_i = centers_i[request.grid.spot_table - 1]
        spot_centers_j = centers_j[request.grid.spot_table - 1]

        return _run_circle(
            request.grid,
            spot_centers_i,
            spot_centers_j,
            _circle_radius(request, filtered_guides),
            guiding_labels,
        )


class NaturalGridShapeStrategy(GridShapeStrategy):
    """Preserve accepted guide-object shapes and relabel by center grid cell."""

    shape_choice = ShapeChoice.NATURAL.value
    requires_guides = True

    def labels(self, request: GridShapeRequest) -> np.ndarray:
        return _natural_grid_labels_from_guides(
            _require_guiding_labels(request),
            request.grid,
        )


def _grid_shape_labels(
    shape_choice: ShapeChoice,
    request: GridShapeRequest,
) -> np.ndarray:
    """Materialize labels, falling back to rectangles when guides are absent."""
    strategy = GridShapeStrategy.for_shape_choice(shape_choice)
    if strategy.requires_guides and request.guiding_labels is None:
        strategy = GridShapeStrategy.for_shape_choice(ShapeChoice.RECTANGLE)
    return strategy.labels(request)


def _circle_radius(
    request: GridShapeRequest,
    filtered_guides: np.ndarray,
) -> float:
    """Return manual or area-derived circle radius for grid object modes."""
    if request.diameter_choice is DiameterChoice.MANUAL:
        return request.circle_diameter / 2.0
    areas = np.bincount(filtered_guides[filtered_guides != 0].flatten())
    if len(areas) > 0 and np.any(areas != 0):
        median_area = np.median(areas[areas != 0])
        return max(1, np.sqrt(median_area / np.pi))
    return request.circle_diameter / 2.0


def _require_guiding_labels(request: GridShapeRequest) -> np.ndarray:
    if request.guiding_labels is None:
        raise ValueError("Grid shape strategy requires guiding labels.")
    return request.guiding_labels


def _require_filtered_guides(request: GridShapeRequest) -> np.ndarray:
    if request.filtered_guides is None:
        raise ValueError("Grid shape strategy requires filtered guiding labels.")
    return request.filtered_guides


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("grid")
@special_outputs(
    ("grid_stats", csv_materializer(
        fields=["slice_index", "object_count", "grid_rows", "grid_columns", "shape_type"],
        analysis_type="grid_objects"
    )),
    ("labels", segmentation_mask_rois())
)
def identify_objects_in_grid(
    image: np.ndarray,
    grid: SpatialGrid | None = None,
    grid_rows: int = 8,
    grid_columns: int = 12,
    x_spacing: float = 100.0,
    y_spacing: float = 100.0,
    x_origin: float = 50.0,
    y_origin: float = 50.0,
    shape_choice: ShapeChoice = ShapeChoice.RECTANGLE,
    diameter_choice: DiameterChoice = DiameterChoice.MANUAL,
    circle_diameter: int = 20,
) -> Tuple[np.ndarray, GridObjectStats, ObjectLabelPayload]:
    """
    Identify objects within each section of a grid pattern.
    
    This function creates labeled objects based on grid definitions.
    Objects are numbered according to grid position.
    
    Args:
        image: Input image (H, W)
        grid_rows: Number of rows in the grid
        grid_columns: Number of columns in the grid
        x_spacing: Horizontal spacing between grid centers in pixels
        y_spacing: Vertical spacing between grid centers in pixels
        x_origin: X coordinate of the lowest X spot
        y_origin: Y coordinate of the lowest Y spot
        shape_choice: Shape of objects (rectangle, circle_forced, etc.)
        diameter_choice: How to determine circle diameter
        circle_diameter: Manual circle diameter in pixels
    
    Returns:
        Tuple of (image, stats, labels)
    """
    shape_choice = _coerce_function_enum(ShapeChoice, shape_choice)
    diameter_choice = _coerce_function_enum(DiameterChoice, diameter_choice)

    grid_definition = _grid_definition(
        image_shape=image.shape,
        grid=grid,
        grid_rows=grid_rows,
        grid_columns=grid_columns,
        x_spacing=x_spacing,
        y_spacing=y_spacing,
        x_origin=x_origin,
        y_origin=y_origin,
    )
    labels = _grid_shape_labels(
        shape_choice,
        GridShapeRequest(
            grid=grid_definition,
            diameter_choice=diameter_choice,
            circle_diameter=circle_diameter,
        ),
    )
    
    object_count = grid_definition.rows * grid_definition.columns
    
    stats = GridObjectStats(
        slice_index=0,
        object_count=object_count,
        grid_rows=grid_definition.rows,
        grid_columns=grid_definition.columns,
        shape_type=shape_choice.value
    )
    
    return image, stats, ObjectLabelPayload(
        labels=labels.astype(np.int32, copy=False),
        declared_object_count=object_count,
        spatial_origin_yx=image_payload_metadata(image).spatial_origin_yx,
        source_spatial_shape_yx=image_payload_metadata(image).source_spatial_shape_yx,
    )


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("grid", "guiding_labels")
@special_outputs(
    ("grid_stats", csv_materializer(
        fields=["slice_index", "object_count", "grid_rows", "grid_columns", "shape_type"],
        analysis_type="grid_objects"
    )),
    ("labels", segmentation_mask_rois())
)
def identify_objects_in_grid_with_guides(
    image: np.ndarray,
    guiding_labels: np.ndarray,
    grid: SpatialGrid | None = None,
    grid_rows: int = 8,
    grid_columns: int = 12,
    x_spacing: float = 100.0,
    y_spacing: float = 100.0,
    x_origin: float = 50.0,
    y_origin: float = 50.0,
    shape_choice: ShapeChoice = ShapeChoice.CIRCLE_NATURAL,
    diameter_choice: DiameterChoice = DiameterChoice.AUTOMATIC,
    circle_diameter: int = 20,
) -> Tuple[np.ndarray, GridObjectStats, ObjectLabelPayload]:
    """
    Identify objects in grid using guiding objects for shape/location.
    
    This variant uses previously identified objects to guide the
    shape and/or location of grid objects.
    
    Args:
        image: Input image (H, W)
        guiding_labels: Previously identified objects for guidance
        grid_rows: Number of rows in the grid
        grid_columns: Number of columns in the grid
        x_spacing: Horizontal spacing between grid centers
        y_spacing: Vertical spacing between grid centers
        x_origin: X coordinate of the lowest X spot
        y_origin: Y coordinate of the lowest Y spot
        shape_choice: Shape of objects
        diameter_choice: How to determine circle diameter
        circle_diameter: Manual circle diameter in pixels
    
    Returns:
        Tuple of (image, stats, labels)
    """
    shape_choice = _coerce_function_enum(ShapeChoice, shape_choice)
    diameter_choice = _coerce_function_enum(DiameterChoice, diameter_choice)

    grid_definition = _grid_definition(
        image_shape=image.shape,
        grid=grid,
        grid_rows=grid_rows,
        grid_columns=grid_columns,
        x_spacing=x_spacing,
        y_spacing=y_spacing,
        x_origin=x_origin,
        y_origin=y_origin,
    )
    
    # Filter guiding labels
    filtered_guides = _filter_labels_by_grid(guiding_labels, grid_definition)
    labels = _grid_shape_labels(
        shape_choice,
        GridShapeRequest(
            grid=grid_definition,
            guiding_labels=guiding_labels,
            filtered_guides=filtered_guides,
            diameter_choice=diameter_choice,
            circle_diameter=circle_diameter,
        ),
    )
    
    object_count = grid_definition.rows * grid_definition.columns
    
    stats = GridObjectStats(
        slice_index=0,
        object_count=object_count,
        grid_rows=grid_definition.rows,
        grid_columns=grid_definition.columns,
        shape_type=shape_choice.value
    )
    
    return image, stats, ObjectLabelPayload(
        labels=labels.astype(np.int32, copy=False),
        declared_object_count=object_count,
        spatial_origin_yx=image_payload_metadata(image).spatial_origin_yx,
        source_spatial_shape_yx=image_payload_metadata(image).source_spatial_shape_yx,
    )


def _prepare_identify_objects_in_grid() -> None:
    """Compile grid-label kernels before timed execution."""
    image = np.zeros((64, 64), dtype=np.float32)
    grid = SpatialGrid(
        name="Grid",
        rows=4,
        columns=4,
        x_spacing=16.0,
        y_spacing=16.0,
        x_origin=8.0,
        y_origin=8.0,
    )
    guide_labels = np.zeros((64, 64), dtype=np.int32)
    guide_labels[8:18, 8:18] = 1
    guide_labels[24:34, 24:34] = 2
    identify_objects_in_grid.__wrapped__(
        image,
        grid=grid,
        shape_choice=ShapeChoice.RECTANGLE,
    )
    identify_objects_in_grid_with_guides.__wrapped__(
        image,
        guide_labels,
        grid=grid,
        shape_choice=ShapeChoice.NATURAL,
    )


identify_objects_in_grid.__openhcs_prepare__ = _prepare_identify_objects_in_grid
identify_objects_in_grid_with_guides.__openhcs_prepare__ = (
    _prepare_identify_objects_in_grid
)
