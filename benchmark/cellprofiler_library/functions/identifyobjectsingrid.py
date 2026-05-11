"""Converted from CellProfiler: IdentifyObjectsInGrid

Identifies objects within each section of a grid pattern.
This module creates labeled objects based on grid definitions,
with options for rectangles, circles, or natural shapes.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import ClassVar, Tuple
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
    object_label_payload_from_source_image,
)
from openhcs.processing.materialization import csv_materializer, segmentation_mask_rois
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum


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

    @classmethod
    def from_runtime(
        cls,
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
    ) -> "GridDefinition":
        """Build executable grid geometry from a runtime grid or direct kwargs."""
        spatial_grid = (
            grid
            if grid is not None
            else SpatialGrid(
                name="grid",
                rows=grid_rows,
                columns=grid_columns,
                x_spacing=x_spacing,
                y_spacing=y_spacing,
                x_origin=x_origin,
                y_origin=y_origin,
                ordering=coerce_cellprofiler_enum(SpatialGridOrdering, ordering),
                source_spatial_shape_yx=tuple(int(value) for value in image_shape),
            )
        )
        height, width = (
            spatial_grid.source_spatial_shape_yx
            if spatial_grid.source_spatial_shape_yx is not None
            else image_shape
        )
        return cls(
            rows=spatial_grid.rows,
            columns=spatial_grid.columns,
            x_spacing=spatial_grid.x_spacing,
            y_spacing=spatial_grid.y_spacing,
            x_location_of_lowest_x_spot=spatial_grid.x_origin,
            y_location_of_lowest_y_spot=spatial_grid.y_origin,
            x_locations=spatial_grid.x_locations_array(),
            y_locations=spatial_grid.y_locations_array(),
            spot_table=spatial_grid.spot_table_array(),
            image_height=height,
            image_width=width,
            ordering=spatial_grid.ordering,
        )

    def filled_labels(self) -> np.ndarray:
        """Fill a labels matrix by labeling each rectangle in the grid."""
        i_min = int(self.y_location_of_lowest_y_spot - self.y_spacing / 2)
        j_min = int(self.x_location_of_lowest_x_spot - self.x_spacing / 2)
        return _fill_grid_numba(
            int(self.image_height),
            int(self.image_width),
            float(self.y_spacing),
            float(self.x_spacing),
            i_min,
            j_min,
            np.asarray(self.spot_table, dtype=np.int32),
        )

    def labels_for_shape(self, shape: tuple[int, int]) -> np.ndarray:
        """Return grid labels aligned to the requested output shape."""
        labels = self.filled_labels()
        if labels.shape == shape:
            return labels
        result = np.zeros(
            [max(labels.shape[i], shape[i]) for i in range(2)],
            dtype=np.int32,
        )
        result[0:labels.shape[0], 0:labels.shape[1]] = labels
        return result

    def circle_labels(
        self,
        *,
        center_i: np.ndarray,
        center_j: np.ndarray,
        radius: float,
        guiding_labels: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return labels constrained to circles centered on grid spot IDs."""
        labels = (
            self.labels_for_shape(guiding_labels.shape)
            if guiding_labels is not None
            else self.filled_labels()
        )
        center_i_by_label, center_j_by_label = _spot_center_lookup_numba(
            np.asarray(self.spot_table, dtype=np.int32),
            np.asarray(center_i, dtype=np.float64),
            np.asarray(center_j, dtype=np.float64),
            int(self.spot_table.max()),
        )
        return _apply_circle_mask_numba(
            np.asarray(labels, dtype=np.int32),
            center_i_by_label,
            center_j_by_label,
            float(radius),
        )

    def forced_circle_labels(self, radius: float) -> np.ndarray:
        """Return circular labels centered in each grid cell."""
        row_indices, col_indices = np.mgrid[0:self.rows, 0:self.columns]
        return self.circle_labels(
            center_i=(
                self.y_locations[row_indices, col_indices]
                if self.y_locations.ndim == 2
                else self.y_locations[row_indices]
            ),
            center_j=(
                self.x_locations[row_indices, col_indices]
                if self.x_locations.ndim == 2
                else self.x_locations[col_indices]
            ),
            radius=radius,
        )

    def guide_label_center_grid_ids(
        self,
        guide_labels: np.ndarray,
        *,
        grid_labels: np.ndarray | None = None,
    ) -> np.ndarray:
        """Map each guide label ID to the grid object ID containing its center."""
        labels = self.filled_labels() if grid_labels is None else grid_labels
        max_guide = int(np.max(guide_labels))
        label_center_grid_ids = np.zeros(max_guide + 1, dtype=np.int32)
        if max_guide == 0:
            return label_center_grid_ids

        centers = np.zeros((2, max_guide + 1), dtype=np.float64)
        centers_i, centers_j = _centers_of_labels_numba(
            np.asarray(guide_labels, dtype=np.int32),
            max_guide,
        )
        centers[0, 1:] = centers_i
        centers[1, 1:] = centers_j
        bad_centers = (
            (~np.isfinite(centers[0, :]))
            | (~np.isfinite(centers[1, :]))
            | (centers[0, :] >= labels.shape[0])
            | (centers[1, :] >= labels.shape[1])
        )
        rounded_centers = np.round(centers).astype(int)
        masked_labels = labels.copy()
        y_border = int(np.ceil(self.y_spacing / 10))
        x_border = int(np.ceil(self.x_spacing / 10))
        if y_border > 0:
            ymask = labels[y_border:, :] != labels[:-y_border, :]
            masked_labels[y_border:, :][ymask] = 0
            masked_labels[:-y_border, :][ymask] = 0
        if x_border > 0:
            xmask = labels[:, x_border:] != labels[:, :-x_border]
            masked_labels[:, x_border:][xmask] = 0
            masked_labels[:, :-x_border][xmask] = 0
        rounded_centers[:, bad_centers] = 0
        label_center_grid_ids = masked_labels[
            rounded_centers[0, :],
            rounded_centers[1, :],
        ]
        label_center_grid_ids[bad_centers] = 0
        return np.asarray(label_center_grid_ids, dtype=np.int32)

    def filtered_guides(self, guide_labels: np.ndarray) -> np.ndarray:
        """Filter guide labels to object parts accepted by this grid."""
        labels = self.filled_labels()
        return _filter_labels_by_grid_numba(
            np.asarray(guide_labels, dtype=np.int32),
            np.asarray(labels, dtype=np.int32),
            self.guide_label_center_grid_ids(guide_labels, grid_labels=labels),
        )

    def labels_from_filtered_guides(self, filtered_guides: np.ndarray) -> np.ndarray:
        """Return grid labels masked by accepted guide pixels."""
        labels = self.labels_for_shape(filtered_guides.shape)
        return _mask_grid_labels_by_filtered_guides_numba(
            np.asarray(labels, dtype=np.int32),
            np.asarray(filtered_guides, dtype=np.int32),
        )


@dataclass
class GridObjectStats:
    slice_index: int
    object_count: int
    grid_rows: int
    grid_columns: int
    shape_type: str


@dataclass(frozen=True, slots=True, kw_only=True)
class GridShapeContext(ABC):
    """Shared grid-shape execution state carried through nominal requests."""

    grid: GridDefinition
    guiding_labels: np.ndarray | None = None
    diameter_choice: DiameterChoice = DiameterChoice.MANUAL
    circle_diameter: int = 20


@dataclass(frozen=True, slots=True, kw_only=True)
class GridShapeRequest(GridShapeContext):
    """Inputs needed to materialize one grid object shape strategy."""

    filtered_guides: np.ndarray | None = None

    def labels(self, shape_choice: ShapeChoice) -> np.ndarray:
        """Materialize labels through the registered strategy family."""
        strategy = GridShapeStrategy.for_shape_choice(shape_choice)
        if strategy.requires_guides and self.guiding_labels is None:
            strategy = GridShapeStrategy.for_shape_choice(ShapeChoice.RECTANGLE)
        return strategy.labels(self)

    @property
    def required_guiding_labels(self) -> np.ndarray:
        if self.guiding_labels is None:
            raise ValueError("Grid shape strategy requires guiding labels.")
        return self.guiding_labels

    @property
    def required_filtered_guides(self) -> np.ndarray:
        if self.filtered_guides is None:
            raise ValueError("Grid shape strategy requires filtered guiding labels.")
        return self.filtered_guides

    def circle_radius(self) -> float:
        """Return manual or area-derived circle radius for grid object modes."""
        if self.diameter_choice is DiameterChoice.MANUAL:
            return self.circle_diameter / 2.0
        filtered_guides = self.required_filtered_guides
        areas = np.bincount(filtered_guides[filtered_guides != 0].flatten())
        if len(areas) > 0 and np.any(areas != 0):
            median_area = np.median(areas[areas != 0])
            return max(1, np.sqrt(median_area / np.pi))
        return self.circle_diameter / 2.0


@dataclass(frozen=True, slots=True, kw_only=True)
class IdentifyObjectsInGridRequest(GridShapeContext):
    """Executable request for CellProfiler IdentifyObjectsInGrid semantics."""

    image: np.ndarray
    shape_choice: ShapeChoice

    @classmethod
    def from_runtime(
        cls,
        *,
        image: np.ndarray,
        grid: SpatialGrid | None,
        grid_rows: int,
        grid_columns: int,
        x_spacing: float,
        y_spacing: float,
        x_origin: float,
        y_origin: float,
        shape_choice: ShapeChoice | str,
        diameter_choice: DiameterChoice | str,
        circle_diameter: int,
        guiding_labels: np.ndarray | None = None,
    ) -> "IdentifyObjectsInGridRequest":
        """Bind CP/runtime inputs into one nominal executable request."""
        return cls(
            image=image,
            grid=GridDefinition.from_runtime(
                image_shape=image.shape,
                grid=grid,
                grid_rows=grid_rows,
                grid_columns=grid_columns,
                x_spacing=x_spacing,
                y_spacing=y_spacing,
                x_origin=x_origin,
                y_origin=y_origin,
            ),
            shape_choice=coerce_cellprofiler_enum(ShapeChoice, shape_choice),
            diameter_choice=coerce_cellprofiler_enum(DiameterChoice, diameter_choice),
            circle_diameter=circle_diameter,
            guiding_labels=guiding_labels,
        )

    @property
    def object_count(self) -> int:
        return self.grid.rows * self.grid.columns

    @property
    def filtered_guides(self) -> np.ndarray | None:
        if self.guiding_labels is None:
            return None
        return self.grid.filtered_guides(self.guiding_labels)

    def stats(self) -> GridObjectStats:
        return GridObjectStats(
            slice_index=0,
            object_count=self.object_count,
            grid_rows=self.grid.rows,
            grid_columns=self.grid.columns,
            shape_type=self.shape_choice.value,
        )

    def execute(self) -> Tuple[np.ndarray, GridObjectStats, ObjectLabelPayload]:
        labels = GridShapeRequest(
            grid=self.grid,
            guiding_labels=self.guiding_labels,
            filtered_guides=self.filtered_guides,
            diameter_choice=self.diameter_choice,
            circle_diameter=self.circle_diameter,
        ).labels(self.shape_choice)
        return self.image, self.stats(), object_label_payload_from_source_image(
            self.image,
            labels.astype(np.int32, copy=False),
            declared_object_count=self.object_count,
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
            elif not remove:
                remove = True
            if remove:
                filtered[row, col] = 0
    return filtered


@njit(cache=True)
def _mask_grid_labels_by_filtered_guides_numba(
    grid_labels: np.ndarray,
    filtered_guides: np.ndarray,
) -> np.ndarray:
    """Apply CP run_natural semantics: grid label survives where guide survives."""
    labels = grid_labels.copy()
    height, width = labels.shape
    guide_height, guide_width = filtered_guides.shape
    for row in range(height):
        for col in range(width):
            if row >= guide_height or col >= guide_width or filtered_guides[row, col] == 0:
                labels[row, col] = 0
    return labels


class GridShapeStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal strategy for materializing grid object labels."""

    __registry_key__ = "shape_choice"
    __skip_if_no_key__ = True
    shape_choice: ClassVar[str | None] = None
    requires_guides: ClassVar[bool] = False

    @classmethod
    def for_shape_choice(cls, shape_choice: ShapeChoice | str) -> "GridShapeStrategy":
        resolved = coerce_cellprofiler_enum(ShapeChoice, shape_choice)
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
        return request.grid.filled_labels()


class ForcedCircleGridShapeStrategy(GridShapeStrategy):
    """Draw fixed-diameter circles at grid centers."""

    shape_choice = ShapeChoice.CIRCLE_FORCED.value

    def labels(self, request: GridShapeRequest) -> np.ndarray:
        return request.grid.forced_circle_labels(request.circle_diameter / 2.0)


class NaturalCircleGridShapeStrategy(GridShapeStrategy):
    """Draw automatic circles using accepted guide objects for centers/area."""

    shape_choice = ShapeChoice.CIRCLE_NATURAL.value
    requires_guides = True

    def labels(self, request: GridShapeRequest) -> np.ndarray:
        guiding_labels = request.required_guiding_labels
        filtered_guides = request.required_filtered_guides
        labels = request.grid.filled_labels()
        labels[filtered_guides[0:labels.shape[0], 0:labels.shape[1]] == 0] = 0
        centers_i, centers_j = _centers_of_labels(labels)

        nmissing = np.max(request.grid.spot_table) - len(centers_i)
        if nmissing > 0:
            centers_i = np.hstack((centers_i, [np.nan] * nmissing))
            centers_j = np.hstack((centers_j, [np.nan] * nmissing))

        spot_centers_i = centers_i[request.grid.spot_table - 1]
        spot_centers_j = centers_j[request.grid.spot_table - 1]

        return request.grid.circle_labels(
            center_i=spot_centers_i,
            center_j=spot_centers_j,
            radius=request.circle_radius(),
            guiding_labels=guiding_labels,
        )


class NaturalGridShapeStrategy(GridShapeStrategy):
    """Preserve accepted guide-object shapes and relabel by center grid cell."""

    shape_choice = ShapeChoice.NATURAL.value
    requires_guides = True

    def labels(self, request: GridShapeRequest) -> np.ndarray:
        return request.grid.labels_from_filtered_guides(request.required_filtered_guides)


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
    return IdentifyObjectsInGridRequest.from_runtime(
        image=image,
        grid=grid,
        grid_rows=grid_rows,
        grid_columns=grid_columns,
        x_spacing=x_spacing,
        y_spacing=y_spacing,
        x_origin=x_origin,
        y_origin=y_origin,
        shape_choice=shape_choice,
        diameter_choice=diameter_choice,
        circle_diameter=circle_diameter,
    ).execute()


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
    return IdentifyObjectsInGridRequest.from_runtime(
        image=image,
        grid=grid,
        grid_rows=grid_rows,
        grid_columns=grid_columns,
        x_spacing=x_spacing,
        y_spacing=y_spacing,
        x_origin=x_origin,
        y_origin=y_origin,
        shape_choice=shape_choice,
        diameter_choice=diameter_choice,
        circle_diameter=circle_diameter,
        guiding_labels=guiding_labels,
    ).execute()


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
