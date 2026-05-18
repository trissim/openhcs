"""Grid-label backends for CellProfiler-compatible processing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit

from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.core.runtime_semantics import SpatialGridOrdering, SpatialGridOrigin
from openhcs.core.runtime_values import (
    ObjectLabelPayload,
    SpatialGrid,
    object_label_dense_array,
    object_label_payload_from_source_image,
)
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.materialization import csv_materializer, segmentation_mask_rois

GridInfo = SpatialGrid


def label_centroid_extremes(
    labels: np.ndarray,
) -> tuple[int, float, float, float, float]:
    """Return object count and min/max centroid coordinates for dense labels."""
    label_array = np.ascontiguousarray(labels, dtype=np.int32)
    if label_array.ndim != 2:
        raise ValueError(f"Automatic grid labels must be 2D, got {label_array.ndim}D.")
    return _label_centroid_extremes_numba(label_array)


@njit(cache=True)
def _label_centroid_extremes_numba(
    labels: np.ndarray,
) -> tuple[int, float, float, float, float]:
    max_label = 0
    height, width = labels.shape
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label > max_label:
                max_label = label

    if max_label <= 0:
        return 0, 0.0, 0.0, 0.0, 0.0

    counts = np.zeros(max_label + 1, dtype=np.int64)
    y_sums = np.zeros(max_label + 1, dtype=np.float64)
    x_sums = np.zeros(max_label + 1, dtype=np.float64)
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label <= 0:
                continue
            counts[label] += 1
            y_sums[label] += float(y)
            x_sums[label] += float(x)

    object_count = 0
    first_y = np.inf
    first_x = np.inf
    second_y = -np.inf
    second_x = -np.inf
    for label in range(1, max_label + 1):
        count = counts[label]
        if count == 0:
            continue
        object_count += 1
        centroid_y = y_sums[label] / count
        centroid_x = x_sums[label] / count
        if centroid_y < first_y:
            first_y = centroid_y
        if centroid_y > second_y:
            second_y = centroid_y
        if centroid_x < first_x:
            first_x = centroid_x
        if centroid_x > second_x:
            second_x = centroid_x

    if object_count == 0:
        return 0, 0.0, 0.0, 0.0, 0.0
    return object_count, first_y, first_x, second_y, second_x


class ShapeChoice(Enum):
    RECTANGLE = "rectangle_forced_location"
    CIRCLE_FORCED = "circle_forced_location"
    CIRCLE_NATURAL = "circle_natural_location"
    NATURAL = "natural_shape_and_location"


class DiameterChoice(Enum):
    AUTOMATIC = "automatic"
    MANUAL = "manual"


@dataclass(frozen=True, slots=True)
class GridSpotReference:
    """One user-selected spot in CellProfiler DefineGrid coordinates."""

    x: float
    y: float
    row: int
    column: int


@dataclass(frozen=True, slots=True)
class SpatialGridDefinitionBase:
    """Shared CellProfiler DefineGrid coordinate policy."""

    rows: int
    columns: int
    origin: SpatialGridOrigin
    ordering: SpatialGridOrdering
    image_shape_yx: tuple[int, int]

    def canonical_row_index(self, row: int) -> int:
        if self.origin.reverses_rows:
            return self.rows - row
        return row - 1

    def canonical_column_index(self, column: int) -> int:
        if self.origin.reverses_columns:
            return self.columns - column
        return column - 1

    def canonical_row_col(self, row: int, column: int) -> tuple[int, int]:
        return self.canonical_row_index(row), self.canonical_column_index(column)


@dataclass(frozen=True, slots=True)
class SpatialGridManualDefinition(SpatialGridDefinitionBase):
    """Manual two-spot CellProfiler DefineGrid geometry policy."""

    first_spot: GridSpotReference
    second_spot: GridSpotReference

    def spatial_grid(self) -> SpatialGrid:
        first_row, first_column = self.canonical_row_col(
            self.first_spot.row,
            self.first_spot.column,
        )
        second_row, second_column = self.canonical_row_col(
            self.second_spot.row,
            self.second_spot.column,
        )
        x_spacing = (
            1.0
            if first_column == second_column
            else float(self.first_spot.x - self.second_spot.x)
            / float(first_column - second_column)
        )
        y_spacing = (
            1.0
            if first_row == second_row
            else float(self.first_spot.y - self.second_spot.y)
            / float(first_row - second_row)
        )
        x_origin = int(self.first_spot.x - first_column * x_spacing)
        y_origin = int(self.first_spot.y - first_row * y_spacing)
        return spatial_grid_from_spacing(
            rows=self.rows,
            columns=self.columns,
            x_spacing=abs(x_spacing),
            y_spacing=abs(y_spacing),
            x_origin=x_origin,
            y_origin=y_origin,
            origin=self.origin,
            ordering=self.ordering,
            image_shape_yx=self.image_shape_yx,
        )


@dataclass(frozen=True, slots=True)
class SpatialGridAutomaticDefinition(SpatialGridDefinitionBase):
    """Automatic CellProfiler DefineGrid geometry policy from object extrema."""

    labels: np.ndarray

    def spatial_grid(self) -> SpatialGrid:
        object_count, first_y, first_x, second_y, second_x = label_centroid_extremes(
            object_label_dense_array(self.labels, dtype=np.int32)
        )
        if object_count < 2:
            raise ValueError("Need at least 2 objects to define grid automatically.")

        first_row, second_row = (
            (self.rows, 1)
            if self.origin.reverses_rows
            else (1, self.rows)
        )
        first_column, second_column = (
            (self.columns, 1)
            if self.origin.reverses_columns
            else (1, self.columns)
        )
        manual_definition = SpatialGridManualDefinition(
            rows=self.rows,
            columns=self.columns,
            first_spot=GridSpotReference(first_x, first_y, first_row, first_column),
            second_spot=GridSpotReference(
                second_x,
                second_y,
                second_row,
                second_column,
            ),
            origin=self.origin,
            ordering=self.ordering,
            image_shape_yx=self.image_shape_yx,
        )
        first_row_c, first_col_c = manual_definition.canonical_row_col(
            first_row,
            first_column,
        )
        second_row_c, second_col_c = manual_definition.canonical_row_col(
            second_row,
            second_column,
        )
        if first_col_c != second_col_c:
            x_spacing = float(first_x - second_x) / float(first_col_c - second_col_c)
        else:
            x_spacing = (second_x - first_x) / max(self.columns - 1, 1)
        if first_row_c != second_row_c:
            y_spacing = float(first_y - second_y) / float(first_row_c - second_row_c)
        else:
            y_spacing = (second_y - first_y) / max(self.rows - 1, 1)
        return spatial_grid_from_spacing(
            rows=self.rows,
            columns=self.columns,
            x_spacing=abs(x_spacing),
            y_spacing=abs(y_spacing),
            x_origin=int(np.floor(first_x - first_col_c * x_spacing)),
            y_origin=int(np.floor(first_y - first_row_c * y_spacing)),
            origin=self.origin,
            ordering=self.ordering,
            image_shape_yx=self.image_shape_yx,
        )


def spatial_grid_from_spacing(
    *,
    rows: int,
    columns: int,
    x_spacing: float,
    y_spacing: float,
    x_origin: float,
    y_origin: float,
    origin: SpatialGridOrigin,
    ordering: SpatialGridOrdering,
    image_shape_yx: tuple[int, int],
) -> SpatialGrid:
    """Build an OpenHCS SpatialGrid from CP DefineGrid spacing fields."""
    total_width = int(abs(x_spacing) * columns)
    total_height = int(abs(y_spacing) * rows)
    return SpatialGrid(
        name="grid_info",
        rows=rows,
        columns=columns,
        x_spacing=abs(x_spacing),
        y_spacing=abs(y_spacing),
        x_origin=int(x_origin),
        y_origin=int(y_origin),
        total_width=total_width,
        total_height=total_height,
        origin=origin,
        ordering=ordering,
        x_locations=tuple(float(int(x_origin + index * abs(x_spacing))) for index in range(columns)),
        y_locations=tuple(float(int(y_origin + index * abs(y_spacing))) for index in range(rows)),
        source_spatial_shape_yx=image_shape_yx,
    )


@dataclass
class GridDefinition:
    """Executable grid geometry derived from OpenHCS spatial-grid artifacts."""

    rows: int
    columns: int
    x_spacing: float
    y_spacing: float
    x_location_of_lowest_x_spot: float
    y_location_of_lowest_y_spot: float
    x_locations: np.ndarray
    y_locations: np.ndarray
    spot_table: np.ndarray
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
        """Relabel accepted guide objects by the grid cell containing their center."""
        labels = self.filled_labels()
        return _filter_labels_by_grid_numba(
            np.asarray(guide_labels, dtype=np.int32),
            self.guide_label_center_grid_ids(guide_labels, grid_labels=labels),
        )

    def labels_from_filtered_guides(self, filtered_guides: np.ndarray) -> np.ndarray:
        """Return accepted guide shapes already relabeled by center grid cell."""
        labels = self.labels_for_shape(filtered_guides.shape)
        result = np.zeros_like(labels)
        result[0 : filtered_guides.shape[0], 0 : filtered_guides.shape[1]] = (
            filtered_guides
        )
        return result


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

    def execute(self) -> tuple[np.ndarray, GridObjectStats, ObjectLabelPayload]:
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


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(
    (
        "grid_info",
        csv_materializer(
            fields=[
                "slice_index",
                "rows",
                "columns",
                "x_spacing",
                "y_spacing",
                "x_location_of_lowest_x_spot",
                "y_location_of_lowest_y_spot",
                "total_width",
                "total_height",
            ],
            analysis_type="grid_definition",
        ),
    )
)
def define_grid_manual(
    image: np.ndarray,
    grid_rows: int = 8,
    grid_columns: int = 12,
    first_spot_x: int = 100,
    first_spot_y: int = 100,
    first_spot_row: int = 1,
    first_spot_col: int = 1,
    second_spot_x: int = 200,
    second_spot_y: int = 200,
    second_spot_row: int = 8,
    second_spot_col: int = 12,
    origin: SpatialGridOrigin = SpatialGridOrigin.TOP_LEFT,
    ordering: SpatialGridOrdering = SpatialGridOrdering.BY_ROWS,
) -> tuple[np.ndarray, SpatialGrid]:
    """Define a CellProfiler grid manually from two spot references."""
    grid = SpatialGridManualDefinition(
        rows=grid_rows,
        columns=grid_columns,
        first_spot=GridSpotReference(
            first_spot_x,
            first_spot_y,
            first_spot_row,
            first_spot_col,
        ),
        second_spot=GridSpotReference(
            second_spot_x,
            second_spot_y,
            second_spot_row,
            second_spot_col,
        ),
        origin=coerce_cellprofiler_enum(SpatialGridOrigin, origin),
        ordering=coerce_cellprofiler_enum(SpatialGridOrdering, ordering),
        image_shape_yx=tuple(int(value) for value in image.shape[-2:]),
    ).spatial_grid()
    return image, grid


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(
    (
        "grid_info",
        csv_materializer(
            fields=[
                "slice_index",
                "rows",
                "columns",
                "x_spacing",
                "y_spacing",
                "x_location_of_lowest_x_spot",
                "y_location_of_lowest_y_spot",
                "total_width",
                "total_height",
            ],
            analysis_type="grid_definition",
        ),
    )
)
def define_grid_automatic(
    image: np.ndarray,
    labels: np.ndarray,
    grid_rows: int = 8,
    grid_columns: int = 12,
    origin: SpatialGridOrigin = SpatialGridOrigin.TOP_LEFT,
    ordering: SpatialGridOrdering = SpatialGridOrdering.BY_ROWS,
) -> tuple[np.ndarray, SpatialGrid]:
    """Define a CellProfiler grid from object-label centroid extrema."""
    grid = SpatialGridAutomaticDefinition(
        rows=grid_rows,
        columns=grid_columns,
        labels=labels,
        origin=coerce_cellprofiler_enum(SpatialGridOrigin, origin),
        ordering=coerce_cellprofiler_enum(SpatialGridOrdering, ordering),
        image_shape_yx=tuple(int(value) for value in image.shape[-2:]),
    ).spatial_grid()
    return image, grid


@numpy(contract=ProcessingContract.PURE_2D)
def draw_grid_overlay(
    image: np.ndarray,
    grid_rows: int = 8,
    grid_columns: int = 12,
    x_spacing: float = 50.0,
    y_spacing: float = 50.0,
    x_origin: float = 25.0,
    y_origin: float = 25.0,
    line_width: int = 1,
) -> np.ndarray:
    """Draw grid lines on an image plane."""
    result = image.copy().astype(np.float32)
    height, width = result.shape
    if result.max() > 1.0:
        result = result / result.max()

    line_left_x = int(x_origin - x_spacing / 2)
    line_top_y = int(y_origin - y_spacing / 2)
    for index in range(grid_columns + 1):
        x = int(line_left_x + index * x_spacing)
        if 0 <= x < width:
            y_start = max(0, line_top_y)
            y_end = min(height, int(line_top_y + grid_rows * y_spacing))
            for dx in range(-line_width // 2, line_width // 2 + 1):
                if 0 <= x + dx < width:
                    result[y_start:y_end, x + dx] = 1.0
    for index in range(grid_rows + 1):
        y = int(line_top_y + index * y_spacing)
        if 0 <= y < height:
            x_start = max(0, line_left_x)
            x_end = min(width, int(line_left_x + grid_columns * x_spacing))
            for dy in range(-line_width // 2, line_width // 2 + 1):
                if 0 <= y + dy < height:
                    result[y + dy, x_start:x_end] = 1.0
    return result


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("grid")
@special_outputs(
    (
        "grid_stats",
        csv_materializer(
            fields=[
                "slice_index",
                "object_count",
                "grid_rows",
                "grid_columns",
                "shape_type",
            ],
            analysis_type="grid_objects",
        ),
    ),
    ("labels", segmentation_mask_rois()),
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
) -> tuple[np.ndarray, GridObjectStats, ObjectLabelPayload]:
    """Identify objects within each section of a grid pattern."""
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
    (
        "grid_stats",
        csv_materializer(
            fields=[
                "slice_index",
                "object_count",
                "grid_rows",
                "grid_columns",
                "shape_type",
            ],
            analysis_type="grid_objects",
        ),
    ),
    ("labels", segmentation_mask_rois()),
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
) -> tuple[np.ndarray, GridObjectStats, ObjectLabelPayload]:
    """Identify grid objects using guiding objects for shape/location."""
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


def prepare_identify_objects_in_grid() -> None:
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


identify_objects_in_grid.__openhcs_prepare__ = prepare_identify_objects_in_grid
identify_objects_in_grid_with_guides.__openhcs_prepare__ = (
    prepare_identify_objects_in_grid
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


def centers_of_labels(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    label_center_grid_ids: np.ndarray,
) -> np.ndarray:
    filtered = np.zeros_like(guide_labels)
    guide_height, guide_width = guide_labels.shape
    for row in range(guide_height):
        for col in range(guide_width):
            guide_id = int(guide_labels[row, col])
            if guide_id <= 0 or guide_id >= len(label_center_grid_ids):
                continue
            center_grid_id = int(label_center_grid_ids[guide_id])
            if center_grid_id > 0:
                filtered[row, col] = center_grid_id
    return filtered


@njit(cache=True)
def _mask_grid_labels_by_filtered_guides_numba(
    grid_labels: np.ndarray,
    filtered_guides: np.ndarray,
) -> np.ndarray:
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
        centers_i, centers_j = centers_of_labels(labels)

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


__all__ = [
    "DiameterChoice",
    "GridDefinition",
    "GridInfo",
    "GridObjectStats",
    "GridShapeContext",
    "GridShapeRequest",
    "GridShapeStrategy",
    "GridSpotReference",
    "IdentifyObjectsInGridRequest",
    "ShapeChoice",
    "SpatialGridAutomaticDefinition",
    "SpatialGridManualDefinition",
    "centers_of_labels",
    "define_grid_automatic",
    "define_grid_manual",
    "draw_grid_overlay",
    "identify_objects_in_grid",
    "identify_objects_in_grid_with_guides",
    "label_centroid_extremes",
    "prepare_identify_objects_in_grid",
    "spatial_grid_from_spacing",
]
