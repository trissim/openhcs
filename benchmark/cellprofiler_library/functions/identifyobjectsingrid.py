"""Converted from CellProfiler: IdentifyObjectsInGrid

Identifies objects within each section of a grid pattern.
This module creates labeled objects based on grid definitions,
with options for rectangles, circles, or natural shapes.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_outputs, special_inputs
from openhcs.processing.materialization import csv_materializer
from openhcs.processing.backends.analysis.cell_counting_cpu import materialize_segmentation_masks


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


@dataclass
class GridObjectStats:
    slice_index: int
    object_count: int
    grid_rows: int
    grid_columns: int
    shape_type: str


def _fill_grid(grid: GridDefinition) -> np.ndarray:
    """Fill a labels matrix by labeling each rectangle in the grid."""
    i, j = np.mgrid[0:grid.image_height, 0:grid.image_width]
    i_min = int(grid.y_location_of_lowest_y_spot - grid.y_spacing / 2)
    j_min = int(grid.x_location_of_lowest_x_spot - grid.x_spacing / 2)
    i_idx = np.floor((i - i_min) / grid.y_spacing).astype(int)
    j_idx = np.floor((j - j_min) / grid.x_spacing).astype(int)
    mask = (
        (i_idx >= 0) &
        (j_idx >= 0) &
        (i_idx < grid.spot_table.shape[0]) &
        (j_idx < grid.spot_table.shape[1])
    )
    labels = np.zeros((grid.image_height, grid.image_width), dtype=np.int32)
    labels[mask] = grid.spot_table[i_idx[mask], j_idx[mask]]
    return labels


def _centers_of_labels(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate centers of mass for each label."""
    from scipy.ndimage import center_of_mass
    
    max_label = labels.max()
    if max_label == 0:
        return np.array([]), np.array([])
    
    centers_i = np.zeros(max_label)
    centers_j = np.zeros(max_label)
    
    for label_id in range(1, max_label + 1):
        mask = labels == label_id
        if np.any(mask):
            coords = np.where(mask)
            centers_i[label_id - 1] = np.mean(coords[0])
            centers_j[label_id - 1] = np.mean(coords[1])
        else:
            centers_i[label_id - 1] = np.nan
            centers_j[label_id - 1] = np.nan
    
    return centers_i, centers_j


def _run_rectangle(grid: GridDefinition) -> np.ndarray:
    """Return a labels matrix composed of grid rectangles."""
    return _fill_grid(grid)


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
    
    # Build lookup for spot centers
    spot_center_i_flat = np.zeros(grid.spot_table.max() + 1)
    spot_center_j_flat = np.zeros(grid.spot_table.max() + 1)
    spot_center_i_flat[grid.spot_table.flatten()] = spot_center_i.flatten()
    spot_center_j_flat[grid.spot_table.flatten()] = spot_center_j.flatten()
    
    centers_i = spot_center_i_flat[labels]
    centers_j = spot_center_j_flat[labels]
    i, j = np.mgrid[0:labels.shape[0], 0:labels.shape[1]]
    
    # Create circular mask
    mask = (i - centers_i) ** 2 + (j - centers_j) ** 2 <= (radius + 0.5) ** 2
    labels[~mask] = 0
    
    # Remove labels with invalid centers
    labels[np.isnan(centers_i) | np.isnan(centers_j)] = 0
    
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
    
    centers_i, centers_j = _centers_of_labels(guide_labels)
    max_guide = guide_labels.max()
    
    centers = np.zeros((2, max_guide + 1))
    if len(centers_i) > 0:
        centers[0, 1:len(centers_i)+1] = centers_i
        centers[1, 1:len(centers_j)+1] = centers_j
    
    bad_centers = (
        (~np.isfinite(centers[0, :])) |
        (~np.isfinite(centers[1, :])) |
        (centers[0, :] >= labels.shape[0]) |
        (centers[1, :] >= labels.shape[1])
    )
    centers_int = np.round(centers).astype(int)
    
    masked_labels = labels.copy()
    x_border = int(np.ceil(grid.x_spacing / 10))
    y_border = int(np.ceil(grid.y_spacing / 10))
    
    # Erase border regions
    if y_border > 0 and labels.shape[0] > y_border:
        ymask = labels[y_border:, :] != labels[:-y_border, :]
        masked_labels[y_border:, :][ymask] = 0
        masked_labels[:-y_border, :][ymask] = 0
    
    if x_border > 0 and labels.shape[1] > x_border:
        xmask = labels[:, x_border:] != labels[:, :-x_border]
        masked_labels[:, x_border:][xmask] = 0
        masked_labels[:, :-x_border][xmask] = 0
    
    centers_int[:, bad_centers] = 0
    centers_int[0, :] = np.clip(centers_int[0, :], 0, masked_labels.shape[0] - 1)
    centers_int[1, :] = np.clip(centers_int[1, :], 0, masked_labels.shape[1] - 1)
    
    lcenters = masked_labels[centers_int[0, :], centers_int[1, :]]
    lcenters[bad_centers] = 0
    
    # Filter guide labels
    mask = np.zeros(guide_labels.shape, bool)
    ii_labels = (slice(0, labels.shape[0]), slice(0, labels.shape[1]))
    
    guide_subset = guide_labels[ii_labels]
    mask[ii_labels] = lcenters[guide_subset] != labels
    mask[guide_labels == 0] = True
    mask[lcenters[guide_labels] == 0] = True
    
    filtered = guide_labels.copy()
    filtered[mask] = 0
    return filtered


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(
    ("grid_stats", csv_materializer(
        fields=["slice_index", "object_count", "grid_rows", "grid_columns", "shape_type"],
        analysis_type="grid_objects"
    )),
    ("labels", materialize_segmentation_masks)
)
def identify_objects_in_grid(
    image: np.ndarray,
    grid_rows: int = 8,
    grid_columns: int = 12,
    x_spacing: float = 100.0,
    y_spacing: float = 100.0,
    x_origin: float = 50.0,
    y_origin: float = 50.0,
    shape_choice: ShapeChoice = ShapeChoice.RECTANGLE,
    diameter_choice: DiameterChoice = DiameterChoice.MANUAL,
    circle_diameter: int = 20,
) -> Tuple[np.ndarray, GridObjectStats, np.ndarray]:
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
    height, width = image.shape
    
    # Build grid definition
    i_grid, j_grid = np.mgrid[0:grid_rows, 0:grid_columns]
    y_locations = y_origin + i_grid * y_spacing
    x_locations = x_origin + j_grid * x_spacing
    
    # Create spot table (1-indexed labels)
    spot_table = np.arange(1, grid_rows * grid_columns + 1).reshape(grid_rows, grid_columns)
    
    grid = GridDefinition(
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
        image_width=width
    )
    
    # Generate labels based on shape choice
    if shape_choice == ShapeChoice.RECTANGLE:
        labels = _run_rectangle(grid)
    elif shape_choice == ShapeChoice.CIRCLE_FORCED:
        radius = circle_diameter / 2.0
        labels = _run_forced_circle(grid, radius)
    else:
        # Default to rectangle for unsupported modes without guiding objects
        labels = _run_rectangle(grid)
    
    object_count = grid_rows * grid_columns
    
    stats = GridObjectStats(
        slice_index=0,
        object_count=object_count,
        grid_rows=grid_rows,
        grid_columns=grid_columns,
        shape_type=shape_choice.value
    )
    
    return image, stats, labels.astype(np.int32)


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("guiding_labels")
@special_outputs(
    ("grid_stats", csv_materializer(
        fields=["slice_index", "object_count", "grid_rows", "grid_columns", "shape_type"],
        analysis_type="grid_objects"
    )),
    ("labels", materialize_segmentation_masks)
)
def identify_objects_in_grid_with_guides(
    image: np.ndarray,
    guiding_labels: np.ndarray,
    grid_rows: int = 8,
    grid_columns: int = 12,
    x_spacing: float = 100.0,
    y_spacing: float = 100.0,
    x_origin: float = 50.0,
    y_origin: float = 50.0,
    shape_choice: ShapeChoice = ShapeChoice.CIRCLE_NATURAL,
    diameter_choice: DiameterChoice = DiameterChoice.AUTOMATIC,
    circle_diameter: int = 20,
) -> Tuple[np.ndarray, GridObjectStats, np.ndarray]:
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
    height, width = image.shape
    
    # Build grid definition
    i_grid, j_grid = np.mgrid[0:grid_rows, 0:grid_columns]
    y_locations = y_origin + i_grid * y_spacing
    x_locations = x_origin + j_grid * x_spacing
    spot_table = np.arange(1, grid_rows * grid_columns + 1).reshape(grid_rows, grid_columns)
    
    grid = GridDefinition(
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
        image_width=width
    )
    
    # Filter guiding labels
    filtered_guides = _filter_labels_by_grid(guiding_labels, grid)
    
    if shape_choice == ShapeChoice.CIRCLE_NATURAL:
        # Use guiding object centers for circle placement
        labels = _fill_grid(grid)
        labels[filtered_guides[0:labels.shape[0], 0:labels.shape[1]] == 0] = 0
        centers_i, centers_j = _centers_of_labels(labels)
        
        nmissing = np.max(grid.spot_table) - len(centers_i)
        if nmissing > 0:
            centers_i = np.hstack((centers_i, [np.nan] * nmissing))
            centers_j = np.hstack((centers_j, [np.nan] * nmissing))
        
        spot_centers_i = centers_i[grid.spot_table - 1]
        spot_centers_j = centers_j[grid.spot_table - 1]
        
        # Calculate radius
        if diameter_choice == DiameterChoice.AUTOMATIC:
            areas = np.bincount(filtered_guides[filtered_guides != 0].flatten())
            if len(areas) > 0 and np.any(areas != 0):
                median_area = np.median(areas[areas != 0])
                radius = max(1, np.sqrt(median_area / np.pi))
            else:
                radius = circle_diameter / 2.0
        else:
            radius = circle_diameter / 2.0
        
        labels = _run_circle(grid, spot_centers_i, spot_centers_j, radius, guiding_labels)
        
    elif shape_choice == ShapeChoice.NATURAL:
        # Use natural shape from guiding objects
        labels = _fill_grid(grid)
        
        # Fit to guiding objects size
        if any(guiding_labels.shape[i] > labels.shape[i] for i in range(2)):
            result = np.zeros(
                [max(guiding_labels.shape[i], labels.shape[i]) for i in range(2)],
                dtype=np.int32
            )
            result[0:labels.shape[0], 0:labels.shape[1]] = labels
            labels = result
        
        labels[filtered_guides == 0] = 0
        
    else:
        # Fall back to forced circle
        if diameter_choice == DiameterChoice.AUTOMATIC:
            areas = np.bincount(filtered_guides[filtered_guides != 0].flatten())
            if len(areas) > 0 and np.any(areas != 0):
                median_area = np.median(areas[areas != 0])
                radius = max(1, np.sqrt(median_area / np.pi))
            else:
                radius = circle_diameter / 2.0
        else:
            radius = circle_diameter / 2.0
        
        labels = _run_forced_circle(grid, radius)
    
    object_count = grid_rows * grid_columns
    
    stats = GridObjectStats(
        slice_index=0,
        object_count=object_count,
        grid_rows=grid_rows,
        grid_columns=grid_columns,
        shape_type=shape_choice.value
    )
    
    return image, stats, labels.astype(np.int32)