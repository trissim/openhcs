"""Converted from CellProfiler: DefineGrid

DefineGrid produces a grid of desired specifications either manually,
or automatically based on previously identified objects. This module
defines the location of a grid that can be used by modules downstream.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_outputs, special_inputs
from openhcs.processing.materialization import csv_materializer


class GridOrigin(Enum):
    TOP_LEFT = "top_left"
    BOTTOM_LEFT = "bottom_left"
    TOP_RIGHT = "top_right"
    BOTTOM_RIGHT = "bottom_right"


class GridOrdering(Enum):
    BY_ROWS = "rows"
    BY_COLUMNS = "columns"


class GridMode(Enum):
    AUTOMATIC = "automatic"
    MANUAL = "manual"


@dataclass
class GridInfo:
    """Grid definition information."""
    slice_index: int
    rows: int
    columns: int
    x_spacing: float
    y_spacing: float
    x_location_of_lowest_x_spot: float
    y_location_of_lowest_y_spot: float
    total_width: float
    total_height: float


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(
    ("grid_info", csv_materializer(
        fields=["slice_index", "rows", "columns", "x_spacing", "y_spacing",
                "x_location_of_lowest_x_spot", "y_location_of_lowest_y_spot",
                "total_width", "total_height"],
        analysis_type="grid_definition"
    ))
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
    origin: GridOrigin = GridOrigin.TOP_LEFT,
    ordering: GridOrdering = GridOrdering.BY_ROWS,
) -> Tuple[np.ndarray, GridInfo]:
    """Define a grid manually based on two cell coordinates.
    
    Args:
        image: Input image (H, W)
        grid_rows: Number of rows in the grid
        grid_columns: Number of columns in the grid
        first_spot_x: X coordinate of first cell center
        first_spot_y: Y coordinate of first cell center
        first_spot_row: Row number of first cell
        first_spot_col: Column number of first cell
        second_spot_x: X coordinate of second cell center
        second_spot_y: Y coordinate of second cell center
        second_spot_row: Row number of second cell
        second_spot_col: Column number of second cell
        origin: Location of the first spot (numbering origin)
        ordering: Order of spots (by rows or columns)
    
    Returns:
        Tuple of (image, GridInfo)
    """
    # Convert to canonical row/column (0-indexed from top-left)
    def canonical_row_col(row, col):
        if origin in (GridOrigin.BOTTOM_LEFT, GridOrigin.BOTTOM_RIGHT):
            row = grid_rows - row
        else:
            row = row - 1
        if origin in (GridOrigin.TOP_RIGHT, GridOrigin.BOTTOM_RIGHT):
            col = grid_columns - col
        else:
            col = col - 1
        return row, col
    
    first_row_c, first_col_c = canonical_row_col(first_spot_row, first_spot_col)
    second_row_c, second_col_c = canonical_row_col(second_spot_row, second_spot_col)
    
    # Calculate spacing
    if first_col_c == second_col_c:
        x_spacing = 1.0  # Default if same column
    else:
        x_spacing = float(first_spot_x - second_spot_x) / float(first_col_c - second_col_c)
    
    if first_row_c == second_row_c:
        y_spacing = 1.0  # Default if same row
    else:
        y_spacing = float(first_spot_y - second_spot_y) / float(first_row_c - second_row_c)
    
    # Calculate origin location
    x_location_of_lowest_x_spot = first_spot_x - first_col_c * x_spacing
    y_location_of_lowest_y_spot = first_spot_y - first_row_c * y_spacing
    
    # Calculate total dimensions
    total_width = abs(x_spacing) * grid_columns
    total_height = abs(y_spacing) * grid_rows
    
    grid_info = GridInfo(
        slice_index=0,
        rows=grid_rows,
        columns=grid_columns,
        x_spacing=abs(x_spacing),
        y_spacing=abs(y_spacing),
        x_location_of_lowest_x_spot=x_location_of_lowest_x_spot,
        y_location_of_lowest_y_spot=y_location_of_lowest_y_spot,
        total_width=total_width,
        total_height=total_height
    )
    
    return image, grid_info


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(
    ("grid_info", csv_materializer(
        fields=["slice_index", "rows", "columns", "x_spacing", "y_spacing",
                "x_location_of_lowest_x_spot", "y_location_of_lowest_y_spot",
                "total_width", "total_height"],
        analysis_type="grid_definition"
    ))
)
def define_grid_automatic(
    image: np.ndarray,
    labels: np.ndarray,
    grid_rows: int = 8,
    grid_columns: int = 12,
    origin: GridOrigin = GridOrigin.TOP_LEFT,
    ordering: GridOrdering = GridOrdering.BY_ROWS,
) -> Tuple[np.ndarray, GridInfo]:
    """Define a grid automatically based on previously identified objects.
    
    The left-most, right-most, top-most, and bottom-most objects are used
    to define the edges of the grid.
    
    Args:
        image: Input image (H, W)
        labels: Label image from previous segmentation
        grid_rows: Number of rows in the grid
        grid_columns: Number of columns in the grid
        origin: Location of the first spot (numbering origin)
        ordering: Order of spots (by rows or columns)
    
    Returns:
        Tuple of (image, GridInfo)
    """
    from scipy.ndimage import center_of_mass, find_objects
    
    # Find centroids of all labeled objects
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]  # Exclude background
    
    if len(unique_labels) < 2:
        raise ValueError("Need at least 2 objects to define grid automatically")
    
    # Calculate centroids
    centroids = []
    for label_id in unique_labels:
        mask = labels == label_id
        y_coords, x_coords = np.where(mask)
        if len(y_coords) > 0:
            cy = np.mean(y_coords)
            cx = np.mean(x_coords)
            centroids.append((cy, cx))
    
    centroids = np.array(centroids)
    
    # Find extremes
    first_x = np.min(centroids[:, 1])
    first_y = np.min(centroids[:, 0])
    second_x = np.max(centroids[:, 1])
    second_y = np.max(centroids[:, 0])
    
    # Determine row/column assignments based on origin
    if origin in (GridOrigin.BOTTOM_LEFT, GridOrigin.BOTTOM_RIGHT):
        first_row, second_row = grid_rows, 1
    else:
        first_row, second_row = 1, grid_rows
    
    if origin in (GridOrigin.TOP_RIGHT, GridOrigin.BOTTOM_RIGHT):
        first_col, second_col = grid_columns, 1
    else:
        first_col, second_col = 1, grid_columns
    
    # Convert to canonical coordinates
    def canonical_row_col(row, col):
        if origin in (GridOrigin.BOTTOM_LEFT, GridOrigin.BOTTOM_RIGHT):
            row = grid_rows - row
        else:
            row = row - 1
        if origin in (GridOrigin.TOP_RIGHT, GridOrigin.BOTTOM_RIGHT):
            col = grid_columns - col
        else:
            col = col - 1
        return row, col
    
    first_row_c, first_col_c = canonical_row_col(first_row, first_col)
    second_row_c, second_col_c = canonical_row_col(second_row, second_col)
    
    # Calculate spacing
    if first_col_c != second_col_c:
        x_spacing = float(first_x - second_x) / float(first_col_c - second_col_c)
    else:
        x_spacing = (second_x - first_x) / max(grid_columns - 1, 1)
    
    if first_row_c != second_row_c:
        y_spacing = float(first_y - second_y) / float(first_row_c - second_row_c)
    else:
        y_spacing = (second_y - first_y) / max(grid_rows - 1, 1)
    
    # Calculate origin location
    x_location_of_lowest_x_spot = first_x - first_col_c * x_spacing
    y_location_of_lowest_y_spot = first_y - first_row_c * y_spacing
    
    # Calculate total dimensions
    total_width = abs(x_spacing) * grid_columns
    total_height = abs(y_spacing) * grid_rows
    
    grid_info = GridInfo(
        slice_index=0,
        rows=grid_rows,
        columns=grid_columns,
        x_spacing=abs(x_spacing),
        y_spacing=abs(y_spacing),
        x_location_of_lowest_x_spot=x_location_of_lowest_x_spot,
        y_location_of_lowest_y_spot=y_location_of_lowest_y_spot,
        total_width=total_width,
        total_height=total_height
    )
    
    return image, grid_info


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
    """Draw grid lines on an image.
    
    Args:
        image: Input image (H, W)
        grid_rows: Number of rows in the grid
        grid_columns: Number of columns in the grid
        x_spacing: Horizontal spacing between grid cells
        y_spacing: Vertical spacing between grid cells
        x_origin: X coordinate of grid origin
        y_origin: Y coordinate of grid origin
        line_width: Width of grid lines in pixels
    
    Returns:
        Image with grid overlay
    """
    result = image.copy().astype(np.float32)
    h, w = result.shape
    
    # Normalize to 0-1 if needed
    if result.max() > 1.0:
        result = result / result.max()
    
    # Calculate line positions
    line_left_x = int(x_origin - x_spacing / 2)
    line_top_y = int(y_origin - y_spacing / 2)
    
    # Draw vertical lines
    for i in range(grid_columns + 1):
        x = int(line_left_x + i * x_spacing)
        if 0 <= x < w:
            y_start = max(0, line_top_y)
            y_end = min(h, int(line_top_y + grid_rows * y_spacing))
            for dx in range(-line_width // 2, line_width // 2 + 1):
                if 0 <= x + dx < w:
                    result[y_start:y_end, x + dx] = 1.0
    
    # Draw horizontal lines
    for i in range(grid_rows + 1):
        y = int(line_top_y + i * y_spacing)
        if 0 <= y < h:
            x_start = max(0, line_left_x)
            x_end = min(w, int(line_left_x + grid_columns * x_spacing))
            for dy in range(-line_width // 2, line_width // 2 + 1):
                if 0 <= y + dy < h:
                    result[y + dy, x_start:x_end] = 1.0
    
    return result