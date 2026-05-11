"""Converted from CellProfiler: IdentifyObjectsInGrid

Identifies objects within each section of a grid pattern.
This module creates labeled objects based on grid definitions,
with options for rectangles, circles, or natural shapes.
"""

import numpy as np
from typing import Tuple

from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_outputs, special_inputs
from openhcs.core.runtime_values import (
    ObjectLabelPayload,
    SpatialGrid,
)
from openhcs.processing.backends.cellprofiler.grid import (
    DiameterChoice,
    GridObjectStats,
    IdentifyObjectsInGridRequest,
    ShapeChoice,
)
from openhcs.processing.materialization import csv_materializer, segmentation_mask_rois


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
