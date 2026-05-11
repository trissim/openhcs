"""
Converted from CellProfiler: LabelImages
Original: LabelImages.run

Assigns plate metadata (plate, well, row, column, site) to image sets
based on the order in which they are processed.
"""

import numpy as np
from typing import Tuple

from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.interop.cellprofiler.plate_metadata import (
    ImageOrder,
    PlateMetadata,
    label_images_plate_metadata,
)
from openhcs.processing.materialization import csv_materializer


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(("plate_metadata", csv_materializer(
    fields=["image_set_number", "site", "row", "column", "well", "plate"],
    analysis_type="plate_metadata"
)))
def label_images(
    image: np.ndarray,
    image_set_number: int = 1,
    site_count: int = 1,
    column_count: int = 12,
    row_count: int = 8,
    order: ImageOrder = ImageOrder.ROW,
) -> Tuple[np.ndarray, PlateMetadata]:
    """
    Assign plate metadata to image sets based on processing order.
    
    This function calculates plate, well, row, column, and site metadata
    based on the image set number and plate layout parameters.
    
    Args:
        image: Input image array of shape (H, W). Passed through unchanged.
        image_set_number: The 1-based index of the current image set.
        site_count: Number of image sites (fields of view) per well.
        column_count: Number of columns per plate.
        row_count: Number of rows per plate.
        order: Order of image data - ROW (A01, A02, ...) or COLUMN (A01, B01, ...).
    
    Returns:
        Tuple of:
            - Original image (unchanged)
            - PlateMetadata dataclass with plate, well, row, column, site info
    
    Measurements produced:
        - site: Site number within the well (1-based)
        - row: Row name (A, B, C, ...)
        - column: Column number (1-based)
        - well: Well name (e.g., A01, B12)
        - plate: Plate number (1-based)
    """
    metadata = label_images_plate_metadata(
        image_set_number=image_set_number,
        site_count=site_count,
        column_count=column_count,
        row_count=row_count,
        order=order,
    )
    return image, metadata
