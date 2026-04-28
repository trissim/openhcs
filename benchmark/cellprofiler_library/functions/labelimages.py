"""
Converted from CellProfiler: LabelImages
Original: LabelImages.run

Assigns plate metadata (plate, well, row, column, site) to image sets
based on the order in which they are processed.
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.processing.materialization import csv_materializer


class ImageOrder(Enum):
    ROW = "row"
    COLUMN = "column"


@dataclass
class PlateMetadata:
    """Plate metadata for an image set."""
    image_set_number: int
    site: int
    row: str
    column: int
    well: str
    plate: int


def _calculate_row_digits(row_count: int) -> int:
    """Calculate the number of letters needed to represent a row."""
    return int(1 + np.log(max(1, row_count)) / np.log(26))


def _calculate_column_digits(column_count: int) -> int:
    """Calculate the number of digits needed to represent a column."""
    return int(1 + np.log10(max(1, column_count)))


def _row_index_to_text(row_index: int, row_digits: int) -> str:
    """Convert a row index to letter representation (A, B, ..., Z, AA, AB, ...)."""
    row_text_indexes = [
        x % 26
        for x in reversed(
            [int(row_index / (26 ** i)) for i in range(row_digits)]
        )
    ]
    row_text = ["ABCDEFGHIJKLMNOPQRSTUVWXYZ"[x] for x in row_text_indexes]
    return reduce(lambda x, y: x + y, row_text)


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
    # Calculate indices from image set number
    well_count, site_index = divmod(image_set_number - 1, site_count)
    
    if order == ImageOrder.ROW:
        row_count_calc, column_index = divmod(well_count, column_count)
        plate_index, row_index = divmod(row_count_calc, row_count)
    else:  # COLUMN order
        column_count_calc, row_index = divmod(well_count, row_count)
        plate_index, column_index = divmod(column_count_calc, column_count)
    
    # Calculate row text (A, B, ..., Z, AA, AB, ...)
    row_digits = _calculate_row_digits(row_count)
    column_digits = _calculate_column_digits(column_count)
    
    row_text = _row_index_to_text(row_index, row_digits)
    
    # Format well name
    well_template = "%s%0" + str(column_digits) + "d"
    well = well_template % (row_text, column_index + 1)
    
    metadata = PlateMetadata(
        image_set_number=image_set_number,
        site=site_index + 1,
        row=row_text,
        column=column_index + 1,
        well=well,
        plate=plate_index + 1
    )
    
    return image, metadata