"""CellProfiler LabelImages plate metadata semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import reduce

import numpy as np


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


def label_images_plate_metadata(
    *,
    image_set_number: int,
    site_count: int,
    column_count: int,
    row_count: int,
    order: ImageOrder,
) -> PlateMetadata:
    """Return CellProfiler LabelImages plate/well metadata."""
    well_count, site_index = divmod(image_set_number - 1, site_count)

    if order is ImageOrder.ROW:
        row_count_calc, column_index = divmod(well_count, column_count)
        plate_index, row_index = divmod(row_count_calc, row_count)
    else:
        column_count_calc, row_index = divmod(well_count, row_count)
        plate_index, column_index = divmod(column_count_calc, column_count)

    row_digits = row_digits_for_count(row_count)
    column_digits = column_digits_for_count(column_count)
    row_text = row_index_to_text(row_index, row_digits)
    well_template = "%s%0" + str(column_digits) + "d"
    well = well_template % (row_text, column_index + 1)

    return PlateMetadata(
        image_set_number=image_set_number,
        site=site_index + 1,
        row=row_text,
        column=column_index + 1,
        well=well,
        plate=plate_index + 1,
    )


def row_digits_for_count(row_count: int) -> int:
    """Calculate the number of letters needed to represent a row."""
    return int(1 + np.log(max(1, row_count)) / np.log(26))


def column_digits_for_count(column_count: int) -> int:
    """Calculate the number of digits needed to represent a column."""
    return int(1 + np.log10(max(1, column_count)))


def row_index_to_text(row_index: int, row_digits: int) -> str:
    """Convert a row index to CellProfiler row text."""
    row_text_indexes = [
        x % 26
        for x in reversed(
            [int(row_index / (26 ** i)) for i in range(row_digits)]
        )
    ]
    row_text = ["ABCDEFGHIJKLMNOPQRSTUVWXYZ"[x] for x in row_text_indexes]
    return reduce(lambda x, y: x + y, row_text)
