"""CellProfiler FlagImage QC semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class CombinationChoice(Enum):
    ANY = "any"
    ALL = "all"


class MeasurementSource(Enum):
    IMAGE = "image"
    AVERAGE_OBJECT = "average_object"
    ALL_OBJECTS = "all_objects"


@dataclass
class FlagResult:
    """Result of flag evaluation for an image."""

    slice_index: int
    flag_name: str
    flag_value: int
    measurement_name: str
    measurement_value: float
    min_threshold: float
    max_threshold: float
    pass_fail: str


def flag_image_result(
    *,
    flag_name: str,
    flag_category: str,
    measurement_name: str,
    measurement_value: float,
    check_minimum: bool,
    minimum_value: float,
    check_maximum: bool,
    maximum_value: float,
) -> FlagResult:
    """Return CellProfiler-compatible FlagImage row semantics."""
    fail = False
    if not np.isnan(measurement_value):
        if check_minimum and measurement_value < minimum_value:
            fail = True
        if check_maximum and measurement_value > maximum_value:
            fail = True

    flag_value = 1 if fail else 0
    return FlagResult(
        slice_index=0,
        flag_name=f"{flag_category}_{flag_name}",
        flag_value=flag_value,
        measurement_name=measurement_name,
        measurement_value=float(measurement_value),
        min_threshold=minimum_value if check_minimum else float("nan"),
        max_threshold=maximum_value if check_maximum else float("nan"),
        pass_fail="Fail" if fail else "Pass",
    )
