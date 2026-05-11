"""CellProfiler FlagImage QC semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.materialization import csv_materializer


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


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(
    (
        "flag_results",
        csv_materializer(
            fields=[
                "slice_index",
                "flag_name",
                "flag_value",
                "measurement_name",
                "measurement_value",
                "min_threshold",
                "max_threshold",
                "pass_fail",
            ],
            analysis_type="flag",
        ),
    )
)
def flag_image(
    image: np.ndarray,
    flag_name: str = "QCFlag",
    flag_category: str = "Metadata",
    measurement_value: float | None = None,
    check_minimum: bool = True,
    minimum_value: float = 0.0,
    check_maximum: bool = True,
    maximum_value: float = 1.0,
    combination_choice: CombinationChoice = CombinationChoice.ANY,
) -> tuple[np.ndarray, FlagResult]:
    """Flag an image based on a provided or image-derived measurement."""
    del combination_choice
    if measurement_value is None:
        measurement_value = float(np.mean(image))
    return image, flag_image_result(
        flag_name=flag_name,
        flag_category=flag_category,
        measurement_name="intensity_mean",
        measurement_value=measurement_value,
        check_minimum=check_minimum,
        minimum_value=minimum_value,
        check_maximum=check_maximum,
        maximum_value=maximum_value,
    )


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(
    (
        "flag_results",
        csv_materializer(
            fields=[
                "slice_index",
                "flag_name",
                "flag_value",
                "measurement_name",
                "measurement_value",
                "min_threshold",
                "max_threshold",
                "pass_fail",
            ],
            analysis_type="flag",
        ),
    )
)
def flag_image_intensity(
    image: np.ndarray,
    flag_name: str = "IntensityQC",
    flag_category: str = "Metadata",
    check_minimum: bool = True,
    minimum_value: float = 0.0,
    check_maximum: bool = True,
    maximum_value: float = 1.0,
    use_mean: bool = True,
) -> tuple[np.ndarray, FlagResult]:
    """Flag an image based on mean or median intensity."""
    if use_mean:
        measurement_value = float(np.mean(image))
        measurement_name = "intensity_mean"
    else:
        measurement_value = float(np.median(image))
        measurement_name = "intensity_median"
    return image, flag_image_result(
        flag_name=flag_name,
        flag_category=flag_category,
        measurement_name=measurement_name,
        measurement_value=measurement_value,
        check_minimum=check_minimum,
        minimum_value=minimum_value,
        check_maximum=check_maximum,
        maximum_value=maximum_value,
    )


__all__ = [
    "CombinationChoice",
    "FlagResult",
    "MeasurementSource",
    "flag_image",
    "flag_image_intensity",
    "flag_image_result",
]
