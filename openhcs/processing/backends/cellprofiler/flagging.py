"""FlagImage backend entrypoints for CellProfiler-compatible processing."""

from __future__ import annotations
from openhcs.interop.cellprofiler.module_declarations import (
    ProcessingContract,
    CellProfilerModule,
)
import numpy as np
from openhcs.core.memory.decorators import numpy
from openhcs.core.public_api import public_names_from_objects
from openhcs.interop.cellprofiler.flag_image import (
    CombinationChoice,
    FlagResult,
    MeasurementSource,
    flag_image as _flag_image,
    flag_image_intensity as _flag_image_intensity,
    flag_image_result,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


@numpy(contract=ProcessingContract.PURE_2D)
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
    """Backend-discoverable FlagImage callable backed by interop semantics."""
    return _flag_image(
        image,
        flag_name=flag_name,
        flag_category=flag_category,
        measurement_value=measurement_value,
        check_minimum=check_minimum,
        minimum_value=minimum_value,
        check_maximum=check_maximum,
        maximum_value=maximum_value,
        combination_choice=combination_choice,
    )


@numpy(contract=ProcessingContract.PURE_2D)
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
    """Backend-discoverable intensity FlagImage variant."""
    return _flag_image_intensity(
        image,
        flag_name=flag_name,
        flag_category=flag_category,
        check_minimum=check_minimum,
        minimum_value=minimum_value,
        check_maximum=check_maximum,
        maximum_value=maximum_value,
        use_mean=use_mean,
    )


class FlagImageModule(CellProfilerModule):
    module_name = "FlagImage"
    function_name = "flag_image"
    validated = True
    confidence = 1.0


__all__ = public_names_from_objects(
    CombinationChoice,
    FlagResult,
    MeasurementSource,
    flag_image,
    flag_image_intensity,
    flag_image_result,
)
