"""
Converted from CellProfiler: FlagImage
Original: FlagImage module

Flags images based on measurement criteria for quality control.
The flag value is 1 if the image meets the flagging criteria (fails QC),
and 0 if it does not meet the criteria (passes QC).
"""

import numpy as np
from typing import Tuple, Optional

from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.interop.cellprofiler.flag_image import (
    CombinationChoice,
    FlagResult,
    MeasurementSource,
    flag_image_result,
)
from openhcs.processing.materialization import csv_materializer


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(("flag_results", csv_materializer(
    fields=["slice_index", "flag_name", "flag_value", "measurement_name", 
            "measurement_value", "min_threshold", "max_threshold", "pass_fail"],
    analysis_type="flag"
)))
def flag_image(
    image: np.ndarray,
    flag_name: str = "QCFlag",
    flag_category: str = "Metadata",
    measurement_value: Optional[float] = None,
    check_minimum: bool = True,
    minimum_value: float = 0.0,
    check_maximum: bool = True,
    maximum_value: float = 1.0,
    combination_choice: CombinationChoice = CombinationChoice.ANY,
) -> Tuple[np.ndarray, FlagResult]:
    """
    Flag an image based on measurement criteria.
    
    This function evaluates whether an image should be flagged based on
    measurement thresholds. The flag is set to 1 if the measurement
    falls outside the specified bounds.
    
    Args:
        image: Input image array of shape (H, W)
        flag_name: Name for the flag measurement
        flag_category: Category for the flag (default: Metadata)
        measurement_value: The measurement value to evaluate. If None,
                          uses mean intensity of the image.
        check_minimum: Whether to flag images with values below minimum
        minimum_value: Lower threshold for flagging
        check_maximum: Whether to flag images with values above maximum
        maximum_value: Upper threshold for flagging
        combination_choice: How to combine multiple criteria
    
    Returns:
        Tuple of (original image, FlagResult dataclass)
    """
    # If no measurement value provided, compute mean intensity
    if measurement_value is None:
        measurement_value = float(np.mean(image))
    
    result = flag_image_result(
        flag_name=flag_name,
        flag_category=flag_category,
        measurement_name="intensity_mean",
        measurement_value=measurement_value,
        check_minimum=check_minimum,
        minimum_value=minimum_value,
        check_maximum=check_maximum,
        maximum_value=maximum_value,
    )
    return image, result


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(("flag_results", csv_materializer(
    fields=["slice_index", "flag_name", "flag_value", "measurement_name",
            "measurement_value", "min_threshold", "max_threshold", "pass_fail"],
    analysis_type="flag"
)))
def flag_image_intensity(
    image: np.ndarray,
    flag_name: str = "IntensityQC",
    flag_category: str = "Metadata",
    check_minimum: bool = True,
    minimum_value: float = 0.0,
    check_maximum: bool = True,
    maximum_value: float = 1.0,
    use_mean: bool = True,
) -> Tuple[np.ndarray, FlagResult]:
    """
    Flag an image based on intensity measurements.
    
    Computes intensity statistics from the image and flags based on thresholds.
    
    Args:
        image: Input image array of shape (H, W)
        flag_name: Name for the flag measurement
        flag_category: Category for the flag
        check_minimum: Whether to flag images with values below minimum
        minimum_value: Lower threshold for flagging
        check_maximum: Whether to flag images with values above maximum  
        maximum_value: Upper threshold for flagging
        use_mean: If True, use mean intensity; if False, use median
    
    Returns:
        Tuple of (original image, FlagResult dataclass)
    """
    # Compute intensity measurement
    if use_mean:
        measurement_value = float(np.mean(image))
        measurement_name = "intensity_mean"
    else:
        measurement_value = float(np.median(image))
        measurement_name = "intensity_median"
    
    result = flag_image_result(
        flag_name=flag_name,
        flag_category=flag_category,
        measurement_name=measurement_name,
        measurement_value=measurement_value,
        check_minimum=check_minimum,
        minimum_value=minimum_value,
        check_maximum=check_maximum,
        maximum_value=maximum_value,
    )
    return image, result
