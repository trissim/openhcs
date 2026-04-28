"""
Converted from CellProfiler: FlagImage
Original: FlagImage module

Flags images based on measurement criteria for quality control.
The flag value is 1 if the image meets the flagging criteria (fails QC),
and 0 if it does not meet the criteria (passes QC).
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.processing.materialization import csv_materializer


class CombinationChoice(Enum):
    ANY = "any"  # Flag if any measurement fails
    ALL = "all"  # Flag if all measurements fail


class MeasurementSource(Enum):
    IMAGE = "image"  # Whole-image measurement
    AVERAGE_OBJECT = "average_object"  # Average measurement for all objects
    ALL_OBJECTS = "all_objects"  # Measurements for all objects


@dataclass
class FlagResult:
    """Result of flag evaluation for an image."""
    slice_index: int
    flag_name: str
    flag_value: int  # 0 = pass, 1 = fail
    measurement_name: str
    measurement_value: float
    min_threshold: float
    max_threshold: float
    pass_fail: str


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
    
    # Evaluate flag conditions
    fail = False
    
    # Check if value is NaN - don't flag NaN values
    if np.isnan(measurement_value):
        fail = False
    else:
        # Check minimum threshold
        if check_minimum and measurement_value < minimum_value:
            fail = True
        
        # Check maximum threshold
        if check_maximum and measurement_value > maximum_value:
            fail = True
    
    # Flag value: 1 = fail (flagged), 0 = pass (not flagged)
    flag_value = 1 if fail else 0
    pass_fail = "Fail" if fail else "Pass"
    
    full_flag_name = f"{flag_category}_{flag_name}"
    
    result = FlagResult(
        slice_index=0,
        flag_name=full_flag_name,
        flag_value=flag_value,
        measurement_name="intensity_mean",
        measurement_value=float(measurement_value),
        min_threshold=minimum_value if check_minimum else float('nan'),
        max_threshold=maximum_value if check_maximum else float('nan'),
        pass_fail=pass_fail
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
    
    # Evaluate flag conditions
    fail = False
    
    if not np.isnan(measurement_value):
        if check_minimum and measurement_value < minimum_value:
            fail = True
        if check_maximum and measurement_value > maximum_value:
            fail = True
    
    flag_value = 1 if fail else 0
    pass_fail = "Fail" if fail else "Pass"
    
    full_flag_name = f"{flag_category}_{flag_name}"
    
    result = FlagResult(
        slice_index=0,
        flag_name=full_flag_name,
        flag_value=flag_value,
        measurement_name=measurement_name,
        measurement_value=measurement_value,
        min_threshold=minimum_value if check_minimum else float('nan'),
        max_threshold=maximum_value if check_maximum else float('nan'),
        pass_fail=pass_fail
    )
    
    return image, result